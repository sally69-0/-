"""
database.py
============
وحدة الذاكرة الدائمة (Persistent Storage & Vector DB)

تجمع بين:
- SQLite: لتخزين البيانات المنظّمة (نتائج التحاليل، الجلسات، القياسات الرقمية)
- ChromaDB: لتخزين "المعنى" الدلالي (embeddings) للنصوص والملاحظات، بحيث
  يمكن للنظام لاحقاً البحث عن "عينات سابقة تشبه هذه الحالة" (Long-term Memory)

كل الملفات تُخزَّن محلياً داخل مجلد data/ ولا تُرسل لأي خادم خارجي.
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DB_DIR, exist_ok=True)

SQLITE_PATH = os.path.join(DB_DIR, "ckd_saliva.db")
CHROMA_PATH = os.path.join(DB_DIR, "chroma_store")


class LocalMemory:
    """واجهة موحّدة للتعامل مع SQLite + ChromaDB كذاكرة دائمة للتطبيق."""

    def __init__(self):
        self._init_sqlite()
        self._init_chroma()

    # ------------------------------------------------------------------ #
    # SQLite: بيانات منظمة (عينات، مجموعات، نتائج رقمية)
    # ------------------------------------------------------------------ #
    def _init_sqlite(self):
        self.conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                created_at TEXT,
                group_label TEXT,        -- 'affected' | 'at_risk' | 'healthy'
                patient_note TEXT,
                image_path TEXT,
                chemical_data_json TEXT, -- قيم القياسات الكيميائية كـ JSON
                ai_diagnosis TEXT,
                ai_confidence REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,      -- 'user' | 'assistant'
                content TEXT,
                created_at TEXT,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
            )
        """)
        self.conn.commit()

    def save_sample(self, group_label, patient_note, image_path,
                     chemical_data: dict, ai_diagnosis="", ai_confidence=0.0):
        sample_id = str(uuid.uuid4())
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO samples (sample_id, created_at, group_label, patient_note,
                                  image_path, chemical_data_json, ai_diagnosis, ai_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_id, datetime.now().isoformat(), group_label, patient_note,
            image_path, json.dumps(chemical_data, ensure_ascii=False),
            ai_diagnosis, ai_confidence
        ))
        self.conn.commit()

        # نخزّن أيضاً نسخة دلالية في ChromaDB لتُستخدم كذاكرة طويلة الأمد
        text_repr = f"المجموعة: {group_label}. ملاحظة: {patient_note}. " \
                    f"القياسات: {json.dumps(chemical_data, ensure_ascii=False)}. " \
                    f"التشخيص: {ai_diagnosis}"
        self.add_memory(text_repr, metadata={
            "sample_id": sample_id,
            "group_label": group_label,
            "created_at": datetime.now().isoformat()
        })
        return sample_id

    def get_all_samples(self, group_label=None):
        cur = self.conn.cursor()
        if group_label:
            cur.execute("SELECT * FROM samples WHERE group_label=? ORDER BY created_at DESC",
                        (group_label,))
        else:
            cur.execute("SELECT * FROM samples ORDER BY created_at DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # جلسات المحادثة المتعددة (Multi-Chat Tabs)
    # ------------------------------------------------------------------ #
    def create_session(self, title="محادثة جديدة"):
        session_id = str(uuid.uuid4())
        cur = self.conn.cursor()
        cur.execute("INSERT INTO chat_sessions (session_id, title, created_at) VALUES (?,?,?)",
                    (session_id, title, datetime.now().isoformat()))
        self.conn.commit()
        return session_id

    def list_sessions(self):
        cur = self.conn.cursor()
        cur.execute("SELECT session_id, title, created_at FROM chat_sessions ORDER BY created_at")
        return cur.fetchall()

    def add_message(self, session_id, role, content):
        cur = self.conn.cursor()
        cur.execute("""INSERT INTO chat_messages (session_id, role, content, created_at)
                       VALUES (?,?,?,?)""",
                    (session_id, role, content, datetime.now().isoformat()))
        self.conn.commit()

    def get_messages(self, session_id):
        cur = self.conn.cursor()
        cur.execute("""SELECT role, content FROM chat_messages
                       WHERE session_id=? ORDER BY id""", (session_id,))
        return [{"role": r, "content": c} for r, c in cur.fetchall()]

    # ------------------------------------------------------------------ #
    # ChromaDB: ذاكرة دلالية طويلة الأمد (Long-term Memory)
    # ------------------------------------------------------------------ #
    def _init_chroma(self):
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="ckd_saliva_memory"
        )

    def add_memory(self, text: str, metadata: dict):
        doc_id = str(uuid.uuid4())
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def query_similar(self, query_text: str, n_results=5):
        """يبحث في الذاكرة طويلة الأمد عن عينات/محادثات سابقة مشابهة دلالياً."""
        try:
            count = self.collection.count()
            if count == 0:
                return []
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, count)
            )
            out = []
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                out.append({"text": doc, "metadata": meta, "distance": dist})
            return out
        except Exception:
            return []


# نسخة واحدة مشتركة (Singleton) عبر التطبيق
_memory_instance = None

def get_memory() -> LocalMemory:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = LocalMemory()
    return _memory_instance
