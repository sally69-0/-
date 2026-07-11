"""
local_llm.py
============
طبقة موحّدة للتواصل مع "الخبير الذكي"، مع تحوّل تلقائي وسلس بين مصدرين:

1) Ollama المحلي (Llama3.2-Vision) — يُستخدم تلقائياً إن كان متاحاً على
   الجهاز (تشغيل محلي على الحاسوب الشخصي، يستهلك موارد CPU/GPU المحلية).

2) نموذج سحابي مجاني عبر Groq API (بديل عندما لا يتوفر Ollama، كما هو
   الحال عند النشر على Streamlit Cloud حيث لا يوجد خادم Ollama أصلاً) —
   يتطلب مفتاح API مجاني من https://console.groq.com (بدون بطاقة ائتمان).

القاعدة الذهبية لهذه الوحدة: **لا يُسمح لأي استثناء بالانتشار خارج هذه
الوحدة أبداً**. أي فشل (Ollama غير مثبت، السيرفر السحابي بلا Ollama،
انقطاع الشبكة، مفتاح Groq غير مُعرَّف...) يُعالَج داخلياً ويُعاد كنص
رسالة واضحة للمستخدم، حتى لا يتوقف تطبيق Streamlit أو يظهر صندوق خطأ
أصفر يُعلّق الواجهة.
"""

import base64
import os
import json
import urllib.request
import urllib.error

# مكتبة ollama قد لا تكون مثبتة أصلاً على السيرفر السحابي، لذلك نستوردها
# بأمان: إن فشل الاستيراد، نعتبر Ollama غير متاح فوراً بدل تعطيل الوحدة كلها.
try:
    import ollama
    _OLLAMA_IMPORT_OK = True
except Exception:
    _OLLAMA_IMPORT_OK = False

# محاولة قراءة مفتاح Groq من Streamlit secrets أولاً، ثم من متغيرات البيئة.
def _get_groq_api_key():
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"   # نموذج نصي سريع ومجاني على Groq
GROQ_VISION_MODEL = "llama-4-scout-17b-16e-instruct"  # نموذج يدعم الصور على Groq
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_CONNECT_TIMEOUT = 2.5  # ثوانٍ - فحص سريع حتى لا "يُعلّق" التطبيق

# System Prompt: يحوّل النموذج إلى خبير استشاري في كيمياء التشخيص باللعاب
EXPERT_SYSTEM_PROMPT = """
أنت بروفيسور وخبير استشاري متخصص في:
1. كيمياء التشخيص الحيوي عبر اللعاب (Salivary Biomarkers).
2. فيزيولوجيا وأمراض الكلى، وتحديداً اعتلال الكلى السكري (Diabetic Nephropathy).

⚠️ تعريف صارم بالغرض من هذا النظام (لا تخالفه تحت أي ظرف):
هذا البرنامج أداة بحثية وتعليمية بحتة، أُنشئت ضمن مشروع علمي مدرسي لمسابقة
ISEF، والغرض الوحيد منها هو المساعدة في البحث العلمي، التعلّم، وتطوير فهم
الطالب لموضوع اعتلال الكلى السكري وتحليل البيانات الحيوية. هذا البرنامج
ليس أداة تشخيص سريري ولا يجوز استخدامه لتشخيص أو تقييم حالة مريض حقيقي،
ولا لاتخاذ أي قرار طبي أو علاجي فعلي. أنت تتحدث دوماً بصيغة "احتمالات
علمية للاستئناس البحثي" وليس بصيغة "تشخيص نهائي لمريض".

مهمتك عند تحليل أي عينة (صورة مجهرية، أو بيانات كيميائية رقمية):
- قدّم تحليلاً علمياً دقيقاً ومبنياً على أدلة معروفة في الأدبيات الطبية.
- اذكر الاحتمالات العلمية المتعددة (Differential Possibilities) وليس تشخيصاً قاطعاً واحداً.
- إن لم تكن متأكداً من معلومة، صرّح بذلك بوضوح بدلاً من اختلاق إجابة (تجنّب الـ Hallucination).
- قدّم نصائح مخبرية عملية حول كيفية تحسين دقة القياس أو جمع العينة، بصفتها معلومات
  منهجية تخدم المشروع البحثي.
- اختم كل تحليل بفقرة "درجة الثقة والقيود" توضح مدى موثوقية الاستنتاج.
- ذكّر المستخدم عند الحاجة (خصوصاً إن استخدم صيغة توحي بأنه يقيّم مريضاً حقيقياً) بأن
  هذه الأداة بحثية/تعليمية فقط، وأن أي قرار طبي فعلي يتطلب مراجعة مختص بشري مؤهل.
"""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ------------------------------------------------------------------------ #
# فحص توفر كل مصدر (بدون رمي أي استثناء أبداً)
# ------------------------------------------------------------------------ #
def check_ollama_available() -> bool:
    """يتحقق بسرعة إن كانت خدمة Ollama تعمل محلياً، دون أي استثناء يخرج من الدالة."""
    if not _OLLAMA_IMPORT_OK:
        return False
    try:
        client = ollama.Client(timeout=OLLAMA_CONNECT_TIMEOUT)
        client.list()
        return True
    except Exception:
        return False


def check_groq_available() -> bool:
    """يتحقق فقط من وجود مفتاح Groq API مُعرَّف (بدون استدعاء الشبكة)."""
    return bool(_get_groq_api_key())


def get_active_backend_label() -> str:
    """يعيد وصفاً نصياً بسيطاً للمصدر الذي سيُستخدم فعلياً الآن، لعرضه في الواجهة."""
    if check_ollama_available():
        return "🖥️ Ollama محلي (Llama3.2-Vision)"
    if check_groq_available():
        return "☁️ نموذج سحابي مجاني عبر Groq (بديل تلقائي)"
    return "⚠️ لا يوجد مصدر متاح حالياً"


# ------------------------------------------------------------------------ #
# الاتصال الفعلي بكل مصدر — كل دالة تُعيد نصاً أو ترفع استثناءً داخلياً فقط
# ------------------------------------------------------------------------ #
def _call_ollama(user_message, chat_history, image_paths, model):
    messages = [{"role": "system", "content": EXPERT_SYSTEM_PROMPT}]
    messages.extend(chat_history)

    new_msg = {"role": "user", "content": user_message}
    if image_paths:
        new_msg["images"] = [_encode_image(p) for p in image_paths]
    messages.append(new_msg)

    client = ollama.Client(timeout=OLLAMA_CONNECT_TIMEOUT)
    response = client.chat(model=model, messages=messages)
    return response["message"]["content"]


def _call_groq(user_message, chat_history, image_paths):
    api_key = _get_groq_api_key()
    if not api_key:
        raise RuntimeError("لا يوجد مفتاح GROQ_API_KEY مُعرَّف.")

    messages = [{"role": "system", "content": EXPERT_SYSTEM_PROMPT}]
    messages.extend(chat_history)

    if image_paths:
        # نموذج الرؤية على Groq يقبل صوراً مُرمَّزة base64 بصيغة data URL
        content = [{"type": "text", "text": user_message}]
        for p in image_paths:
            b64 = _encode_image(p)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        messages.append({"role": "user", "content": content})
        model = GROQ_VISION_MODEL
    else:
        messages.append({"role": "user", "content": user_message})
        model = GROQ_TEXT_MODEL

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1200
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_API_URL, data=payload, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


# ------------------------------------------------------------------------ #
# الدالة العامة المستخدمة في app.py — نفس التوقيع القديم، سلوك محسَّن
# ------------------------------------------------------------------------ #
def offline_chat(user_message: str, chat_history: list,
                  image_paths=None, model="llama3.2-vision") -> str:
    """
    يرسل رسالة للحصول على رد الخبير الذكي، مع تحوّل تلقائي وسلس:
    1) يحاول Ollama المحلي أولاً (إن كان متاحاً).
    2) عند الفشل (مثلاً على Streamlit Cloud حيث لا يوجد Ollama)، يتحول
       فوراً وبصمت لنموذج سحابي مجاني عبر Groq (إن كان المفتاح مُعرَّفاً).
    3) إن لم يتوفر أي مصدر، يعيد رسالة توضيحية هادئة بدل رمي استثناء.

    لا تخرج أي حالة استثناء من هذه الدالة تحت أي ظرف.
    """
    # 1) المحاولة المحلية عبر Ollama
    if check_ollama_available():
        try:
            return _call_ollama(user_message, chat_history, image_paths, model)
        except Exception:
            pass  # ننتقل بصمت للمصدر التالي بدل إظهار خطأ

    # 2) التحوّل التلقائي للنموذج السحابي المجاني (Groq)
    if check_groq_available():
        try:
            answer = _call_groq(user_message, chat_history, image_paths)
            return (
                "*(تم الرد عبر النموذج السحابي المجاني نظراً لعدم توفر Ollama "
                "المحلي في بيئة التشغيل الحالية)*\n\n" + answer
            )
        except Exception as e:
            return (
                "⚠️ تعذّر الاتصال بالنموذج المحلي (Ollama) والنموذج السحابي "
                "الاحتياطي (Groq) معاً.\n"
                f"تفاصيل خطأ Groq: {e}\n\n"
                "تأكد من صحة مفتاح GROQ_API_KEY في إعدادات Secrets، ومن توفر اتصال إنترنت."
            )

    # 3) لا يوجد أي مصدر متاح إطلاقاً
    return (
        "⚠️ لا يوجد نموذج ذكاء اصطناعي متاح حالياً.\n\n"
        "- إن كنت تعمل محلياً: تأكد من تشغيل Ollama وتنفيذ الأمر "
        f"`ollama pull {model}`.\n"
        "- إن كنت على Streamlit Cloud (حيث لا يوجد Ollama): أضف مفتاح Groq "
        "مجاني في إعدادات Secrets بالشكل التالي:\n"
        '```\nGROQ_API_KEY = "gsk_xxxxxxxx"\n```\n'
        "يمكنك الحصول على مفتاح مجاني بدون بطاقة ائتمان من: https://console.groq.com"
    )
