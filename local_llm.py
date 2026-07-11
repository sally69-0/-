import streamlit as st
import requests

def check_ollama_available():
    """يرجع False دائماً على السيرفر لكي يجبر التطبيق على الانتقال للأونلاين"""
    return False

def check_groq_available():
    """يرجع False لكي نتفادى حظر Groq ونعبر مباشرة لـ Gemini"""
    return False

def ask_local_llm(prompt, system_prompt="", image_bytes=None):
    """الدالة الأساسية لاستقبال السؤال والتحويل لـ Gemini"""
    return ask_gemini_cloud(prompt, system_prompt)

def offline_chat(prompt, system_prompt="", image_bytes=None):
    """هذه الدالة يطلبها ملف app.py في السطر 19، قمنا بربطها بـ Gemini مباشرة"""
    return ask_gemini_cloud(prompt, system_prompt)

def ask_gemini_cloud(prompt, system_prompt):
    # جلب مفتاح Gemini من الـ Secrets
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    if not api_key:
        return "⚠️ خطأ: لم يتم العثور على مفتاح GEMINI_API_KEY في إعدادات Secrets الخاصة بالسيرفر."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_prompt}\n\nUser: {prompt}"}
                ]
            }
        ]
    }
    
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            result_json = res.json()
            return result_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"❌ خطأ من خادم Google Gemini (كود الخطأ: {res.status_code})"
    except Exception as e:
        return f"🚨 تعذر الاتصال بالنموذج السحابي: {str(e)}"
        
