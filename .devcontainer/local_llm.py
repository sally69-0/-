import streamlit as st
import requests

def check_ollama_available():
    return False

def check_groq_available():
    return False

def ask_local_llm(prompt, system_prompt="", image_bytes=None):
    return ask_gemini_cloud(prompt, system_prompt)

def offline_chat(prompt, system_prompt="", image_bytes=None):
    return ask_gemini_cloud(prompt, system_prompt)

def ask_gemini_cloud(prompt, system_prompt):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ خطأ: لم يتم العثور على مفتاح GEMINI_API_KEY في إعدادات Secrets الخاصة بالسيرفر."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser: {prompt}"}]}]
    }
    
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"❌ خطأ من خادم غوغل (كود الخطأ: {res.status_code})"
    except Exception as e:
        return f"🚨 تعذر الاتصال بالنموذج السحابي: {str(e)}"
      
