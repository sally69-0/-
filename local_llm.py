import streamlit as st
import requests

def ask_local_llm(prompt, system_prompt="", image_bytes=None):
    """
    دالة تحاول الاتصال بـ Ollama محلياً، وإذا فشلت تتحول تلقائياً 
    إلى النمط السحابي المعتمد على Gemini API من جوجل.
    """
    ollama_url = "http://localhost:11434/api/generate"
    
    try:
        # 1. محاولة الاتصال بـ Ollama محلياً
        response = requests.post(
            ollama_url, 
            json={
                "model": "llama3.2-vision", 
                "prompt": f"{system_prompt}\n\nUser: {prompt}",
                "stream": False
            },
            timeout=2
        )
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception:
        pass

    # 2. التحويل التلقائي للنمط السحابي (Gemini API) عند عدم توفر Ollama
    return ask_gemini_cloud(prompt, system_prompt)

def ask_gemini_cloud(prompt, system_prompt):
    # جلب مفتاح Gemini من إعدادات السيرفر الآمنة
    api_key = st.secrets.get("GEMINI_API_KEY")
    
    if not api_key:
        return "⚠️ خطأ: لم يتم العثور على مفتاح GEMINI_API_KEY في إعدادات Secrets الخاصة بالسيرفر."
    
    # رابط استدعاء نموذج Gemini 1.5 Flash السريع والمجاني
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
