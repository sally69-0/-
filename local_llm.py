import streamlit as st
import os

# نحاول استدعاء المكتبة الحديثة لغوغل
try:
    from google import genai
except ImportError:
    pass

def check_ollama_available():
    return False

def check_groq_available():
    return False

def get_active_backend_label():
    return "السحابي (Google Gemini)"

def ask_local_llm(prompt, system_prompt="", image_bytes=None, image_paths=None):
    return ask_gemini_cloud(prompt, system_prompt)

def offline_chat(prompt, system_prompt="", image_bytes=None, image_paths=None):
    return ask_gemini_cloud(prompt, system_prompt)

def ask_gemini_cloud(prompt, system_prompt):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ خطأ: لم يتم العثور على مفتاح GEMINI_API_KEY في إعدادات Secrets الخاصة بالسيرفر."
    
    try:
        # إعداد العميل باستخدام المكتبة الرسمية
        client = genai.Client(api_key=api_key)
        
        # دمج الـ System Prompt مع السؤال
        full_prompt = f"{system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
        
        # استدعاء الموديل المستقر والسريع
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
        )
        return response.text
    except Exception as e:
        return f"🚨 خطأ أثناء استدعاء Gemini: {str(e)}"
