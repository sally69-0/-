import streamlit as st
import os
from groq import Groq

def check_ollama_available():
    return False

def check_groq_available():
    return True

def get_active_backend_label():
    return "السحابي (Groq API)"

def ask_local_llm(prompt, system_prompt="", image_bytes=None, image_paths=None):
    return ask_groq_cloud(prompt, system_prompt)

def offline_chat(prompt, system_prompt="", image_bytes=None, image_paths=None):
    return ask_groq_cloud(prompt, system_prompt)

def ask_groq_cloud(prompt, system_prompt):
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ خطأ: لم يتم العثور على مفتاح GROQ_API_KEY في إعدادات Secrets الخاصة بالسيرفر."
    
    try:
        client = Groq(api_key=api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"🚨 خطأ أثناء استدعاء Groq: {str(e)}"
        
