"""
local_llm.py
============
نمط الأوفلاين (Offline Mode): تشغيل نموذج Llama3.2-Vision محلياً عبر Ollama.
يستهلك موارد الجهاز (CPU/GPU) فقط، بدون أي اتصال إنترنت.

يتطلب تثبيت Ollama مسبقاً وتنزيل النموذج:
    ollama pull llama3.2-vision
"""

import base64
import ollama

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


def check_ollama_available() -> bool:
    """يتحقق إن كانت خدمة Ollama تعمل محلياً."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def offline_chat(user_message: str, chat_history: list,
                  image_paths=None, model="llama3.2-vision") -> str:
    """
    يرسل رسالة للنموذج المحلي عبر Ollama ويعيد الرد النصي.
    chat_history: قائمة [{"role": "user"/"assistant", "content": "..."}]
    image_paths: قائمة مسارات صور اختيارية لإرفاقها بالرسالة الحالية (متعدد الصور)
    """
    messages = [{"role": "system", "content": EXPERT_SYSTEM_PROMPT}]
    messages.extend(chat_history)

    new_msg = {"role": "user", "content": user_message}
    if image_paths:
        new_msg["images"] = [_encode_image(p) for p in image_paths]
    messages.append(new_msg)

    try:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        return (
            f"⚠️ تعذّر الاتصال بنموذج Ollama المحلي ({model}).\n"
            f"تأكد من تشغيل Ollama وتنفيذ الأمر: `ollama pull {model}`\n"
            f"تفاصيل الخطأ: {e}"
        )
