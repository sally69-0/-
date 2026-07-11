"""
web_intelligence.py
====================
نمط الأونلاين (Online Mode): البحث في الويب عبر DuckDuckGo-Search
لجلب أحدث الأبحاث السريرية، وتوفير خاصية "التحقق من الإجابات" لتقليل
احتمالية تخريف النموذج (Hallucination Mitigation).

ملاحظة: هذه الوحدة اختيارية بالكامل — إن لم يتوفر إنترنت، يتجاهلها
التطبيق تلقائياً ويعمل بنمط الأوفلاين فقط.
"""

import socket
from duckduckgo_search import DDGS


def has_internet(timeout=2.0) -> bool:
    """فحص سريع لتوفر الإنترنت قبل محاولة البحث."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except OSError:
        return False


def search_medical_literature(query: str, max_results=5) -> list:
    """
    يبحث عن أحدث الأبحاث/المصطلحات الطبية المرتبطة بالاستعلام.
    يعيد قائمة من dict: {title, snippet, url}
    """
    if not has_internet():
        return []

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                f"{query} diabetic nephropathy saliva biomarker research",
                max_results=max_results
            ):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
    except Exception as e:
        results.append({"title": "خطأ في البحث", "snippet": str(e), "url": ""})
    return results


def build_context_from_search(query: str, max_results=5) -> str:
    """يحوّل نتائج البحث إلى نص سياقي (context) يُدمج مع System Prompt للنموذج."""
    results = search_medical_literature(query, max_results)
    if not results:
        return ""

    context_lines = ["نتائج بحث حديثة ذات صلة (للاستئناس، وليست مصدراً نهائياً):"]
    for i, r in enumerate(results, 1):
        context_lines.append(f"{i}. {r['title']} — {r['snippet'][:200]}... (المصدر: {r['url']})")
    return "\n".join(context_lines)


def fact_check_claim(claim: str, max_results=3) -> dict:
    """
    خاصية 'التحقق من السؤال والنتائج': يبحث عن أدلة داعمة أو مضادة لادعاء
    معين صادر عن النموذج، ويعيد ملخصاً يساعد المستخدم على تقييم الموثوقية.
    """
    if not has_internet():
        return {"status": "offline", "evidence": [], "note": "لا يوجد اتصال إنترنت للتحقق."}

    evidence = search_medical_literature(claim, max_results)
    return {
        "status": "checked",
        "evidence": evidence,
        "note": "راجع المصادر أدناه للتأكد من مطابقة الادعاء للأدبيات المنشورة. "
                "هذا التحقق آلي وأولي فقط، ولا يغني عن مراجعة خبير بشري."
    }
