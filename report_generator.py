"""
report_generator.py
=====================
تصدير تقرير PDF تلقائي لكل عينة، يتضمن: بيانات العينة، القياسات الكيميائية،
تشخيص/تحليل الذكاء الاصطناعي (إن وُجد)، وتنويه علمي واضح في أعلى وأسفل
كل تقرير يوضّح أن الغرض بحثي/تعليمي فقط.
"""

import os
from datetime import datetime
from fpdf import FPDF

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "data", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

DISCLAIMER_TEXT = (
    "تنويه علمي: هذا التقرير نتاج مشروع بحثي/تعليمي (ISEF) ولا يمثل تشخيصاً "
    "طبياً معتمداً. الغرض من هذا البرنامج هو المساعدة في البحث العلمي والتطوير "
    "والتعلّم فقط، وليس تشخيص أو علاج مرضى حقيقيين. لأي قرار طبي فعلي يجب "
    "مراجعة مختص بشري مؤهل."
)


class SampleReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Salivary Diabetic Nephropathy Research Report", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(150, 30, 30)
        self.multi_cell(0, 5, "FOR RESEARCH & EDUCATIONAL PURPOSES ONLY - NOT A MEDICAL DIAGNOSIS", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-20)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.multi_cell(0, 4, DISCLAIMER_TEXT, align="C")


def generate_sample_report(sample: dict, ai_analysis: str = "", ml_prediction: dict = None) -> str:
    """
    يولّد ملف PDF لعينة واحدة ويعيد مساره.
    sample: dict يحتوي (sample_id, group_label, patient_note, created_at, chemical_data)
    """
    pdf = SampleReportPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, f"Sample ID: {sample.get('sample_id', '')[:12]}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Group Label: {sample.get('group_label', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Date: {sample.get('created_at', datetime.now().isoformat())[:19]}", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Patient Note:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, sample.get("patient_note", "") or "N/A")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Chemical Measurements:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    chem_data = sample.get("chemical_data", {}) or {}
    if chem_data:
        for k, v in chem_data.items():
            pdf.cell(0, 6, f"  - {k}: {v}", ln=True)
    else:
        pdf.cell(0, 6, "  N/A", ln=True)
    pdf.ln(3)

    if ml_prediction:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "ML Model Prediction (Preliminary):", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  Predicted group: {ml_prediction.get('predicted_group', 'N/A')}", ln=True)
        for cls, p in (ml_prediction.get("probabilities") or {}).items():
            pdf.cell(0, 6, f"    - {cls}: {p*100:.1f}%", ln=True)
        pdf.ln(3)

    if ai_analysis:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "AI Assistant Analysis:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        # fpdf2 لا يدعم كل حروف اليونيكود افتراضياً بخط Helvetica؛
        # لذلك نحول أي نص عربي/رموز غير مدعومة بأمان قدر الإمكان
        safe_text = ai_analysis.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_text)
        pdf.ln(3)

    filename = f"report_{sample.get('sample_id', 'sample')[:8]}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)
    pdf.output(filepath)
    return filepath
