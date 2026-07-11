"""
app.py
=======
التطبيق الرئيسي: مساعد بحثي لتشخيص اعتلال الكلى السكري عبر اللعاب.
مشروع ISEF 2027 — للاستخدام البحثي/التعليمي فقط، وليس أداة تشخيص سريري فعلي.

للتشغيل:
    streamlit run app.py
"""

import os
import gc
import psutil
import numpy as np
import pandas as pd
import streamlit as st

from database import get_memory
from local_llm import offline_chat, check_ollama_available, check_groq_available, get_active_backend_label
from web_intelligence import has_internet, build_context_from_search, fact_check_claim
from multimodal_processor import (
    save_uploaded_image, parse_excel_data, build_combined_matrix,
    statistical_cross_comparison, plot_group_comparison,
    plot_radar_profile, plot_correlation_heatmap
)
from visualization_3d import generate_cell_damage_model
from ml_classifier import train_classifier, predict_new_sample
from report_generator import generate_sample_report

st.set_page_config(
    page_title="مساعد بحثي - اعتلال الكلى السكري عبر اللعاب (ISEF)",
    page_icon="🧫",
    layout="wide"
)

memory = get_memory()

# ==================================================================== #
# تنويه دائم وبارز أعلى كل صفحة
# ==================================================================== #
st.warning(
    "🎓 **أداة بحثية وتعليمية فقط (مشروع ISEF)** — هذا البرنامج مُصمَّم "
    "حصراً للمساعدة في البحث العلمي والتعلّم وتطوير مهارات تحليل البيانات "
    "الحيوية. **هو ليس أداة تشخيص طبي، ولا يجوز استخدامه لتشخيص أو تقييم "
    "أو اتخاذ أي قرار علاجي بخصوص مريض حقيقي.** أي نتيجة يعرضها (نصية، "
    "إحصائية، أو من نموذج التعلم الآلي) هي لأغراض الاستفادة والتطوير "
    "البحثي فقط، ولا تُغني بأي شكل عن مراجعة مختص طبي بشري مؤهل."
)

# ==================================================================== #
# إدارة الذاكرة العشوائية (Garbage Collection) — تُنفَّذ دورياً
# ==================================================================== #
def cleanup_memory():
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return mem_mb

if "gc_counter" not in st.session_state:
    st.session_state.gc_counter = 0
st.session_state.gc_counter += 1
if st.session_state.gc_counter % 5 == 0:  # كل 5 تفاعلات
    cleanup_memory()

# ==================================================================== #
# session_state الأساسي
# ==================================================================== #
if "sessions" not in st.session_state:
    existing = memory.list_sessions()
    if not existing:
        sid = memory.create_session("محادثة 1")
        st.session_state.sessions = [{"id": sid, "title": "محادثة 1"}]
    else:
        st.session_state.sessions = [{"id": s[0], "title": s[1]} for s in existing]

if "active_session" not in st.session_state:
    st.session_state.active_session = st.session_state.sessions[0]["id"]

# ==================================================================== #
# الشريط الجانبي: الإعدادات العامة + حالة الأنظمة
# ==================================================================== #
with st.sidebar:
    st.title("⚙️ الإعدادات")

    mode = st.radio("نمط التشغيل", ["تلقائي (يفضّل الأونلاين)", "أوفلاين فقط (محلي بالكامل)"])

    st.markdown("### حالة الأنظمة")
    ollama_ok = check_ollama_available()
    groq_ok = check_groq_available()
    internet_ok = has_internet()
    st.write(f"🖥️ Ollama محلي: {'✅ متصل' if ollama_ok else '❌ غير متاح'}")
    st.write(f"☁️ نموذج Groq الاحتياطي: {'✅ مُعرَّف' if groq_ok else '❌ غير مُعرَّف'}")
    st.write(f"🌐 الإنترنت: {'✅ متوفر' if internet_ok else '❌ غير متوفر'}")
    st.info(f"**المصدر النشط الآن:** {get_active_backend_label()}")

    if not ollama_ok and not groq_ok:
        st.warning(
            "لا يوجد أي مصدر ذكاء اصطناعي متاح حالياً. إن كنت تنشر التطبيق على "
            "Streamlit Cloud (حيث لا يوجد Ollama)، أضف مفتاح Groq مجاني من "
            "[console.groq.com](https://console.groq.com) في إعدادات **Secrets** بالشكل:\n\n"
            '```\nGROQ_API_KEY = "gsk_xxxxxxxx"\n```'
        )

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    st.write(f"💾 استهلاك الذاكرة الحالي: {mem_mb:.1f} MB")
    if st.button("🧹 تفريغ الذاكرة العشوائية الآن"):
        new_mb = cleanup_memory()
        st.success(f"تم التفريغ. الاستهلاك الحالي: {new_mb:.1f} MB")

    st.markdown("---")
    st.markdown("### 💬 المحادثات المتعددة")
    for s in st.session_state.sessions:
        if st.button(f"📁 {s['title']}", key=f"btn_{s['id']}"):
            st.session_state.active_session = s["id"]
    if st.button("➕ محادثة جديدة"):
        new_title = f"محادثة {len(st.session_state.sessions) + 1}"
        new_id = memory.create_session(new_title)
        st.session_state.sessions.append({"id": new_id, "title": new_title})
        st.session_state.active_session = new_id
        st.rerun()

    st.markdown("---")
    st.error(
        "⚠️ **للاستخدام البحثي والتعليمي فقط**\n\n"
        "هذا التطبيق أداة مساعدة لمشروع علمي مدرسي (ISEF)، الغرض منها الاستفادة "
        "في البحث العلمي والتطوير والتعلّم فقط. **ليس جهازاً طبياً معتمداً ولا "
        "يُستخدم إطلاقاً لتشخيص أو تقييم مرضى حقيقيين أو اتخاذ قرارات علاجية.**"
    )

# ==================================================================== #
# التبويبات الرئيسية
# ==================================================================== #
tab_chat, tab_upload, tab_compare, tab_ml, tab_3d = st.tabs([
    "💬 المحادثة مع الخبير الذكي",
    "📤 رفع العينات والبيانات",
    "📊 المقارنة الإحصائية بين المجموعات",
    "🧪 نموذج التعلم الآلي",
    "🧬 المجسم ثلاثي الأبعاد"
])

# -------------------------------------------------------------------- #
# التبويب 1: المحادثة (تدمج الذاكرة الطويلة + البحث في الويب + التحقق)
# -------------------------------------------------------------------- #
with tab_chat:
    st.subheader("محادثة مع الخبير الاستشاري الذكي")

    active_id = st.session_state.active_session
    history = memory.get_messages(active_id)

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    col_a, col_b = st.columns([3, 1])
    with col_b:
        attach_images = st.file_uploader(
            "إرفاق صور بالسؤال (اختياري)", type=["png", "jpg", "jpeg"],
            accept_multiple_files=True, key="chat_images"
        )
        do_fact_check = st.checkbox("🔎 تفعيل التحقق من الإجابة (Fact-Check)", value=True)

    user_input = st.chat_input("اكتب سؤالك حول العينة أو النتائج...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        memory.add_message(active_id, "user", user_input)

        # 1) استرجاع ذاكرة طويلة الأمد ذات صلة (عينات/محادثات سابقة مشابهة)
        similar = memory.query_similar(user_input, n_results=3)
        memory_context = ""
        if similar:
            memory_context = "معلومات من عينات/محادثات سابقة مشابهة مخزّنة محلياً:\n" + "\n".join(
                f"- {s['text'][:200]}" for s in similar
            )

        # 2) نمط أونلاين: بحث في الويب إن توفر الإنترنت واختير النمط التلقائي
        web_context = ""
        if internet_ok and mode.startswith("تلقائي"):
            with st.spinner("🔎 يبحث عن أحدث الأبحاث ذات الصلة..."):
                web_context = build_context_from_search(user_input)

        full_prompt = user_input
        if memory_context:
            full_prompt = f"{memory_context}\n\n{full_prompt}"
        if web_context:
            full_prompt = f"{web_context}\n\n{full_prompt}"

        image_paths = []
        if attach_images:
            image_paths = [save_uploaded_image(f) for f in attach_images]

        with st.chat_message("assistant"):
            with st.spinner(f"🤖 يفكّر ويحلل ({get_active_backend_label()})..."):
                chat_hist_for_llm = [{"role": m["role"], "content": m["content"]} for m in history]
                answer = offline_chat(full_prompt, chat_hist_for_llm, image_paths=image_paths)
                st.markdown(answer)

                if do_fact_check and internet_ok:
                    with st.expander("🔎 نتيجة التحقق الآلي (Fact-Check)"):
                        check = fact_check_claim(answer[:300])
                        st.write(check["note"])
                        for e in check["evidence"]:
                            st.markdown(f"- **{e['title']}**: {e['snippet'][:150]}... [مصدر]({e['url']})")

        memory.add_message(active_id, "assistant", answer)
        # حفظ في الذاكرة الدلالية أيضاً
        memory.add_memory(f"سؤال: {user_input}\nإجابة: {answer}",
                          metadata={"type": "chat", "session_id": active_id})

# -------------------------------------------------------------------- #
# التبويب 2: رفع العينات (صور + بيانات كيميائية) وتخزينها بشكل دائم
# -------------------------------------------------------------------- #
with tab_upload:
    st.subheader("رفع عينة جديدة (صورة + بيانات كيميائية)")

    with st.form("upload_form"):
        group_label = st.selectbox("تصنيف المجموعة", ["affected", "at_risk", "healthy"],
                                    format_func=lambda x: {"affected": "مصابة", "at_risk": "معرّضة",
                                                            "healthy": "سليمة"}[x])
        patient_note = st.text_area("ملاحظات حول العينة")
        image_file = st.file_uploader("صورة العينة/المجهر", type=["png", "jpg", "jpeg"])
        excel_file = st.file_uploader("ملف بيانات كيميائية (Excel/CSV) - اختياري", type=["xlsx", "csv"])

        st.markdown("**تسجيل صوتي للملاحظات (اختياري):**")
        try:
            audio_note = st.audio_input("سجّل ملاحظة صوتية")
        except AttributeError:
            # نسخ Streamlit الأقدم لا تدعم st.audio_input؛ نستخدم رفع ملف كبديل
            audio_note = st.file_uploader("أو ارفع ملف صوتي (mp3/wav)", type=["mp3", "wav", "m4a"])

        st.markdown("**أو أدخل القياسات يدوياً:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            creatinine = st.number_input("الكرياتينين (mg/dL)", min_value=0.0, value=0.0, step=0.01)
        with col2:
            albumin = st.number_input("الألبومين (mg/L)", min_value=0.0, value=0.0, step=0.1)
        with col3:
            glucose = st.number_input("الجلوكوز (mg/dL)", min_value=0.0, value=0.0, step=0.1)

        submitted = st.form_submit_button("💾 حفظ العينة في الذاكرة الدائمة")

        if submitted:
            image_path = save_uploaded_image(image_file) if image_file else ""
            chem_data = {"creatinine": creatinine, "albumin": albumin, "glucose": glucose}

            if excel_file:
                df_extra = parse_excel_data(excel_file)
                if not df_extra.empty:
                    chem_data.update(df_extra.iloc[0].to_dict())

            final_note = patient_note
            if audio_note is not None:
                import uuid as _uuid
                audio_dir = os.path.join(os.path.dirname(__file__), "data", "audio_notes")
                os.makedirs(audio_dir, exist_ok=True)
                audio_path = os.path.join(audio_dir, f"{_uuid.uuid4()}.wav")
                with open(audio_path, "wb") as f:
                    f.write(audio_note.getbuffer())
                final_note = f"{patient_note}\n[ملاحظة صوتية مرفقة: {audio_path}]"

            sample_id = memory.save_sample(
                group_label=group_label, patient_note=final_note,
                image_path=image_path, chemical_data=chem_data
            )
            st.success(f"✅ تم حفظ العينة بنجاح (ID: {sample_id[:8]}...)")

    st.markdown("---")
    st.subheader("العينات المخزّنة")
    all_samples = memory.get_all_samples()
    st.write(f"عدد العينات المخزّنة إجمالاً: **{len(all_samples)}**")
    if all_samples:
        import pandas as pd
        import json as _json2
        display_df = pd.DataFrame([
            {"المجموعة": s["group_label"], "الملاحظة": s["patient_note"][:50],
             "التاريخ": s["created_at"][:19]}
            for s in all_samples
        ])
        st.dataframe(display_df, use_container_width=True)

        st.markdown("**📄 تصدير تقرير PDF لعينة محددة**")
        options = {f"{s['sample_id'][:8]} — {s['group_label']} — {s['created_at'][:19]}": s
                   for s in all_samples}
        chosen_key = st.selectbox("اختر عينة", list(options.keys()))
        if st.button("📄 توليد تقرير PDF"):
            chosen_sample = options[chosen_key]
            chosen_sample["chemical_data"] = _json2.loads(chosen_sample["chemical_data_json"])
            ml_pred = predict_new_sample(chosen_sample["chemical_data"])
            pdf_path = generate_sample_report(
                chosen_sample,
                ai_analysis=chosen_sample.get("ai_diagnosis", "") or "",
                ml_prediction=ml_pred
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ تحميل التقرير", data=f.read(),
                    file_name=os.path.basename(pdf_path), mime="application/pdf"
                )
            st.info(
                "ملاحظة: نص التحليل داخل PDF يدعم الحروف اللاتينية بشكل أساسي "
                "(قيود مكتبة fpdf2 مع الخطوط العربية). لتقرير عربي كامل يمكن "
                "لاحقاً إضافة خط Unicode مخصص (مثال: Amiri أو Noto Naskh Arabic)."
            )

# -------------------------------------------------------------------- #
# التبويب 3: المقارنة الإحصائية والرسوم الديناميكية
# -------------------------------------------------------------------- #
with tab_compare:
    st.subheader("مقارنة إحصائية بين المجموعات الثلاث")

    all_samples = memory.get_all_samples()
    if len(all_samples) < 2:
        st.info("ارفع عينتين على الأقل من التبويب السابق لبدء المقارنة.")
    else:
        import json as _json
        parsed_samples = []
        for s in all_samples:
            parsed_samples.append({
                "sample_id": s["sample_id"],
                "group_label": s["group_label"],
                "chemical_data": _json.loads(s["chemical_data_json"])
            })

        df = build_combined_matrix(parsed_samples)
        numeric_cols = [c for c in df.columns if c not in ("sample_id", "group")]

        st.dataframe(df, use_container_width=True)

        if numeric_cols:
            st.markdown("### 📈 النتائج الإحصائية (ANOVA / t-test)")
            stats_results = statistical_cross_comparison(df, numeric_cols)
            for col, res in stats_results.items():
                sig = ""
                if res["p_value"] is not None:
                    sig = " 🔴 دلالة إحصائية محتملة (p<0.05)" if res["p_value"] < 0.05 else " ⚪ غير دالة إحصائياً"
                st.write(f"**{col}** — {res['test']}: statistic={res['statistic']}, p-value={res['p_value']}{sig}")

            selected_col = st.selectbox("اختر متغيراً لعرض المقارنة البصرية", numeric_cols)
            st.plotly_chart(plot_group_comparison(df, selected_col), use_container_width=True)

            if len(numeric_cols) >= 3:
                st.plotly_chart(plot_radar_profile(df, numeric_cols), use_container_width=True)
                st.plotly_chart(plot_correlation_heatmap(df, numeric_cols), use_container_width=True)

# -------------------------------------------------------------------- #
# التبويب 4: نموذج التعلم الآلي (Random Forest حقيقي على البيانات المخزّنة)
# -------------------------------------------------------------------- #
with tab_ml:
    st.subheader("🧪 تدريب نموذج تعلّم آلي حقيقي على العينات المخزّنة")
    st.caption(
        "بدلاً من الاعتماد فقط على النموذج اللغوي (LLM) لتوليد رأي عام، هذا القسم "
        "يدرّب نموذج **Random Forest** إحصائياً على بياناتك الفعلية المخزّنة محلياً، "
        "ويقيس دقته الحقيقية — وهذا يعزز المنهجية العلمية لمشروعك في ISEF. "
        "⚠️ بعدد عينات قليل، النتائج أولية وتعليمية فقط وليست قابلة للتعميم إحصائياً."
    )

    all_samples = memory.get_all_samples()
    if len(all_samples) < 6:
        st.info(
            f"لديك حالياً {len(all_samples)} عينة فقط. يُنصح بجمع 6 عينات على الأقل "
            "(موزعة على مجموعتين أو أكثر) قبل تدريب النموذج للحصول على نتائج ذات معنى."
        )
    else:
        import json as _json3
        parsed_samples = []
        for s in all_samples:
            parsed_samples.append({
                "sample_id": s["sample_id"],
                "group_label": s["group_label"],
                "chemical_data": _json3.loads(s["chemical_data_json"])
            })
        df_ml = build_combined_matrix(parsed_samples)

        if st.button("🚀 تدريب النموذج الآن"):
            with st.spinner("جاري تدريب نموذج Random Forest..."):
                result = train_classifier(df_ml)

            if result["status"] == "insufficient_data":
                st.warning(result["message"])
            else:
                st.success(f"✅ تم تدريب النموذج على {result['n_samples']} عينة.")
                st.warning(result["disclaimer"])

                st.markdown("### 📋 تقرير الأداء على بيانات الاختبار")
                report_df = pd.DataFrame(result["classification_report"]).transpose()
                st.dataframe(report_df, use_container_width=True)

                if result["cv_scores"]:
                    st.write(
                        f"**دقة التحقق المتقاطع (Cross-Validation):** "
                        f"{np.mean(result['cv_scores'])*100:.1f}% "
                        f"(± {np.std(result['cv_scores'])*100:.1f}%)"
                    )

                st.markdown("### 🔑 أهم المتغيرات الكيميائية في التصنيف")
                importance_df = pd.DataFrame(
                    result["feature_importance"].items(), columns=["المتغير", "الأهمية"]
                )
                st.bar_chart(importance_df.set_index("المتغير"))

        st.markdown("---")
        st.markdown("### 🔮 تجربة تنبؤ على قياسات جديدة")
        st.caption("أدخل قياسات افتراضية لعينة جديدة لرؤية تصنيف النموذج المدرَّب (إن وُجد).")
        col1, col2, col3 = st.columns(3)
        with col1:
            test_creatinine = st.number_input("الكرياتينين", min_value=0.0, value=1.0, key="ml_creat")
        with col2:
            test_albumin = st.number_input("الألبومين", min_value=0.0, value=20.0, key="ml_alb")
        with col3:
            test_glucose = st.number_input("الجلوكوز", min_value=0.0, value=100.0, key="ml_gluc")

        if st.button("🔍 تنبؤ"):
            pred = predict_new_sample({
                "creatinine": test_creatinine, "albumin": test_albumin, "glucose": test_glucose
            })
            if pred is None:
                st.error("لا يوجد نموذج مدرَّب بعد. اضغط 'تدريب النموذج الآن' أولاً.")
            else:
                st.success(f"التصنيف المتوقع: **{pred['predicted_group']}**")
                st.write("احتمالات كل فئة:")
                st.bar_chart(pd.DataFrame(pred["probabilities"].items(), columns=["الفئة", "الاحتمال"]).set_index("الفئة"))
                st.caption(pred["disclaimer"])

# -------------------------------------------------------------------- #
# التبويب 5: المجسم ثلاثي الأبعاد التخيلي
# -------------------------------------------------------------------- #
with tab_3d:
    st.subheader("مجسم ثلاثي الأبعاد تخطيطي للتلف الكلوي المحتمل")
    st.caption(
        "⚠️ هذا مجسم تخطيطي توضيحي (Schematic) يُبنى بناءً على شدة تقديرية، "
        "وليس صورة طبية حقيقية أو نتيجة تصوير فعلي للعينة."
    )

    severity = st.slider("شدة التلف التقديرية (بناءً على التشخيص)", 0.0, 1.0, 0.3, 0.05)
    regions_input = st.text_input(
        "مناطق التلف (مفصولة بفاصلة)",
        value="الغشاء القاعدي الكبيبي, الخلايا الرجلية (Podocytes), الشعيرات الدموية"
    )
    regions = [r.strip() for r in regions_input.split(",") if r.strip()]

    fig3d = generate_cell_damage_model(severity, regions)
    st.plotly_chart(fig3d, use_container_width=True)
    st.info("💡 يمكنك تدوير المجسم وتكبيره باستخدام الماوس مباشرة داخل الرسم أعلاه.")

# ==================================================================== #
# تذييل عام
# ==================================================================== #
st.markdown("---")
st.caption(
    "مشروع بحثي لمسابقة ISEF 2027 — جميع البيانات تُخزَّن محلياً على جهازك فقط "
    "(SQLite + ChromaDB) ولا تُرسل لأي خادم خارجي، باستثناء استعلامات البحث "
    "الاختيارية في نمط الأونلاين. **هذا البرنامج مخصص حصراً للاستفادة والتطوير "
    "في البحث العلمي، وليس أداة تشخيص طبي لمرضى حقيقيين.**"
)
