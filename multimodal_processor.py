"""
multimodal_processor.py
========================
معالجة متعددة الأنماط (صور + بيانات كيميائية رقمية) وربطها في مصفوفة واحدة،
مع مقارنة إحصائية فورية بين المجموعات الثلاث:
    'affected' (مصابة) | 'at_risk' (معرّضة) | 'healthy' (سليمة)
"""

import os
import uuid
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_image(uploaded_file) -> str:
    """يحفظ صورة مرفوعة من Streamlit محلياً ويعيد المسار."""
    ext = os.path.splitext(uploaded_file.name)[1] or ".png"
    filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    img = Image.open(uploaded_file).convert("RGB")
    img.save(path)
    return path


def parse_excel_data(uploaded_file) -> pd.DataFrame:
    """يقرأ ملف إكسل/CSV يحتوي قياسات كيميائية رقمية (مثال: كرياتينين، ألبومين، الخ)."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def build_combined_matrix(samples: list) -> pd.DataFrame:
    """
    يدمج عينات متعددة (كل عينة dict فيها group_label + chemical_data)
    في مصفوفة واحدة (DataFrame) جاهزة للتحليل والمقارنة.
    """
    rows = []
    for s in samples:
        row = {"sample_id": s.get("sample_id"), "group": s.get("group_label")}
        chem = s.get("chemical_data") or {}
        row.update(chem)
        rows.append(row)
    return pd.DataFrame(rows)


def statistical_cross_comparison(df: pd.DataFrame, numeric_columns=None) -> dict:
    """
    مقارنة إحصائية بين المجموعات الثلاث لكل متغير كيميائي رقمي:
    - ANOVA بين المجموعات الثلاث (إن توفرت جميعها)
    - أو Independent t-test بين أي مجموعتين متوفرتين
    يعيد dict: {column_name: {"test": ..., "statistic": ..., "p_value": ...}}
    """
    if numeric_columns is None:
        numeric_columns = [c for c in df.columns if c not in ("sample_id", "group")
                           and pd.api.types.is_numeric_dtype(df[c])]

    results = {}
    groups_present = df["group"].dropna().unique().tolist()

    for col in numeric_columns:
        groups_data = [
            df[df["group"] == g][col].dropna().values
            for g in groups_present
        ]
        groups_data = [g for g in groups_data if len(g) >= 2]  # يحتاج عينتين على الأقل

        if len(groups_data) >= 3:
            stat, p = stats.f_oneway(*groups_data)
            results[col] = {"test": "ANOVA (3 مجموعات)", "statistic": round(float(stat), 4),
                             "p_value": round(float(p), 5)}
        elif len(groups_data) == 2:
            stat, p = stats.ttest_ind(groups_data[0], groups_data[1], equal_var=False)
            results[col] = {"test": "Independent t-test", "statistic": round(float(stat), 4),
                             "p_value": round(float(p), 5)}
        else:
            results[col] = {"test": "بيانات غير كافية", "statistic": None, "p_value": None}

    return results


def plot_group_comparison(df: pd.DataFrame, column: str):
    """رسم صندوقي (Box Plot) ديناميكي يقارن توزيع متغير معين بين المجموعات الثلاث."""
    fig = px.box(
        df, x="group", y=column, color="group", points="all",
        title=f"مقارنة {column} بين المجموعات (مصابة / معرّضة / سليمة)",
        labels={"group": "المجموعة", column: column},
        color_discrete_map={"affected": "#d62728", "at_risk": "#ff7f0e", "healthy": "#2ca02c"}
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_radar_profile(df: pd.DataFrame, numeric_columns: list):
    """رسم راداري (Radar Chart) يقارن متوسط عدة متغيرات كيميائية معاً بين المجموعات."""
    fig = go.Figure()
    for group in df["group"].dropna().unique():
        sub = df[df["group"] == group]
        means = [sub[c].mean() if c in sub else 0 for c in numeric_columns]
        fig.add_trace(go.Scatterpolar(
            r=means, theta=numeric_columns, fill='toself', name=str(group)
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="الملف الكيميائي المقارن بين المجموعات",
        template="plotly_white"
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_columns: list):
    """خريطة حرارية للارتباط بين المتغيرات الكيميائية المختلفة."""
    corr = df[numeric_columns].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="مصفوفة الارتباط بين المتغيرات الكيميائية"
    )
    return fig
