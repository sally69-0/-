"""
visualization_3d.py
=====================
توليد مجسمات ثلاثية الأبعاد "تخيلية توضيحية" (Illustrative 3D Models) للخلية
الكلوية وأماكن التلف الكيميائي المحتملة، بناءً على نتيجة التشخيص.

⚠️ ملاحظة علمية مهمة (يجب إظهارها في الواجهة):
هذه المجسمات هي تمثيل تخطيطي/تعليمي (schematic) وليست صورة حقيقية أو نتيجة
تصوير طبي فعلي. تُستخدم فقط لتسهيل شرح موقع الضرر الكيميائي المحتمل
(مثال: أغشية الترشيح الكبيبية Glomerular Basement Membrane) بصرياً.

نستخدم Plotly 3D Mesh لأنها الأكثر استقراراً وتوافقاً مع Streamlit مقارنة
بمكتبات threejs الإضافية التي قد تحتاج تهيئة معقدة.
"""

import numpy as np
import plotly.graph_objects as go


def _generate_sphere(radius=1.0, resolution=40, center=(0, 0, 0)):
    """يولّد إحداثيات كرة (تمثيل مبسط لخلية كلوية / كبيبة Glomerulus)."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    return x, y, z


def generate_cell_damage_model(damage_severity: float, damage_regions: list = None):
    """
    يبني مجسم 3D تفاعلي (قابل للتدوير/التحجيم بالماوس داخل Streamlit).

    damage_severity: قيمة بين 0 و1 تمثل شدة التلف المقدّرة من التشخيص
    damage_regions: قائمة أسماء مناطق تخيلية للتلف (مثال: ["الغشاء القاعدي", "الخلايا الرجلية"])

    يعيد كائن plotly.graph_objects.Figure جاهز للعرض عبر st.plotly_chart.
    """
    if damage_regions is None:
        damage_regions = ["الغشاء القاعدي الكبيبي", "الخلايا الرجلية (Podocytes)", "الشعيرات الدموية"]

    fig = go.Figure()

    # الخلية الأساسية (الكبيبة الكلوية) - كرة شفافة خضراء تمثل الأنسجة السليمة
    x, y, z = _generate_sphere(radius=1.0, resolution=45)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#2ca02c"], [1, "#2ca02c"]],
        opacity=0.35,
        showscale=False,
        name="نسيج سليم (تمثيل تخطيطي)"
    ))

    # مناطق تلف تخيلية: كتل حمراء متناثرة على سطح الكرة، حجمها يعتمد على شدة التلف
    rng = np.random.default_rng(42)  # seed ثابت لثبات الشكل بين مرات العرض
    n_spots = max(3, int(damage_severity * 25))
    theta = rng.uniform(0, 2 * np.pi, n_spots)
    phi = rng.uniform(0, np.pi, n_spots)
    spot_x = np.cos(theta) * np.sin(phi) * 1.02
    spot_y = np.sin(theta) * np.sin(phi) * 1.02
    spot_z = np.cos(phi) * 1.02

    fig.add_trace(go.Scatter3d(
        x=spot_x, y=spot_y, z=spot_z,
        mode="markers",
        marker=dict(
            size=6 + 10 * damage_severity,
            color=spot_z,
            colorscale="Reds",
            opacity=0.85
        ),
        name=f"مناطق تلف كيميائي محتملة (شدة تقديرية: {damage_severity*100:.0f}%)",
        text=[damage_regions[i % len(damage_regions)] for i in range(n_spots)],
        hoverinfo="text"
    ))

    fig.update_layout(
        title="نموذج تخطيطي ثلاثي الأبعاد للكبيبة الكلوية ومناطق التلف المحتملة (توضيحي وليس تشخيصياً فعلياً)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube"
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h", y=-0.05)
    )
    return fig


def export_simple_obj(damage_severity: float, filepath: str):
    """
    تصدير اختياري لملف OBJ بسيط (كرة) يمكن فتحه في برامج ثلاثية الأبعاد خارجية.
    مفيد إن أراد الطالب عرض المجسم في برنامج عرض OBJ مستقل أثناء المسابقة.
    """
    x, y, z = _generate_sphere(radius=1.0, resolution=24)
    verts = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            verts.append((x[i, j], y[i, j], z[i, j]))

    with open(filepath, "w") as f:
        f.write(f"# Schematic glomerulus model - damage severity {damage_severity}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    return filepath
