"""
ml_classifier.py
==================
نموذج تعلّم آلي حقيقي (Random Forest) يُدرَّب على العينات المخزّنة فعلياً
في قاعدة البيانات المحلية (وليس فقط على استنتاج LLM عام).

هذا يعزز المنهجية العلمية للمشروع: بدل الاعتماد كلياً على نموذج لغوي
عام لتوليد "رأي"، نبني نموذج تصنيف إحصائي مدرَّب على بيانات العينات
الفعلية (المصابة / المعرّضة / السليمة) ونقيس دقته الحقيقية.

⚠️ ملاحظة منهجية مهمة: بعدد عينات قليل (كما هو متوقع في مشروع ثانوي/ISEF)
النتائج ستكون أولية وغير قابلة للتعميم إحصائياً. النموذج هنا أداة تعلّم
وتوضيح لمفهوم "التصنيف بالتعلم الآلي"، وليس أداة تشخيص سريري بأي شكل.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "rf_classifier.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")


def prepare_training_data(df: pd.DataFrame):
    """يجهّز مصفوفة الميزات (X) والتصنيفات (y) من DataFrame العينات المدمجة."""
    feature_cols = [c for c in df.columns if c not in ("sample_id", "group")
                     and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].fillna(df[feature_cols].mean(numeric_only=True))
    y = df["group"]
    return X, y, feature_cols


def train_classifier(df: pd.DataFrame, n_estimators=200, test_size=0.25, random_state=42):
    """
    يدرّب نموذج Random Forest على العينات المخزّنة فعلياً.
    يعيد dict فيه: النموذج، تقرير الأداء، أهمية المتغيرات، ومصفوفة الالتباس.
    """
    X, y, feature_cols = prepare_training_data(df)

    if len(df) < 6 or df["group"].nunique() < 2:
        return {
            "status": "insufficient_data",
            "message": "عدد العينات أو عدد المجموعات غير كافٍ للتدريب. "
                       "يُنصح بجمع 6 عينات على الأقل موزعة على مجموعتين مختلفتين فأكثر."
        }

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
    except ValueError:
        # عدد العينات لكل فئة قليل جداً للتقسيم الطبقي
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                  class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    # التحقق المتقاطع (Cross-Validation) لتقدير أكثر موثوقية بعدد عينات قليل
    try:
        cv_scores = cross_val_score(clf, X, y_encoded, cv=min(3, len(df) // 2))
    except Exception:
        cv_scores = np.array([])

    importance = dict(sorted(
        zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]
    ))

    # حفظ النموذج محلياً لإعادة استخدامه لاحقاً دون إعادة تدريب
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    return {
        "status": "trained",
        "model": clf,
        "label_encoder": le,
        "feature_columns": feature_cols,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": le.classes_.tolist(),
        "cv_scores": cv_scores.tolist(),
        "n_samples": len(df),
        "feature_importance": importance,
        "disclaimer": "نتائج أولية بعدد عينات محدود — لأغراض تعليمية وبحثية فقط، "
                      "لا تُستخدم كأداة تشخيص فعلية."
    }


def predict_new_sample(chemical_data: dict):
    """
    يستخدم النموذج المحفوظ مسبقاً (إن وُجد) للتنبؤ بتصنيف عينة جديدة.
    يعيد None إن لم يوجد نموذج مدرَّب بعد.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        return None

    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    feature_names = clf.feature_names_in_ if hasattr(clf, "feature_names_in_") else None
    if feature_names is None:
        return None

    row = {f: chemical_data.get(f, 0.0) for f in feature_names}
    X_new = pd.DataFrame([row])

    pred_encoded = clf.predict(X_new)[0]
    pred_proba = clf.predict_proba(X_new)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    proba_dict = {cls: round(float(p), 3) for cls, p in zip(le.classes_, pred_proba)}

    return {
        "predicted_group": pred_label,
        "probabilities": proba_dict,
        "disclaimer": "تنبؤ إحصائي أولي من نموذج تعلّم آلي مدرَّب محلياً على عدد عينات محدود. "
                      "لأغراض بحثية/تعليمية فقط — ليس تشخيصاً طبياً."
    }
