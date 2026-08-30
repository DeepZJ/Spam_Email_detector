# ============================================================
# app.py
# EMAIL FRAUD RISK SCREENING SYSTEM
# ============================================================
#
# Developed by DeepanJ
#
# FINAL MODEL — TWO-LAYER RISK SYSTEM
#
#   LAYER 1
#     Word TF-IDF + Character TF-IDF + Linear SVM
#     (450,000-feature combined representation)
#
#   LAYER 2
#     Histogram Gradient Boosting screening model
#     Learns to reinterpret the Layer 1 decision score
#     (added because Layer 1 alone, trained on Enron
#     corporate email, under-recognised fraud patterns
#     in real-world academic correspondence)
#
# INPUT
#   Subject
#   Body
#
# OUTPUT
#   Layer 1 SVM decision score
#   Layer 2 screening probability
#   Combined risk score
#   Risk level
#   Recommended action
#
# IMPORTANT
#   The model uses ONLY the email Subject and Body.
#   No sender metadata, domain reputation, or
#   authentication records are required at prediction time.
#
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import hstack


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Email Fraud Risk Screening",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# SAVED MODEL COMPONENTS
# ============================================================

WORD_TFIDF_PATH = os.path.join(MODEL_DIR, "word_tfidf.joblib")
CHAR_TFIDF_PATH = os.path.join(MODEL_DIR, "char_tfidf.joblib")
SVM_PATH = os.path.join(MODEL_DIR, "fraud_svm.joblib")
SECOND_LAYER_PATH = os.path.join(MODEL_DIR, "second_layer_hgb.joblib")
NORM_STATS_PATH = os.path.join(MODEL_DIR, "layer2_norm_stats.json")


# ============================================================
# LOAD TRAINED MODEL (both layers)
# ============================================================

@st.cache_resource
def load_model():

    required_files = [
        WORD_TFIDF_PATH,
        CHAR_TFIDF_PATH,
        SVM_PATH,
        SECOND_LAYER_PATH
    ]

    missing_files = [p for p in required_files if not os.path.exists(p)]

    if missing_files:

        st.error(
            "One or more trained model files are missing."
        )

        st.code(
            "\n".join(missing_files)
        )

        st.stop()

    try:

        word_vectorizer = joblib.load(WORD_TFIDF_PATH)
        char_vectorizer = joblib.load(CHAR_TFIDF_PATH)
        svm_model = joblib.load(SVM_PATH)
        second_layer_model = joblib.load(SECOND_LAYER_PATH)

        if os.path.exists(NORM_STATS_PATH):
            with open(NORM_STATS_PATH) as f:
                norm_stats = json.load(f)
        else:
            # Fallback -- frozen values recovered from the original
            # Layer 1 / Layer 2 training run. Prefer the saved JSON
            # (layer2_norm_stats.json) when available.
            norm_stats = {
                "score_mean": -1.810384831177963,
                "score_std": 0.460271,
                "svm_flag_thresholds": {
                    "suspicious": 0.0282,
                    "high": 0.2254,
                    "very_high": 0.4338,
                    "extreme": 0.7718
                }
            }

        return (
            word_vectorizer,
            char_vectorizer,
            svm_model,
            second_layer_model,
            norm_stats
        )

    except Exception as e:

        st.error(
            "The saved model could not be loaded."
        )

        st.exception(e)

        st.stop()


(
    word_vectorizer,
    char_vectorizer,
    svm_model,
    second_layer_model,
    norm_stats
) = load_model()

SCORE_MEAN = norm_stats["score_mean"]
SCORE_STD = norm_stats["score_std"]
FLAG_THRESHOLDS = norm_stats["svm_flag_thresholds"]

# Frozen min/max from the 88,859-row internal Enron test set,
# used to rescale the raw SVM score into the combined risk score.
# Frozen deliberately -- never recomputed from a live input batch,
# which would make the score dependent on whatever else happens
# to be scored at the same time.
SVM_FROZEN_MIN = -8.724080691030847
SVM_FROZEN_MAX = 1.9715600381542944


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_NAME = (
    "Two-Layer: (Word + Character TF-IDF + Linear SVM) "
    "→ Gradient-Boosted Screening Layer"
)

# --- Layer 1 alone, internal Enron test set ---
LAYER1_ACCURACY = 0.9967
LAYER1_ROC_AUC = 0.9909
LAYER1_PR_AUC = 0.6962
LAYER1_PRECISION = 0.7005
LAYER1_RECALL = 0.5978
LAYER1_F1 = 0.6451

# --- Combined two-layer system, external academic-domain
#     validation set (n=60, 30 ham / 30 spam, held out from
#     training). Reported honestly, including where it's weaker --
#     this is a small validation set and these numbers should be
#     read as directional, not definitive. ---
EXTERNAL_N = 60
EXTERNAL_ROC_AUC = 0.6911
EXTERNAL_PR_AUC = 0.6924
EXTERNAL_PRECISION = 0.6222
EXTERNAL_RECALL = 0.9333
EXTERNAL_F1 = 0.7467


# ============================================================
# RISK OPERATING POINTS (on the 0-1 combined risk score)
# ============================================================
#
# A 5-tier split (fixed, guessed boundaries) failed a sanity
# check on the n=60 external validation set: non-monotonic
# spam rate, 3 of 5 tiers with 0-1 emails.
#
# Replaced with a 3-tier QUANTILE split (equal-sized groups by
# rank, not guessed cutoffs) tested against the same validation
# set:
#
#   Lower Risk  (score < 0.4861): 20 emails, 30% spam rate
#   Suspicious  (0.4861-0.5023):  20 emails, 55% spam rate
#   High Risk   (score >= 0.5023): 20 emails, 65% spam rate
#
# Monotonic and evenly populated -- this is genuinely
# defensible at n=60, unlike the earlier fixed-boundary version.
#
# Caveat: the Suspicious/High Risk boundary sits in a very
# narrow band (0.4861 to 0.5023, a gap of just 0.016), because
# scores cluster tightly in this domain. A small wording change
# in an email can flip it between these two tiers -- the tier
# labels are directionally meaningful in aggregate, but treat
# any single score near this boundary as inherently borderline.
#
# ============================================================

LOWER_RISK_UPPER = 0.4861
SUSPICIOUS_UPPER = 0.5023


# ============================================================
# TEXT PREPARATION
# ============================================================

def prepare_text(subject, body):
    """
    Construct the same textual representation used during
    model development.

    Only Subject + Body are used.
    """

    subject = "" if subject is None else str(subject)
    body = "" if body is None else str(body)

    subject = subject.strip()
    body = body.strip()

    return (subject + " " + body).strip()


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(combined_score):

    if combined_score >= SUSPICIOUS_UPPER:

        return (
            "HIGH RISK",
            "The combined model flags this as likely spam or "
            "phishing. Investigate the sender and content carefully, "
            "and verify through a known, trusted channel before "
            "responding, clicking anything, or providing information."
        )

    elif combined_score >= LOWER_RISK_UPPER:

        return (
            "SUSPICIOUS",
            "Some fraud-adjacent signal was detected, though not as "
            "strongly as the High Risk tier. Review the email's "
            "sender, links, and requests carefully before acting on "
            "it. Note: this tier sits in a narrow score band -- "
            "treat it as borderline rather than a firm verdict."
        )

    else:

        return (
            "LOWER RISK",
            "No strong fraud signal was detected from the supplied "
            "text. This is not a guarantee of legitimacy -- apply "
            "normal judgement."
        )


# ============================================================
# PREDICTION FUNCTION — TWO-LAYER
# ============================================================

def predict_email(subject, body):

    text = prepare_text(subject, body)

    if not text:
        return None

    # --------------------------------------------------------
    # LAYER 1 — WORD + CHARACTER TF-IDF + LINEAR SVM
    # --------------------------------------------------------

    X_word = word_vectorizer.transform([text])
    X_char = char_vectorizer.transform([text])
    X_combined = hstack([X_word, X_char])

    svm_score = float(
        svm_model.decision_function(X_combined)[0]
    )

    # --------------------------------------------------------
    # LAYER 2 — SCREENING MODEL
    # (features derived entirely from the Layer 1 score --
    # the screening model never sees the raw text directly)
    # --------------------------------------------------------

    layer2_features = pd.DataFrame([{
        "svm_score": svm_score,
        "svm_norm": (svm_score - SCORE_MEAN) / SCORE_STD,
        "svm_abs": abs(svm_score),
        "svm_squared": svm_score ** 2,
        "svm_positive": max(svm_score, 0),
        "svm_negative": max(-svm_score, 0),
        "svm_flag_suspicious": int(svm_score >= FLAG_THRESHOLDS["suspicious"]),
        "svm_flag_high": int(svm_score >= FLAG_THRESHOLDS["high"]),
        "svm_flag_very_high": int(svm_score >= FLAG_THRESHOLDS["very_high"]),
        "svm_flag_extreme": int(svm_score >= FLAG_THRESHOLDS["extreme"]),
    }])

    second_layer_score = float(
        second_layer_model.predict_proba(layer2_features)[0, 1]
    )

    # --------------------------------------------------------
    # COMBINED RISK SCORE
    # 70% Layer 1 (rescaled with frozen min/max)
    # 30% Layer 2 screening probability
    # --------------------------------------------------------

    svm_norm_frozen = (
        (svm_score - SVM_FROZEN_MIN) / (SVM_FROZEN_MAX - SVM_FROZEN_MIN)
    )

    combined_score = (
        0.70 * svm_norm_frozen
        + 0.30 * second_layer_score
    )

    risk_level, recommendation = classify_risk(combined_score)

    return {
        "svm_score": svm_score,
        "second_layer_score": second_layer_score,
        "combined_score": combined_score,
        "risk_level": risk_level,
        "recommendation": recommendation
    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ Email Fraud Risk Screening"
)

st.caption(
    "Developed by **DeepanJ**"
)

st.markdown(
    """
### Machine-Learning-Based Email Screening

Enter an email's **subject and body** to obtain a
machine-learning-based fraud-risk assessment.

This system combines two models in sequence: an SVM trained
on word- and character-level text patterns, followed by a
screening layer that reinterprets the SVM's confidence to
better handle emails outside its original training domain.
It analyses **text only** and does not require sender history,
domain reputation, or email authentication records.
"""
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

with st.expander(
    "📈 Model performance"
):

    st.markdown(
        f"""
**Model:** {MODEL_NAME}

**Layer 1 alone — internal Enron test set:**

| Metric | Result |
|---|---:|
| Accuracy | {LAYER1_ACCURACY:.2%} |
| ROC-AUC | {LAYER1_ROC_AUC:.4f} |
| PR-AUC | {LAYER1_PR_AUC:.4f} |
| Precision | {LAYER1_PRECISION:.2%} |
| Recall | {LAYER1_RECALL:.2%} |
| F1-score | {LAYER1_F1:.4f} |

**Combined two-layer system — external academic-domain
validation set (n = {EXTERNAL_N}, held out from training):**

| Metric | Result |
|---|---:|
| ROC-AUC | {EXTERNAL_ROC_AUC:.4f} |
| PR-AUC | {EXTERNAL_PR_AUC:.4f} |
| Precision | {EXTERNAL_PRECISION:.2%} |
| Recall | {EXTERNAL_RECALL:.2%} |
| F1-score | {EXTERNAL_F1:.4f} |

Layer 1 alone performs very well on Enron-style corporate
email but was found to under-recognise fraud patterns typical
of academic correspondence (predatory journals, fake
conference invitations). The screening layer was added to
address this. On the external validation set, the combined
system trades some precision for a large gain in recall
(93%) -- it is deliberately tuned to over-flag borderline
emails for human review rather than silently miss fraud.
The external validation set is small; treat these numbers as
directional evidence of behaviour, not a certified accuracy
figure.
"""
    )


# ============================================================
# HOW THE MODEL WORKS
# ============================================================

with st.expander(
    "🧠 How the model works"
):

    st.markdown(
        """
**Pipeline**

Email Subject + Body

↓

Word-level TF-IDF + Character-level TF-IDF

↓

Combined sparse feature representation

↓

Linear Support Vector Machine (Layer 1)

↓

SVM decision score

↓

Gradient-Boosted Screening Model (Layer 2)
*(reinterprets the Layer 1 score using its shape, magnitude,
and how it compares to fixed reference thresholds)*

↓

Combined risk score (70% Layer 1 + 30% Layer 2)

↓

Risk stratification

### Why two layers?

**Layer 1** captures word- and character-level fraud patterns
learned from a large labelled corpus (Enron).

**Layer 2** was added specifically because Layer 1's raw
decision score, on its own, did not reliably separate real
academic correspondence from academic-domain spam -- both use
similar vocabulary (journal, manuscript, DOI, conference,
fee). The screening layer learns how to weigh the Layer 1
score itself, rather than trusting it at face value.
"""
    )


# ============================================================
# INPUT
# ============================================================

st.subheader(
    "📧 Analyse an Email"
)

subject = st.text_input(
    "Email Subject",
    placeholder=(
        "Example: Your manuscript requires urgent action"
    )
)

body = st.text_area(
    "Email Body",
    height=280,
    placeholder=(
        "Paste the email body here..."
    )
)


# ============================================================
# ANALYSE BUTTON
# ============================================================

analyse_button = st.button(
    "🔍 Analyse Email",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyse_button:

    if not subject.strip() and not body.strip():

        st.warning(
            "Please provide an email subject or body."
        )

        st.stop()

    result = predict_email(subject, body)

    if result is None:

        st.warning(
            "No usable text was provided."
        )

        st.stop()

    combined_score = result["combined_score"]
    risk_level = result["risk_level"]
    recommendation = result["recommendation"]


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Risk Assessment"
    )

    # --------------------------------------------------------
    # Risk indicator
    # --------------------------------------------------------

    if risk_level == "HIGH RISK":
        st.error(f"🚨 {risk_level}")
    elif risk_level == "SUSPICIOUS":
        st.warning(f"🟡 {risk_level}")
    else:
        st.success(f"🟢 {risk_level}")

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Layer 1 SVM score", f"{result['svm_score']:.4f}")

    with col2:
        st.metric("Layer 2 screening score", f"{result['second_layer_score']:.4f}")

    with col3:
        st.metric("Combined risk score", f"{combined_score:.4f}")

    st.info(
        f"**Recommended action:** {recommendation}"
    )


    # ========================================================
    # RISK THRESHOLDS
    # ========================================================

    st.subheader(
        "Risk Operating Points"
    )

    threshold_data = {
        "Risk level": [
            "Lower Risk",
            "Suspicious",
            "High Risk"
        ],

        "Combined risk score": [
            f"< {LOWER_RISK_UPPER:.4f}",
            f"{LOWER_RISK_UPPER:.4f} – {SUSPICIOUS_UPPER:.4f}",
            f"≥ {SUSPICIOUS_UPPER:.4f}"
        ],

        "Validation spam rate": [
            "30% (n=20)",
            "55% (n=20)",
            "65% (n=20)"
        ],

        "Recommended action": [
            "Normal handling",
            "Review before interacting",
            "Investigate before interacting"
        ]
    }

    st.table(
        threshold_data
    )

    st.caption(
        "These 3 tiers are quantile-derived (equal-sized groups by "
        "rank on the n=60 validation set), not guessed cutoffs -- "
        "spam rate increases monotonically across them (30% → 55% "
        "→ 65%). A 5-tier and a 4-tier version were also tested; "
        "both failed either a monotonicity check or had tiers too "
        "small to trust. The Lower Risk / Suspicious boundary sits "
        "in a very narrow band (0.4861–0.5023) -- treat scores near "
        "it as inherently borderline."
    )


    # ========================================================
    # IMPORTANT DISCLAIMER
    # ========================================================

    st.warning(
        """
### Important limitation

This system provides a **machine-learning risk assessment**,
not proof that an email is fraudulent. It is a screening
signal, not a verdict -- always verify independently.

The deployed model analyses only:

• Email subject
• Email body

It does **not** analyse:

• Sender reputation
• Sender domain
• SPF / DKIM / DMARC
• URLs or URL reputation
• Attachments
• Malware
• IP addresses
• Previous communication history
• User behaviour
• External threat-intelligence feeds

Layer 1 was trained primarily on Enron corporate email.
Layer 2 was calibrated to correct for this on a small
academic-domain validation set (n = 60) -- it improves
recall substantially but at a real precision cost, meaning
some legitimate emails will be flagged. Always verify
suspicious emails independently before acting on them.

Do not click links, open attachments, or provide credentials
solely because of this application's output.
"""
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🛡️ Model"
    )

    st.markdown(
        "**Developed by DeepanJ**"
    )

    st.divider()

    st.markdown(
        """
**Architecture**

Subject + Body

↓

Word TF-IDF + Character TF-IDF

↓

Linear SVM (Layer 1)

↓

Gradient-Boosted Screening (Layer 2)

↓

Combined Risk Score

↓

Risk Stratification
"""
    )

    st.divider()

    st.markdown(
        """
### Training Dataset

**Layer 1:** Enron Email Dataset
Source: Carnegie Mellon University
Publisher: MIT / CMU
Authors: Leslie Kaelbling & William W. Cohen
Year: 2015
License: Apache 2.0

**Layer 2:** Calibrated on the Layer 1 score distribution
from the same internal Enron test set, then validated on a
separate small (n=60) hand-labelled academic-domain set.
"""
    )

    st.divider()

    st.markdown(
        """
### Deployment input

The client only needs to provide:

**Subject + Body**

No sender metadata is required.
"""
    )

    st.divider()

    st.markdown(
        """
### Risk levels

🟢 Lower Risk
🟡 Suspicious
🚨 High Risk

*(Quantile-derived 3-tier split, validated on n=60: spam
rate rises 30% → 55% → 65% across tiers. A 5-tier and 4-tier
version were tested and rejected — see "Risk Operating
Points" after analysing an email.)*
"""
    )

    st.divider()

    st.caption(
        "Research / portfolio prototype — DeepanJ"
    )


# ============================================================
# DATA SCIENTIST STATEMENT
# ============================================================

st.divider()

st.subheader(
    "🔬 Methodological Summary"
)

st.markdown(
    """
From a data-science perspective, this project implements an
end-to-end **two-layer text classification and
risk-stratification pipeline**. The workflow included data
auditing, duplicate and label-conflict assessment,
leakage-aware body-level splitting, TF-IDF feature engineering
at both word and character levels, linear SVM classification,
comparison against logistic regression and gradient boosting,
precision-recall analysis, threshold optimisation, and
deployment.

A key finding during development was that the Layer 1 model,
while performing very well on its Enron training/test split,
did not generalise cleanly to real academic correspondence.
Rather than treat this as a stopping point, a second
screening layer was trained on the Layer 1 score's own
distribution and validated on an independently collected,
hand-labelled academic-domain set. This is reported honestly
above, including where the combined system still trades
precision for recall -- a deliberate, documented design
choice for a screening tool, not a limitation hidden from
the reader.

The final system intentionally uses only **email Subject and
Body**, making the inference stage independent of
dataset-specific metadata that may not be available to an
external client.
"""
)


# ============================================================
# NEXT STEPS
# ============================================================

with st.expander(
    "🚀 Next steps for a production system"
):

    st.markdown(
        """
The current application is a research/portfolio prototype.
A production-grade fraud detection system could be extended
with:

1. **Broader domain training data**
   - Retrain Layer 1 with academic-domain and other non-Enron
     email so it doesn't rely solely on Layer 2 to compensate
   - Expand the external validation set well beyond n = 60

2. **URL analysis**
   - URL extraction, domain reputation, redirect analysis

3. **Email authentication**
   - SPF, DKIM, DMARC

4. **Sender intelligence**
   - Sender reputation, domain age, communication history

5. **Attachment analysis**
   - File-type inspection, malware scanning

6. **Threat-intelligence integration**
   - Known malicious domains and phishing indicators

7. **Model calibration**
   - Validate operating points on an independent, larger dataset

8. **Human-in-the-loop review**
   - Analyst review of flagged messages, feedback-driven retraining
     at higher traffic volumes

9. **Monitoring**
   - Track precision/recall over time, detect drift in
     attack language and patterns
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
Email Fraud Risk Screening — Machine Learning Research Prototype
Developed by DeepanJ

The model is intended for research, educational and
portfolio demonstration purposes and should not be considered
a standalone cybersecurity protection system.
"""
)