
# Email Fraud Risk Screening System

## Developed by DeepanJ

A machine-learning-based email fraud screening prototype
using only the textual content of an email.

## Final Model

Word TF-IDF + Character TF-IDF + Linear SVM

### Pipeline

Email Subject + Body
→ Word TF-IDF
→ Character TF-IDF
→ Combined sparse representation
→ Linear SVM
→ Decision Score
→ Risk Stratification

## Input

The deployed application requires only:

- Email Subject
- Email Body

No Enron-specific metadata is required at prediction time.

## Model Performance

Accuracy: 99.67%

ROC-AUC: 0.9909

PR-AUC: 0.6962

Precision: 70.05%

Recall: 59.78%

F1-score: 64.51%

Because fraud is a minority class, precision, recall,
F1-score and PR-AUC are important evaluation metrics.

## Risk Operating Points

Suspicious: 0.0282

High Risk: 0.2254

Very High Risk: 0.4338

Extreme Risk: 0.7718

These are Linear SVM decision-score thresholds and
are not calibrated probabilities.

## Dataset

Enron Email Dataset

Source:
https://www.cs.cmu.edu/~enron/

Publisher:
MIT / Carnegie Mellon University

Authors:
Leslie Kaelbling and William W. Cohen

Year:
2015

License:
Apache 2.0

The deployed classifier was trained using a labelled,
processed derivative of the Enron email dataset.

## Methodological Summary

This project implements an end-to-end text classification
and risk-stratification workflow including data auditing,
duplicate and label-conflict assessment, leakage-aware
body-level splitting, word- and character-level TF-IDF
feature engineering, Linear SVM classification, model
comparison, precision-recall analysis, threshold optimisation
and deployment.

The inference stage intentionally uses only Subject and Body
so that the system does not depend on dataset-specific
metadata that may not be available to an external client.

## Limitations

The current prototype does not analyse:

- Sender reputation
- SPF
- DKIM
- DMARC
- URL reputation
- Attachments
- Malware
- IP addresses
- Previous communication history
- User behaviour
- External threat intelligence

The output should therefore be treated as a screening signal,
not definitive proof that an email is fraudulent.

## Future Work

Potential extensions include:

1. URL analysis
2. Sender and domain reputation
3. SPF/DKIM/DMARC analysis
4. Attachment analysis
5. Threat-intelligence integration
6. Probability calibration
7. External validation
8. Human-in-the-loop review
9. Model and data-drift monitoring
10. Periodic model retraining

## Running the Application

Install dependencies:

python3 -m pip install --user -r requirements.txt

Run the application:

python3 -m streamlit run app.py

## Developer

DeepanJ

Research / Portfolio Machine Learning Prototype

## If you use this two-layer architecture or reference this approach, please credit: DeepanJ, "Email Fraud Risk Screening — Two-Layer SVM + Gradient-Boosted Screening," GitHub, 2026, [repo link]
