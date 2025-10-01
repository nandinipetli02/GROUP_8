\# GROUP 8: Multimodal Skin-Lesion Classification (Derm7pt)



\## 1. Introduction \& Objectives

\- Multiclass diagnosis + binary malignancy

\- Generative \& discriminative model components



\## 2. Data Overview

\- Clinical + dermoscopic images, 34 sites

\- Metadata: seven-point checklist + patient info

\- Official train/val/test splits



\## 3. Baselines \& Metrics

\- Image-only (CNN), metadata-only (LR, RF)

\- Metrics: Macro F1, ROC-AUC, sensitivity/specificity



\## 4. Fusion Architecture

\- Dual-CNN + metadata-MLP → late fusion classifier

\- Imbalance handling: class weights / focal loss



\## 5. Evaluation \& Next Steps

\- Ablations: modality vs metadata vs fusion

\- Explainability: Grad-CAM + SHAP

\- Lab mapping: EDA → baselines → fusion



