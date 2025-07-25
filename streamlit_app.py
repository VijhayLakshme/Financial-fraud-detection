import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fraud_detection import load_and_preprocess_data, train_models, train_autoencoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

st.set_page_config(page_title="💳 Financial Fraud Detection", layout="wide")
st.title("💳 Financial Fraud Detection Dashboard")

# Sidebar for CSV Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a bank transaction CSV file", type="csv")

if uploaded_file is not None:
    with st.spinner("Processing dataset..."):
        X, y, df = load_and_preprocess_data(uploaded_file)
        X_train, X_test = X[:6000], X[6000:]
        y_train, y_test = y[:6000], y[6000:]

        models = train_models(X_train, y_train)
        ae_model = train_autoencoder(X_train)

        model_name = st.selectbox("Select a Model", list(models.keys()) + ["Autoencoder"])

        if st.button("Run Detection"):
            if model_name == "Autoencoder":
                recon = ae_model.predict(X_test)
                loss = np.mean(np.square(recon - X_test), axis=1)
                threshold = np.percentile(loss, 95)
                preds = (loss > threshold).astype(int)
                y_scores = loss
            else:
                model = models[model_name]
                preds = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_scores = model.predict_proba(X_test)[:, 1]
                else:
                    y_scores = model.decision_function(X_test)

            # ===========================
            # 📊 OUTPUT + PERFORMANCE
            # ===========================

            # 1. Show Predictions Table
            st.subheader("🔍 Prediction Results")
            result_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': preds})
            st.dataframe(result_df)

            # 2. Accuracy Metric
            accuracy = round(np.mean(preds == y_test.values) * 100, 2)
            st.metric("✅ Accuracy", f"{accuracy}%")

            # 3. Confusion Matrix
            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(y_test, preds)
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax_cm)
            st.pyplot(fig_cm)

            # 4. Classification Report Table
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, preds, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            # 5. ROC Curve (only for supervised)
            if model_name != "Autoencoder":
                st.subheader("📊 ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
                ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve")
                ax_roc.legend()
                st.pyplot(fig_roc)

                # 6. Precision-Recall Curve
                st.subheader("🔵 Precision-Recall Curve")
                precision, recall, _ = precision_recall_curve(y_test, y_scores)
                fig_pr, ax_pr = plt.subplots()
                ax_pr.plot(recall, precision, color="purple")
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.set_title("Precision-Recall Curve")
                st.pyplot(fig_pr)

else:
    st.warning("📁 Please upload a CSV file to get started.")
