import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_URL = "http://backend:8000/predict"
EXPLAIN_URL = "http://backend:8000/explain"
SHAP_URL = "http://backend:8000/shap"
BATCH_URL = "http://backend:8000/batch_predict"

st.set_page_config(
    page_title="AI Personal Finance Platform",
    page_icon="💳",
    layout="wide"
)

if "history" not in st.session_state:
    st.session_state.history = []

if "last_transaction" not in st.session_state:
    st.session_state.last_transaction = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "batch_result_df" not in st.session_state:
    st.session_state.batch_result_df = None

st.title("💳 AI Personal Finance Fraud Detection Platform")
st.caption("Machine Learning + FastAPI + Gemini AI + SHAP Explainability + Batch Fraud Detection")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Transaction Prediction",
    "📂 Batch CSV Prediction",
    "📊 Transaction History"
])

# ==========================================================
# TAB 1 — SINGLE TRANSACTION PREDICTION
# ==========================================================

with tab1:
    st.sidebar.title("Transaction Input")

    step = st.sidebar.number_input("Step", min_value=1, value=1)

    transaction_type = st.sidebar.selectbox(
        "Transaction Type",
        ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    )

    amount = st.sidebar.number_input("Amount", min_value=0.0, value=50000.0)

    nameOrig = st.sidebar.text_input("Sender ID", value="C12345")
    oldbalanceOrg = st.sidebar.number_input("Sender Old Balance", min_value=0.0, value=100000.0)
    newbalanceOrig = st.sidebar.number_input("Sender New Balance", min_value=0.0, value=50000.0)

    nameDest = st.sidebar.text_input("Receiver ID", value="C67890")
    oldbalanceDest = st.sidebar.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
    newbalanceDest = st.sidebar.number_input("Receiver New Balance", min_value=0.0, value=50000.0)

    predict_button = st.sidebar.button("🔍 Predict Fraud", use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Platform Status", "Online")
    col2.metric("Model", "Random Forest")
    col3.metric("API", "FastAPI")
    col4.metric("Explainability", "Gemini + SHAP")

    st.divider()

    data = {
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "nameOrig": nameOrig,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "nameDest": nameDest,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }

    st.subheader("Transaction Summary")

    summary_df = pd.DataFrame({
        "Field": [
            "Transaction Type",
            "Amount",
            "Sender Old Balance",
            "Sender New Balance",
            "Receiver Old Balance",
            "Receiver New Balance"
        ],
        "Value": [
            transaction_type,
            f"${amount:,.2f}",
            f"${oldbalanceOrg:,.2f}",
            f"${newbalanceOrig:,.2f}",
            f"${oldbalanceDest:,.2f}",
            f"${newbalanceDest:,.2f}"
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

    if predict_button:
        try:
            with st.spinner("Analyzing transaction with ML model..."):
                response = requests.post(API_URL, json=data, timeout=20)

            if response.status_code == 200:
                result = response.json()

                st.session_state.last_transaction = data
                st.session_state.last_result = result

                history_record = {
                    "Type": transaction_type,
                    "Amount": amount,
                    "Fraud Prediction": result["fraud_prediction"],
                    "Fraud Probability": float(result["fraud_probability"]),
                    "Risk Level": result["risk_level"]
                }

                st.session_state.history.append(history_record)

            else:
                st.error("API returned an error.")
                st.write(response.text)

        except Exception as e:
            st.error("Could not connect to FastAPI backend.")
            st.write(e)

    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        risk_level = result["risk_level"]
        probability = float(result["fraud_probability"])

        st.subheader("Fraud Risk Result")

        r1, r2, r3 = st.columns(3)

        prediction_label = "Fraud" if result["fraud_prediction"] == 1 else "Not Fraud"

        r1.metric("Fraud Prediction", prediction_label)
        r2.metric("Fraud Probability", f"{probability:.4f}")
        r3.metric("Risk Level", risk_level)

        if risk_level == "HIGH":
            st.error("🚨 HIGH RISK: This transaction may be fraudulent.")
        elif risk_level == "MEDIUM":
            st.warning("⚠️ MEDIUM RISK: This transaction needs review.")
        else:
            st.success("✅ LOW RISK: This transaction appears safe.")

        st.subheader("Fraud Risk Gauge")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Fraud Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 40], "color": "lightgreen"},
                    {"range": [40, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ]
            }
        ))

        st.plotly_chart(gauge, use_container_width=True)

        chart_df = pd.DataFrame({
            "Category": ["Safe", "Fraud"],
            "Probability": [1 - probability, probability]
        })

        st.subheader("Safe vs Fraud Probability")
        st.bar_chart(chart_df.set_index("Category"))

        st.subheader("Rule-Based Fraud Explanation")

        explanation = f"""
This transaction has been classified as **{risk_level} risk**.

Reason:
- Transaction type: **{transaction_type}**
- Amount: **${amount:,.2f}**
- Fraud probability: **{probability:.4f}**
- Sender balance changed from **${oldbalanceOrg:,.2f}** to **${newbalanceOrig:,.2f}**
- Receiver balance changed from **${oldbalanceDest:,.2f}** to **${newbalanceDest:,.2f}**

Recommendation:
"""

        if risk_level == "HIGH":
            explanation += "Block this transaction and send it for manual fraud review."
        elif risk_level == "MEDIUM":
            explanation += "Allow only after extra verification such as OTP or user confirmation."
        else:
            explanation += "Transaction appears safe, but continue normal monitoring."

        st.info(explanation)

        st.subheader("🤖 Gemini AI Fraud Explanation")

        if st.button("Generate Gemini Explanation", use_container_width=True):
            try:
                with st.spinner("Generating Gemini AI explanation..."):
                    explain_response = requests.post(
                        EXPLAIN_URL,
                        json=st.session_state.last_transaction,
                        timeout=90
                    )

                if explain_response.status_code == 200:
                    explain_result = explain_response.json()
                    st.success("Gemini AI Analysis Complete")
                    st.info(explain_result["ai_explanation"])
                else:
                    st.error("Gemini explanation API returned an error.")
                    st.write(explain_response.text)

            except Exception as e:
                st.error("Gemini explanation request failed.")
                st.write(e)

        st.subheader("📊 SHAP Explainability Dashboard")

        if st.button("Generate SHAP Explanation", use_container_width=True):
            try:
                with st.spinner("Generating SHAP feature impact..."):
                    shap_response = requests.post(
                        SHAP_URL,
                        json=st.session_state.last_transaction,
                        timeout=60
                    )

                if shap_response.status_code == 200:
                    shap_result = shap_response.json()
                    shap_df = pd.DataFrame(shap_result["shap_values"])

                    st.dataframe(shap_df, use_container_width=True)

                    st.subheader("Top Fraud Feature Impacts")

                    chart_df = shap_df.head(10).set_index("feature")
                    st.bar_chart(chart_df["impact"])

                    st.caption(
                        "Positive impact values increase fraud risk. Negative impact values reduce fraud risk."
                    )

                else:
                    st.error("SHAP API returned an error.")
                    st.write(shap_response.text)

            except Exception as e:
                st.error("SHAP explanation failed.")
                st.write(e)

        st.subheader("Raw API Response")
        st.json(result)

# ==========================================================
# TAB 2 — BATCH CSV PREDICTION
# ==========================================================

with tab2:
    st.subheader("📂 Batch CSV Fraud Prediction")

    st.write("""
Upload a CSV file containing multiple transactions.  
The system will send the file to FastAPI, predict fraud for each transaction, and return batch analytics.
""")

    st.info("""
Required CSV columns:
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest
""")

    uploaded_file = st.file_uploader(
        "Upload transaction CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            preview_df = pd.read_csv(uploaded_file)

            st.subheader("CSV Preview")
            st.dataframe(preview_df.head(10), use_container_width=True)

            uploaded_file.seek(0)

            if st.button("🚀 Run Batch Fraud Prediction", use_container_width=True):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv"
                        )
                    }

                    with st.spinner("Running batch fraud prediction..."):
                        batch_response = requests.post(
                            BATCH_URL,
                            files=files,
                            timeout=120
                        )

                    if batch_response.status_code == 200:
                        batch_result = batch_response.json()

                        st.success("Batch prediction completed successfully.")

                        if "records" in batch_result:
                            result_df = pd.DataFrame(batch_result["records"])

                        elif "predictions" in batch_result:
                            result_df = pd.DataFrame(batch_result["predictions"])

                        else:
                            st.warning(
                                "Backend returned only summary data. "
                                "For full row-by-row predictions, update backend batch_predict.py to return records."
                            )

                            result_df = pd.DataFrame([{
                                "message": batch_result.get("message"),
                                "total_transactions": batch_result.get("total_transactions"),
                                "fraud_count": batch_result.get("fraud_count"),
                                "download_file": batch_result.get("download_file")
                            }])

                            st.subheader("Raw Batch API Response")
                            st.json(batch_result)

                        st.session_state.batch_result_df = result_df

                    else:
                        st.error("Batch prediction API returned an error.")
                        st.write(batch_response.text)

                except Exception as e:
                    st.error("Batch prediction request failed.")
                    st.write(e)

        except Exception as e:
            st.error("Could not read uploaded CSV file.")
            st.write(e)

    if st.session_state.batch_result_df is not None:
        result_df = st.session_state.batch_result_df

        st.divider()
        st.subheader("Batch Prediction Results")
        st.dataframe(result_df, use_container_width=True)

        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction Results CSV",
            data=csv_data,
            file_name="batch_fraud_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.subheader("Batch Risk Analytics")

        total_transactions = len(result_df)

        fraud_col = None
        probability_col = None
        risk_col = None
        type_col = None
        amount_col = None

        for col in result_df.columns:
            lower_col = col.lower()

            if lower_col in ["fraud_prediction", "prediction", "isfraud"]:
                fraud_col = col

            if lower_col in ["fraud_probability", "probability", "risk_score"]:
                probability_col = col

            if lower_col in ["risk_level", "risk"]:
                risk_col = col

            if lower_col == "type":
                type_col = col

            if lower_col == "amount":
                amount_col = col

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Rows", total_transactions)

        if fraud_col:
            fraud_count = int(result_df[fraud_col].sum())
            fraud_rate = (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0

            c2.metric("Fraud Count", fraud_count)
            c3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        else:
            c2.metric("Fraud Count", "N/A")
            c3.metric("Fraud Rate", "N/A")

        if amount_col:
            avg_amount = result_df[amount_col].mean()
            c4.metric("Average Amount", f"${avg_amount:,.2f}")
        else:
            c4.metric("Average Amount", "N/A")

        if risk_col:
            st.subheader("Risk Level Distribution")

            risk_counts = result_df[risk_col].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]

            fig_risk = px.bar(
                risk_counts,
                x="Risk Level",
                y="Count",
                title="Batch Risk Level Summary"
            )

            st.plotly_chart(fig_risk, use_container_width=True)

        if probability_col:
            st.subheader("Fraud Probability Distribution")

            fig_prob = px.histogram(
                result_df,
                x=probability_col,
                nbins=20,
                title="Fraud Probability Distribution"
            )

            st.plotly_chart(fig_prob, use_container_width=True)

        if type_col and fraud_col:
            st.subheader("Fraud by Transaction Type")

            type_summary = result_df.groupby(type_col)[fraud_col].sum().reset_index()
            type_summary.columns = ["Transaction Type", "Fraud Count"]

            fig_type = px.bar(
                type_summary,
                x="Transaction Type",
                y="Fraud Count",
                title="Fraud Count by Transaction Type"
            )

            st.plotly_chart(fig_type, use_container_width=True)

        if probability_col:
            st.subheader("Top High-Risk Transactions")

            high_risk_df = result_df.sort_values(
                by=probability_col,
                ascending=False
            ).head(10)

            st.dataframe(high_risk_df, use_container_width=True)

# ==========================================================
# TAB 3 — TRANSACTION HISTORY
# ==========================================================

with tab3:
    st.subheader("Transaction Prediction History")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        risk_counts = history_df["Risk Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        st.subheader("Risk Level Summary")
        st.bar_chart(risk_counts.set_index("Risk Level"))

        st.subheader("Fraud Probability Trend")

        history_df["Transaction Number"] = range(1, len(history_df) + 1)

        fig_history = px.line(
            history_df,
            x="Transaction Number",
            y="Fraud Probability",
            markers=True,
            title="Fraud Probability Over Time"
        )

        st.plotly_chart(fig_history, use_container_width=True)

    else:
        st.info("No transactions predicted yet.")

st.divider()

st.subheader("Upcoming Platform Features")

st.write("""
Next upgrades:
- PostgreSQL database integration
- User authentication
- Docker deployment
- Azure cloud deployment
- MLOps model monitoring pipeline
""")