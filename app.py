import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.pipeline import FraudDetectionPipeline

# ==========================================
# 1️⃣ Page Configuration
# ==========================================
st.set_page_config(
    page_title="Multi-View Fraud Detection",
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Header
st.title("🛡️ Multi-View Fraud Detection System")
st.markdown("**Transaction Monitoring** | Real-time fraud detection using advanced ML")
st.markdown("---")

# Sidebar
st.sidebar.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Fraud+Detection", use_container_width=True)
use_sample = st.sidebar.checkbox("🎯 Use Sample Data", value=True)
threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.5, 0.05)
batch_size = st.sidebar.number_input("Batch Size", 100, 10000, 1000, 100)

# ==========================================
# 2️⃣ Load Pipeline (with error handling)
# ==========================================
@st.cache_resource
def get_pipeline():
    try:
        pipeline = FraudDetectionPipeline(model_dir="models")
        return pipeline
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load pipeline: {str(e)}")
        st.error(f"**Pipeline Loading Error:**\n\n{str(e)}")
        st.info("💡 **Possible solutions:**\n"
                "1. Make sure your models folder contains:\n"
                "   - `scaler_fused.pkl`\n"
                "   - `xgb_fraud_model.json` (or .ubj or .pkl)\n"
                "2. Check that files are not corrupted\n"
                "3. Verify your model was saved correctly")
        return None

pipeline = get_pipeline()

if pipeline is not None and pipeline.loaded:
    model_info = pipeline.get_model_info()
    st.sidebar.success("✅ Pipeline loaded successfully")
    with st.sidebar.expander("🔍 Model Details"):
        st.json(model_info)
else:
    st.sidebar.error("❌ Pipeline failed to load")
    st.stop()

# ==========================================
# 3️⃣ File Upload / Sample Data Selection
# ==========================================
if use_sample:
    transaction_file = "data/transaction_sample.csv"
    behavioral_file = "data/behavioral_sample.csv"
    file_uploaded = True
else:
    transaction_file = st.sidebar.file_uploader("📊 Transaction CSV", type=["csv"])
    behavioral_file = st.sidebar.file_uploader("🧠 Behavioral CSV", type=["csv"])
    file_uploaded = transaction_file is not None and behavioral_file is not None

# ==========================================
# 4️⃣ Run Prediction
# ==========================================
if st.sidebar.button("🚀 Analyze Fraud", type="primary") and file_uploaded:
    with st.spinner("🔄 Analyzing transactions..."):
        try:
            # Load data
            df_tr = pd.read_csv(transaction_file)
            df_beh = pd.read_csv(behavioral_file)
            
            st.info(f"📊 Loaded {len(df_tr)} transactions and {len(df_beh)} behavioral records")
            
            # Validate data shapes
            if len(df_tr) != len(df_beh):
                st.error(f"❌ Data mismatch: {len(df_tr)} transactions vs {len(df_beh)} behavioral records")
                st.stop()
            
            # Show data preview
            with st.expander("🔍 Data Preview"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Transaction Data (first 5 rows)**")
                    st.dataframe(df_tr.head(), use_container_width=True)
                with col2:
                    st.write("**Behavioral Data (first 5 rows)**")
                    st.dataframe(df_beh.head(), use_container_width=True)
            
            # Run prediction
            results = pipeline.predict(df_tr, df_beh, threshold=threshold)

            y_pred = results["predictions"]
            y_proba = results["probabilities"]
            risk_levels = results["risk_levels"]
            stats = results["statistics"]
            explanations = results["explanations"]

            # ==============================
            # Metrics Dashboard
            # ==============================
            st.success("✅ Analysis completed successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", stats['total_transactions'])
            col2.metric("Fraud Detected", stats['fraud_detected'], delta=f"{stats['fraud_percentage']:.1f}%")
            col3.metric("High Risk", stats['high_risk_count'])
            col4.metric("Avg Fraud Prob", f"{stats['avg_fraud_probability']:.2%}")

            st.markdown("---")

            # ==============================
            # Visualizations
            # ==============================
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=y_proba, nbinsx=50, marker_color='#667eea'))
                fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
                fig.update_layout(
                    title="Fraud Probability Distribution", 
                    xaxis_title="Probability", 
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_chart2:
                counts = [
                    list(risk_levels).count('Low'), 
                    list(risk_levels).count('Medium'), 
                    list(risk_levels).count('High')
                ]
                fig2 = go.Figure(go.Pie(
                    labels=['Low', 'Medium', 'High'], 
                    values=counts, 
                    hole=0.4,
                    marker_colors=['#10b981', '#f59e0b', '#ef4444']
                ))
                fig2.update_layout(title="Risk Category Distribution", height=400)
                st.plotly_chart(fig2, use_container_width=True)

            # ==============================
            # Detailed results table with explanations
            # ==============================
            results_df = pd.DataFrame({
                'Transaction_ID': range(1, len(y_pred)+1),
                'Fraud_Probability': y_proba,
                'Predicted_Label': ['🚨 Fraud' if p==1 else '✅ Normal' for p in y_pred],
                'Risk_Level': risk_levels
            })

            st.subheader("📋 Detailed Results (Top 100)")
            st.dataframe(
                results_df.head(100).style.background_gradient(
                    subset=['Fraud_Probability'], 
                    cmap='RdYlGn_r'
                ), 
                use_container_width=True, 
                height=400
            )

            # ==============================
            # Fraud Explanations Section (NEW!)
            # ==============================
            st.markdown("---")
            st.subheader("🔍 Fraud Detection Explanations")
            st.markdown("Click on any flagged transaction to see why it was marked as suspicious:")
            
            # Filter for suspicious transactions
            suspicious_indices = [i for i, (pred, prob) in enumerate(zip(y_pred, y_proba)) if pred == 1 or prob >= 0.5]
            
            if suspicious_indices:
                # Create tabs for high-risk, medium-risk
                high_risk_indices = [i for i in suspicious_indices if y_proba[i] >= 0.8]
                medium_risk_indices = [i for i in suspicious_indices if 0.5 <= y_proba[i] < 0.8]
                
                tab1, tab2 = st.tabs([f"🚨 High Risk ({len(high_risk_indices)})", f"⚠️ Medium Risk ({len(medium_risk_indices)})"])
                
                with tab1:
                    if high_risk_indices:
                        st.markdown("**High-risk transactions require immediate attention:**")
                        for idx in high_risk_indices[:10]:  # Show top 10
                            with st.expander(f"Transaction #{idx + 1} - Probability: {y_proba[idx]:.1%}", expanded=(idx == high_risk_indices[0])):
                                st.markdown(f"**Fraud Score:** {y_proba[idx]:.1%}")
                                st.markdown(f"**Risk Level:** {risk_levels[idx]}")
                                st.markdown("**Why this transaction was flagged:**")
                                for explanation in explanations[idx]:
                                    st.markdown(f"- {explanation}")
                    else:
                        st.info("No high-risk transactions detected")
                
                with tab2:
                    if medium_risk_indices:
                        st.markdown("**Medium-risk transactions for review:**")
                        for idx in medium_risk_indices[:10]:  # Show top 10
                            with st.expander(f"Transaction #{idx + 1} - Probability: {y_proba[idx]:.1%}"):
                                st.markdown(f"**Fraud Score:** {y_proba[idx]:.1%}")
                                st.markdown(f"**Risk Level:** {risk_levels[idx]}")
                                st.markdown("**Why this transaction was flagged:**")
                                for explanation in explanations[idx]:
                                    st.markdown(f"- {explanation}")
                    else:
                        st.info("No medium-risk transactions detected")
            else:
                st.success("🎉 No suspicious transactions detected!")

            # ==============================
            # Download results with explanations
            # ==============================
            st.markdown("---")
            
            # Add explanations to download
            results_df['Explanation'] = ['; '.join(exp) for exp in explanations]
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download All Predictions (with Explanations)", 
                csv, 
                "fraud_predictions_explained.csv", 
                "text/csv",
                use_container_width=True
            )

        except FileNotFoundError as e:
            st.error(f"❌ File not found: {str(e)}")
            st.info("💡 Make sure your data files exist in the correct directory")
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
            
            # Additional debugging info
            with st.expander("🔧 Debug Information"):
                st.write("**Pipeline Status:**", pipeline.loaded)
                st.write("**Model Info:**", pipeline.get_model_info())

else:
    if not file_uploaded:
        st.info("👈 Upload CSV files or use sample data to begin analysis")
    else:
        st.info("👈 Click '🚀 Analyze Fraud' button to start analysis")
    
    # Show instructions
    with st.expander("📚 How to Use This System"):
        st.markdown("""
        ### Steps to Analyze Fraud:
        
        1. **Load Data:**
           - Check "Use Sample Data" for demo, OR
           - Upload your own CSV files (Transaction + Behavioral)
        
        2. **Adjust Settings:**
           - Set fraud threshold (0.0 - 1.0)
           - Configure batch size if needed
        
        3. **Run Analysis:**
           - Click "🚀 Analyze Fraud" button
           - Wait for processing
        
        4. **Review Results:**
           - View metrics dashboard
           - Explore visualizations
           - **NEW: Read detailed explanations** for flagged transactions
           - Download predictions with explanations
        
        ### Required CSV Format:
        - Transaction CSV: Numeric features (e.g., amount, time, etc.)
        - Behavioral CSV: User behavior features
        - Both files must have same number of rows
        
        ### Explainable AI Feature:
        This system provides **human-readable explanations** for each fraud detection:
        - 💰 Transaction amount analysis
        - 🕐 Time-based patterns
        - 📱 Device/location anomalies
        - ⚡ Velocity checks
        - 🏪 Merchant risk assessment
        
        Understanding **why** a transaction was flagged helps investigators make better decisions!
        """)
    
    # Feature highlights
    with st.expander("✨ Key Features"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Multi-View Learning**
            - Combines transaction + behavioral data
            - Advanced feature fusion
            - XGBoost ensemble model
            
            **📊 Real-Time Detection**
            - Instant fraud scoring
            - Risk categorization (Low/Medium/High)
            - Adjustable threshold
            """)
        with col2:
            st.markdown("""
            **🔍 Explainable AI**
            - Understand why transactions are flagged
            - Human-readable explanations
            - Evidence-based decisions
            
            **📈 Professional Interface**
            - Interactive visualizations
            - Detailed reports
            - CSV export with explanations
            """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    st.markdown("*Developed By Y K K Bandara*")
    st.caption("Multi-View Fraud Detection System v2.0 with Explainable AI")