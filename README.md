Credit Card Fraud Detection System

A machine learning-powered web application for detecting fraudulent credit card transactions in real-time. Built with Streamlit and advanced ML models including XGBoost.

🚀 Live Demo

1.Access the application:** [http://192.168.1.2:8501](http://192.168.1.2:8501)

> Note: This URL is accessible on your local network.

2.https://huggingface.co/spaces/yenushka/credit-card-fraud-detection
> above URL can be used by public to view the model hosting public deployment

 📋 Features

- Real-time Fraud Detection**: Analyze transactions instantly with ML-powered predictions
- Transaction Analysis**: Comprehensive evaluation based on transaction and behavioral features
- Probability Scoring**: Get fraud probability percentages for each transaction
- Detailed Explanations**: Understand why a transaction was flagged as fraudulent
- Interactive Dashboard**: User-friendly Streamlit interface for easy data input and visualization

🛠️ Technologies Used

- **Python 3.12**
- **Streamlit**: Web application framework
- **XGBoost**: Primary ML model for fraud detection
- **scikit-learn**: Model evaluation and preprocessing
- **NumPy & Pandas**: Data manipulation
- **Joblib**: Model serialization

 📁 Project Structure


FRAUD_DETECTION_PROJECT/
├── data/
│   ├── behavioral_sample.csv
│   └── transaction_sample.csv
├── models/
│   ├── config_enhanced.json
│   ├── performance_metrics_enhanced.json
│   ├── scaler_fused.pkl
│   ├── X_test_enhanced.npy
│   ├── xgb_fraud_model.json
│   ├── xgb_fraud_model.pkl
│   ├── xgb_fraud_model.ubj
│   ├── y_pred_proba_enhanced.npy
│   └── y_test_enhanced.npy
├── utils/
│   ├── __pycache__/
│   └── pipeline.py
├── venv/
├── app.py
└── requirements.txt


🔧 Installation

 Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

 Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection-System.git
cd Credit-Card-Fraud-Detection-System


2 Create a virtual environment**
```bash
python -m venv venv

On Windows
venv\Scripts\activate
 On macOS/Linux
source venv/bin/activate
```

3. Install dependencies**
```bash
pip install -r requirements.txt
```

4. Run the application**
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

📦 Requirements

```
streamlit
xgboost
scikit-learn
pandas
numpy
joblib
```

 🎯 Usage

1. Launch the application using `streamlit run app.py`
2. Input transaction features including:
   - Transaction amount
   - Transaction features (V1-V28 from PCA transformation)
   - Behavioral features
3. Click "Analyze Transaction" to get predictions
4. Review the fraud probability and detailed explanations

Example Transaction Analysis

The system analyzes transactions based on:
- **Amount thresholds**: Flags high-value transactions (>$500, >$1000)
- **Feature patterns**: Detects anomalies in transaction characteristics
- **Behavioral indicators**: Identifies unusual user behavior patterns

🧠 Model Information

The fraud detection system uses an XGBoost classifier trained on credit card transaction data. The model provides:

- Binary classification (Fraud/Legitimate)
- Probability scores for risk assessment
- Feature-based explanations for predictions

 Model Performance

Model metrics are stored in `models/performance_metrics_enhanced.json` and include accuracy, precision, recall, and F1-score.

🔒 Security & Privacy

- All transaction data is processed locally
- No data is stored or transmitted to external servers
- Models are pre-trained and loaded from disk

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.


 🙏 Acknowledgments

- Credit card transaction dataset providers
- Streamlit community for the excellent framework
- XGBoost developers for the powerful ML library

 📧 Contact

For questions or support, please open an issue on GitHub or contact [kalindiyenushka@gmail.com]

---

⭐ If you found this project helpful, please consider giving it a star on GitHub!
