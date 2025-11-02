import joblib
import numpy as np
import pandas as pd
import os
import json

class FraudDetectionPipeline:
    def __init__(self, model_dir="models"):
        """Initialize and load the fraud detection pipeline"""
        self.model_dir = model_dir
        self.scaler = None
        self.model = None
        self.loaded = False
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load pre-trained scaler and XGBoost model from disk"""
        try:
            scaler_path = os.path.join(self.model_dir, "scaler_fused.pkl")
            
            # Check for scaler
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
            
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from: {scaler_path}")
            
            # Load XGBoost model - try multiple file names and formats
            import xgboost as xgb
            
            # Priority order: JSON > UBJ > PKL (JSON is most reliable)
            model_files = [
                ("xgb_fraud_model.json", "json"),
                ("xgb_model.json", "json"),
                ("xgb_fraud_model.ubj", "ubj"),
                ("xgb_model.ubj", "ubj"),
                ("xgb_fraud_model.pkl", "pkl"),
                ("xgb_model.pkl", "pkl")
            ]
            
            model_loaded = False
            
            for filename, file_type in model_files:
                model_path = os.path.join(self.model_dir, filename)
                
                if os.path.exists(model_path):
                    try:
                        if file_type in ["json", "ubj"]:
                            # Use XGBoost native load for JSON/UBJ
                            print(f"🔄 Loading model from: {filename} ({file_type.upper()} format)")
                            self.model = xgb.XGBClassifier()
                            self.model.load_model(model_path)
                        else:
                            # Use joblib for PKL
                            print(f"🔄 Loading model from: {filename} (PKL format)")
                            self.model = joblib.load(model_path)
                        
                        # Verify the model works
                        _ = self.model.get_booster()
                        print(f"✅ Model loaded successfully from: {filename}")
                        model_loaded = True
                        break
                        
                    except Exception as e:
                        print(f"⚠️  Failed to load {filename}: {str(e)}")
                        continue
            
            if not model_loaded:
                raise FileNotFoundError(
                    f"No valid XGBoost model found in {self.model_dir}. "
                    f"Looked for: xgb_fraud_model.json/ubj/pkl or xgb_model.json/ubj/pkl"
                )
            
            # Verify models loaded correctly
            if self.scaler is None:
                raise Exception("Scaler loaded but is None")
            if self.model is None:
                raise Exception("Model loaded but is None")
            
            # Check if model has required methods
            if not hasattr(self.model, 'predict_proba'):
                raise Exception("Model does not have predict_proba method")
            
            # Final verification - try to get booster
            try:
                _ = self.model.get_booster()
                print("✅ XGBoost booster verified and accessible")
            except Exception as e:
                raise Exception(f"Model booster not accessible: {str(e)}")
            
            self.loaded = True
            print(f"✅ Pipeline loaded successfully")
            print(f"   - Scaler expects {self.scaler.n_features_in_} features")
            print(f"   - Model type: {type(self.model).__name__}")
            
        except Exception as e:
            self.loaded = False
            print(f"❌ Error loading models: {str(e)}")
            raise
    
    def _generate_explanation(self, transaction_features, behavioral_features, probability, idx):
        """
        Generate human-readable explanation for why a transaction was flagged
        
        Args:
            transaction_features: Array of transaction features for this sample
            behavioral_features: Array of behavioral features for this sample
            probability: Fraud probability for this transaction
            idx: Index of the transaction
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Analyze transaction features (assuming standard order)
        # You should adjust these based on your actual feature names
        trans_feat = transaction_features
        behav_feat = behavioral_features
        
        # Example explanations based on feature analysis
        # Transaction amount analysis
        if len(trans_feat) > 0:
            amount = trans_feat[0]
            if amount > 500:
                explanations.append(f"💰 High transaction amount (${amount:.2f})")
            if amount > 1000:
                explanations.append(f"⚠️ Transaction amount significantly above normal threshold")
        
        # Time-based analysis (if time feature exists)
        if len(trans_feat) > 1:
            hour = trans_feat[1] if trans_feat[1] < 24 else trans_feat[1] % 24
            if hour < 6 or hour > 22:
                explanations.append(f"🕐 Transaction at unusual hour ({int(hour)}:00)")
        
        # Behavioral features analysis
        if len(behav_feat) > 0:
            # Login frequency
            if behav_feat[0] < 0.5:  # Assuming normalized
                explanations.append(f"🔐 Low account activity pattern detected")
            
            # Device/location changes
            if len(behav_feat) > 1 and behav_feat[1] > 0.7:
                explanations.append(f"📱 Transaction from unusual device or location")
        
        # Velocity checks
        if len(trans_feat) > 2:
            velocity = trans_feat[2]
            if velocity > 0.8:  # Assuming normalized velocity
                explanations.append(f"⚡ Multiple transactions in short time period")
        
        # Merchant risk analysis
        if len(trans_feat) > 3:
            merchant_risk = trans_feat[3]
            if merchant_risk > 0.6:
                explanations.append(f"🏪 High-risk merchant category")
        
        # Probability-based general explanations
        if probability > 0.9:
            explanations.append(f"🚨 Multiple fraud indicators present (confidence: {probability:.1%})")
        elif probability > 0.7:
            explanations.append(f"⚠️ Several suspicious patterns detected (confidence: {probability:.1%})")
        
        # If no specific explanations, provide general one
        if not explanations:
            explanations.append(f"⚠️ Transaction shows unusual patterns (fraud probability: {probability:.1%})")
        
        return explanations
    
    def predict(self, df_transaction, df_behavioral, threshold=0.5):
        """
        Predict fraud on transaction and behavioral data
        
        Args:
            df_transaction: DataFrame with transaction features
            df_behavioral: DataFrame with behavioral features
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Dictionary with predictions, probabilities, risk levels, statistics, and explanations
        """
        # Verify pipeline is loaded
        if not self.loaded:
            raise Exception("Pipeline not loaded. Models may not have loaded correctly.")
        
        if self.scaler is None or self.model is None:
            raise Exception("Scaler or model is None. Need to load models successfully first.")
        
        try:
            # Store feature names if available
            if isinstance(df_transaction, pd.DataFrame):
                self.feature_names = list(df_transaction.columns) + list(df_behavioral.columns)
            
            # Convert to numpy arrays if DataFrames
            if isinstance(df_transaction, pd.DataFrame):
                X_trans = df_transaction.values
            else:
                X_trans = df_transaction
                
            if isinstance(df_behavioral, pd.DataFrame):
                X_behav = df_behavioral.values
            else:
                X_behav = df_behavioral
            
            # Fuse features horizontally
            X = np.concatenate([X_trans, X_behav], axis=1)
            
            print(f"📊 Data shape: {X.shape}")
            print(f"   Expected features: {self.scaler.n_features_in_}")
            
            # Validate feature dimensions
            expected_features = self.scaler.n_features_in_
            if X.shape[1] != expected_features:
                raise ValueError(
                    f"Feature mismatch: Input has {X.shape[1]} features, "
                    f"but model expects {expected_features} features.\n"
                    f"Transaction features: {X_trans.shape[1]}, "
                    f"Behavioral features: {X_behav.shape[1]}"
                )
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Verify booster one more time before prediction
            try:
                _ = self.model.get_booster()
            except Exception as e:
                raise Exception(f"Model booster error before prediction: {str(e)}")
            
            # Predict probabilities
            proba = self.model.predict_proba(X_scaled)[:, 1]
            
            # Binary predictions based on threshold
            pred = (proba >= threshold).astype(int)
            
            # Assign risk levels
            risk_levels = np.where(
                proba < 0.5, "Low",
                np.where(proba < 0.8, "Medium", "High")
            )
            
            # Generate explanations for flagged transactions
            explanations = []
            for i in range(len(pred)):
                if pred[i] == 1 or proba[i] >= 0.5:  # Explain fraud or medium+ risk
                    exp = self._generate_explanation(X_trans[i], X_behav[i], proba[i], i)
                    explanations.append(exp)
                else:
                    explanations.append(["✅ Transaction appears normal"])
            
            # Calculate statistics
            fraud_mask = pred == 1
            stats = {
                "total_transactions": int(len(pred)),
                "fraud_detected": int(pred.sum()),
                "fraud_percentage": float(pred.mean() * 100),
                "avg_fraud_probability": float(proba[fraud_mask].mean()) if fraud_mask.sum() > 0 else 0.0,
                "high_risk_count": int((proba >= 0.8).sum()),
                "medium_risk_count": int(((proba >= 0.5) & (proba < 0.8)).sum()),
                "low_risk_count": int((proba < 0.5).sum())
            }
            
            print(f"✅ Prediction completed: {stats['fraud_detected']}/{stats['total_transactions']} frauds detected")
            
            return {
                "predictions": pred,
                "probabilities": proba,
                "risk_levels": risk_levels,
                "statistics": stats,
                "explanations": explanations
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            raise
    
    def get_model_info(self):
        """Get information about loaded models"""
        if not self.loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "expected_features": self.scaler.n_features_in_ if self.scaler else None,
            "model_type": type(self.model).__name__ if self.model else None
        }
        
        # Try to get booster info
        try:
            booster = self.model.get_booster()
            info["booster_status"] = "accessible"
        except:
            info["booster_status"] = "not_accessible"
        
        return info