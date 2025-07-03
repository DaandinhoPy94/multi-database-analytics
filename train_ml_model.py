"""
Train First ML Model - Customer Churn Predictor
Real machine learning in action!
"""

import os
from dotenv import load_dotenv
from src.ml_models import CustomerChurnPredictor

# Load environment
load_dotenv()
connection_string = os.getenv("SUPABASE_DATABASE_URL")

print("🤖 Training Customer Churn Prediction Model")
print("=" * 50)

# Initialize and train model
churn_model = CustomerChurnPredictor(connection_string)

print("\n🚀 Starting ML model training...")
try:
    # Train the model
    results = churn_model.train_model()
    
    print(f"\n🎯 Training Results:")
    print(f"   🎪 Training Accuracy: {results['train_accuracy']:.1%}")
    print(f"   🧪 Test Accuracy: {results['test_accuracy']:.1%}")
    print(f"   📊 AUC-ROC Score: {results['auc_score']:.3f}")
    
    print(f"\n🔥 Most Important Features for Churn Prediction:")
    for idx, row in results['feature_importance'].head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Generate predictions for high-risk customers
    print(f"\n🎯 Identifying High-Risk Customers...")
    predictions = churn_model.predict_churn_risk()
    
    high_risk = predictions[predictions['risk_level'] == 'High Risk']
    
    print(f"\n⚠️  HIGH RISK CUSTOMERS IDENTIFIED:")
    print(f"   🚨 Total high-risk customers: {len(high_risk)}")
    
    if len(high_risk) > 0:
        print(f"\n📋 Top 5 Customers at Risk:")
        for idx, customer in high_risk.head().iterrows():
            print(f"   Customer {customer['customer_id']} ({customer['customer_tier']}): "
                  f"{customer['churn_probability']:.1%} risk, "
                  f"€{customer['total_spent']:,.0f} total spent, "
                  f"{customer['days_since_last_order']} days since last order")
    
    # Save the trained model
    print(f"\n💾 Saving trained model...")
    churn_model.save_model('models/churn_model.joblib')
    
    print(f"\n🎉 SUCCESS! Your first ML model is trained and ready!")
    print(f"✅ Model can now predict customer churn with {results['test_accuracy']:.1%} accuracy")
    print(f"✅ {len(high_risk)} high-risk customers identified for retention campaigns")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🚀 Ready to integrate ML predictions into your dashboard!")