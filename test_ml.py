"""
Test AI-Enhanced Analytics Platform
First ML model test!
"""

import os
from dotenv import load_dotenv
from src.ml_models import get_ml_insights, CustomerChurnPredictor

# Load environment
load_dotenv()
connection_string = os.getenv("SUPABASE_DATABASE_URL")

print("🚀 Testing AI-Enhanced Analytics Platform")
print("=" * 50)

# Test 1: Automated Insights
print("\n🧠 Test 1: Generating AI Insights...")
try:
    insights = get_ml_insights(connection_string)
    print(f"✅ Generated {len(insights)} AI insights:")
    
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. 🎯 {insight['title']}")
        print(f"   📊 {insight['message']}")
        print(f"   💡 Action: {insight['action']}")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Customer Churn Model Setup
print(f"\n🤖 Test 2: Setting up Customer Churn Predictor...")
try:
    churn_model = CustomerChurnPredictor(connection_string)
    print("✅ Churn model initialized successfully!")
    
    # Load training data to see what we're working with
    print("\n📊 Loading customer data for analysis...")
    df = churn_model.load_training_data()
    
    print(f"\n🎯 Dataset Summary:")
    print(f"   👥 Total customers: {len(df)}")
    print(f"   📈 Churn rate: {df['is_churned'].mean():.1%}")
    print(f"   💰 Avg total spent: €{df['total_spent'].mean():,.2f}")
    print(f"   🛒 Avg total orders: {df['total_orders'].mean():.1f}")
    
    print("\n✅ ML infrastructure is ready for training!")
    
except Exception as e:
    print(f"❌ Error: {e}")

print(f"\n🎉 AI Platform Test Complete!")
print("Ready to train full ML models and integrate with dashboard! 🚀")