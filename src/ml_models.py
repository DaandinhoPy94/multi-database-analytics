"""
AI-Enhanced Analytics: Machine Learning Models
Customer Churn Prediction, Sales Forecasting, and Product Recommendations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import psycopg2
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

class CustomerChurnPredictor:
    """
    Machine Learning model for predicting customer churn
    Uses Random Forest with advanced feature engineering
    """
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
    def load_training_data(self):
        """Load and prepare training data from database"""
        print("🔄 Loading customer data for ML training...")
        
        query = """
        WITH customer_features AS (
            SELECT 
                c.customer_id,
                c.customer_tier,
                (CURRENT_DATE - c.registration_date) as days_since_registration,
                
                -- Order behavior features
                COALESCE(COUNT(o.order_id), 0) as total_orders,
                COALESCE(SUM(o.total_amount), 0) as total_spent,
                COALESCE(AVG(o.total_amount), 0) as avg_order_value,
                COALESCE(MAX(o.order_date), c.registration_date) as last_order_date,
                (CURRENT_DATE - COALESCE(MAX(o.order_date), c.registration_date)) as days_since_last_order,
                
                -- Recent activity (last 90 days)
                COUNT(CASE WHEN o.order_date >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) as orders_last_90_days,
                COALESCE(SUM(CASE WHEN o.order_date >= CURRENT_DATE - INTERVAL '90 days' THEN o.total_amount END), 0) as spent_last_90_days,
                
                -- Purchase diversity
                COUNT(DISTINCT oi.product_id) as unique_products,
                COUNT(DISTINCT o.payment_method) as payment_methods_used,
                
                -- Discount behavior
                COUNT(CASE WHEN o.discount_amount > 0 THEN 1 END) as discount_orders,
                COALESCE(AVG(CASE WHEN o.discount_amount > 0 THEN o.discount_amount END), 0) as avg_discount_used,
                
                -- Churn label: No orders in last 120 days
                CASE 
                    WHEN MAX(o.order_date) < CURRENT_DATE - INTERVAL '120 days' OR MAX(o.order_date) IS NULL THEN 1 
                    ELSE 0 
                END as is_churned
                
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.order_status IN ('confirmed', 'shipped', 'delivered')
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            GROUP BY c.customer_id, c.customer_tier, c.registration_date
        )
        SELECT * FROM customer_features
        WHERE days_since_registration >= 30  -- Only customers with 30+ days tenure
        """
        
        # Connect and load data
        conn = psycopg2.connect(self.connection_string)
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(df)} customer records")
        print(f"📊 Churn rate: {df['is_churned'].mean():.1%}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        print("🔄 Engineering features for ML...")
        
        # Create derived features
        df['order_frequency'] = np.where(df['days_since_registration'] > 0, 
                                       df['total_orders'] / (df['days_since_registration'] / 30), 0)
        
        df['recent_activity_ratio'] = np.where(df['total_orders'] > 0,
                                             df['orders_last_90_days'] / df['total_orders'], 0)
        
        df['spending_per_day'] = np.where(df['days_since_registration'] > 0,
                                        df['total_spent'] / df['days_since_registration'], 0)
        
        # Encode categorical variables
        le = LabelEncoder()
        df['customer_tier_encoded'] = le.fit_transform(df['customer_tier'])
        
        # Select feature columns (exclude ID and target)
        self.feature_columns = [
            'customer_tier_encoded', 'days_since_registration', 'total_orders',
            'total_spent', 'avg_order_value', 'days_since_last_order',
            'orders_last_90_days', 'spent_last_90_days', 'unique_products',
            'payment_methods_used', 'discount_orders', 'avg_discount_used',
            'order_frequency', 'recent_activity_ratio', 'spending_per_day'
        ]
        
        # Ensure all features exist and are numeric
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        print(f"✅ Feature engineering complete. Using {len(self.feature_columns)} features")
        return df
    
    def train_model(self):
        """Train the churn prediction model"""
        print("🤖 Training Customer Churn Prediction Model...")
        
        # Load and prepare data
        df = self.load_training_data()
        df = self.prepare_features(df)
        
        # Prepare training data
        X = df[self.feature_columns]
        y = df['is_churned']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"✅ Model Training Complete!")
        print(f"📊 Training Accuracy: {train_score:.3f}")
        print(f"📊 Test Accuracy: {test_score:.3f}")
        print(f"📊 AUC-ROC Score: {auc_score:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        self.is_trained = True
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'auc_score': auc_score,
            'feature_importance': feature_importance
        }
    
    def predict_churn_risk(self, customer_ids=None):
        """Predict churn risk for customers"""
        if not self.is_trained:
            raise ValueError("Model must be trained first. Call train_model()")
        
        print("🔮 Generating churn predictions...")
        
        # Load current customer data
        df = self.load_training_data()
        df = self.prepare_features(df)
        
        # Filter to specific customers if provided
        if customer_ids:
            df = df[df['customer_id'].isin(customer_ids)]
        
        # Only predict for non-churned customers
        active_customers = df[df['is_churned'] == 0].copy()
        
        if len(active_customers) == 0:
            return pd.DataFrame()
        
        # Prepare features
        X = active_customers[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Generate predictions
        churn_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create results DataFrame
        results = active_customers[['customer_id', 'customer_tier', 'total_orders', 'total_spent', 'days_since_last_order']].copy()
        results['churn_probability'] = churn_probabilities
        results['risk_level'] = pd.cut(
            churn_probabilities, 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        print(f"✅ Generated predictions for {len(results)} active customers")
        return results.sort_values('churn_probability', ascending=False)
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        print(f"📂 Model loaded from {filepath}")


class SalesForecaster:
    """
    Sales forecasting using time series analysis
    Predicts future revenue based on historical patterns
    """
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.model = None
        self.is_trained = False
    
    def load_sales_data(self):
        """Load daily sales data for forecasting"""
        print("📈 Loading sales data for forecasting...")
        
        query = """
        SELECT 
            DATE(order_date) as sales_date,
            COUNT(*) as order_count,
            SUM(total_amount) as daily_revenue,
            AVG(total_amount) as avg_order_value,
            COUNT(DISTINCT customer_id) as unique_customers,
            EXTRACT(DOW FROM order_date) as day_of_week,
            EXTRACT(MONTH FROM order_date) as month
        FROM orders
        WHERE order_status IN ('confirmed', 'shipped', 'delivered')
        GROUP BY DATE(order_date)
        ORDER BY sales_date
        """
        
        conn = psycopg2.connect(self.connection_string)
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df['sales_date'] = pd.to_datetime(df['sales_date'])
        print(f"✅ Loaded {len(df)} days of sales data")
        return df
    
    def create_forecast_features(self, df):
        """Create features for sales forecasting"""
        print("🔄 Creating forecasting features...")
        
        # Sort by date
        df = df.sort_values('sales_date').reset_index(drop=True)
        
        # Create lag features
        for lag in [1, 7, 14, 30]:
            df[f'revenue_lag_{lag}'] = df['daily_revenue'].shift(lag)
        
        # Rolling averages
        df['revenue_ma_7'] = df['daily_revenue'].rolling(window=7).mean()
        df['revenue_ma_30'] = df['daily_revenue'].rolling(window=30).mean()
        
        # Trend features
        df['revenue_trend_7'] = df['daily_revenue'] / df['revenue_ma_7']
        df['revenue_growth'] = df['daily_revenue'].pct_change()
        
        # Calendar features
        df['is_weekend'] = (df['day_of_week'].isin([0, 6])).astype(int)
        df['is_month_start'] = (df['sales_date'].dt.day <= 5).astype(int)
        df['is_month_end'] = (df['sales_date'].dt.day >= 25).astype(int)
        
        # Drop rows with NaN (due to lag features)
        df = df.dropna()
        
        print(f"✅ Created forecasting features. Dataset size: {len(df)} days")
        return df
    
    def train_forecasting_model(self):
        """Train the sales forecasting model"""
        print("🤖 Training Sales Forecasting Model...")
        
        # Load and prepare data
        df = self.load_sales_data()
        df = self.create_forecast_features(df)
        
        # Select features
        feature_columns = [
            'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_14', 'revenue_lag_30',
            'revenue_ma_7', 'revenue_ma_30', 'revenue_trend_7',
            'order_count', 'avg_order_value', 'unique_customers',
            'day_of_week', 'month', 'is_weekend', 'is_month_start', 'is_month_end'
        ]
        
        X = df[feature_columns]
        y = df['daily_revenue']
        
        # Split data (use last 20% for testing)
        split_index = int(len(df) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mape = mean_absolute_percentage_error(y_train, train_pred)
        test_mape = mean_absolute_percentage_error(y_test, test_pred)
        
        print(f"✅ Forecasting Model Training Complete!")
        print(f"📊 Training MAPE: {train_mape:.1%}")
        print(f"📊 Test MAPE: {test_mape:.1%}")
        
        self.feature_columns = feature_columns
        self.is_trained = True
        
        return {
            'train_mape': train_mape,
            'test_mape': test_mape,
            'last_actual': y_test.iloc[-1],
            'last_predicted': test_pred[-1]
        }
    
    def generate_forecast(self, days_ahead=30):
        """Generate future sales forecast"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        print(f"🔮 Generating {days_ahead}-day sales forecast...")
        
        # Load recent data
        df = self.load_sales_data()
        df = self.create_forecast_features(df)
        
        # Use last available data as starting point
        last_row = df.iloc[-1:].copy()
        forecasts = []
        
        for day in range(days_ahead):
            # Predict next day
            X = last_row[self.feature_columns]
            predicted_revenue = self.model.predict(X)[0]
            
            forecasts.append({
                'forecast_date': last_row['sales_date'].iloc[0] + pd.Timedelta(days=1),
                'predicted_revenue': predicted_revenue,
                'day_number': day + 1
            })
            
            # Update features for next prediction (simplified)
            last_row['revenue_lag_1'] = predicted_revenue
            last_row['sales_date'] = last_row['sales_date'] + pd.Timedelta(days=1)
            last_row['day_of_week'] = last_row['sales_date'].dt.dayofweek.iloc[0]
            last_row['month'] = last_row['sales_date'].dt.month.iloc[0]
        
        forecast_df = pd.DataFrame(forecasts)
        print(f"✅ Generated {len(forecast_df)}-day forecast")
        return forecast_df


def get_ml_insights(connection_string):
    """
    Generate automated ML insights for dashboard
    """
    print("🧠 Generating AI-powered business insights...")
    
    insights = []
    
    # Customer churn insights
    churn_model = CustomerChurnPredictor(connection_string)
    
    try:
        # Quick churn analysis without full training
        conn = psycopg2.connect(connection_string)
        
        # High-risk customer query
        risk_query = """
        SELECT 
            COUNT(*) as high_risk_customers
        FROM customer_ml_features
        WHERE days_since_last_order > 90 
        AND total_orders > 2
        AND days_since_last_order < 365
        """
        
        high_risk = pd.read_sql_query(risk_query, conn)
        high_risk_count = high_risk['high_risk_customers'].iloc[0]
        
        if high_risk_count > 0:
            insights.append({
                'type': 'warning',
                'title': 'Customer Retention Alert',
                'message': f'{high_risk_count} customers haven\'t ordered in 90+ days and may be at risk of churning.',
                'action': 'Consider targeted re-engagement campaigns.'
            })
        
        # Revenue trend insight
        trend_query = """
        SELECT 
            DATE_TRUNC('month', order_date) as month,
            SUM(total_amount) as monthly_revenue
        FROM orders
        WHERE order_status IN ('confirmed', 'shipped', 'delivered')
        AND order_date >= CURRENT_DATE - INTERVAL '3 months'
        GROUP BY DATE_TRUNC('month', order_date)
        ORDER BY month
        """
        
        trend_data = pd.read_sql_query(trend_query, conn)
        
        if len(trend_data) >= 2:
            latest_revenue = trend_data['monthly_revenue'].iloc[-1]
            previous_revenue = trend_data['monthly_revenue'].iloc[-2]
            growth_rate = (latest_revenue - previous_revenue) / previous_revenue * 100
            
            if growth_rate > 10:
                insights.append({
                    'type': 'success',
                    'title': 'Strong Growth Detected',
                    'message': f'Revenue increased by {growth_rate:.1f}% last month - excellent momentum!',
                    'action': 'Consider scaling successful marketing channels.'
                })
            elif growth_rate < -10:
                insights.append({
                    'type': 'warning',
                    'title': 'Revenue Decline Alert',
                    'message': f'Revenue decreased by {abs(growth_rate):.1f}% last month.',
                    'action': 'Investigate causes and implement recovery strategies.'
                })
        
        conn.close()
        
    except Exception as e:
        insights.append({
            'type': 'info',
            'title': 'ML Analysis',
            'message': 'Advanced AI insights will be available after model training.',
            'action': 'Train ML models for deeper analysis.'
        })
    
    print(f"✅ Generated {len(insights)} AI insights")
    return insights