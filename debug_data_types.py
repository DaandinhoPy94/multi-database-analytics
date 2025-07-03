"""
Debug data types to see what PostgreSQL returns
"""

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()
connection_string = os.getenv("SUPABASE_DATABASE_URL")

print("🔍 Debugging Data Types")
print("=" * 30)

try:
    conn = psycopg2.connect(connection_string)
    
    query = """
    SELECT 
        c.customer_id,
        c.customer_tier,
        (CURRENT_DATE - c.registration_date) as days_since_registration,
        COUNT(o.order_id) as total_orders
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id 
    GROUP BY c.customer_id, c.customer_tier, c.registration_date
    LIMIT 5
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("📊 Data Types:")
    print(df.dtypes)
    
    print(f"\n📋 Sample Data:")
    print(df.head())
    
    print(f"\n🔍 Days Since Registration Type: {type(df['days_since_registration'].iloc[0])}")
    print(f"🔍 Days Since Registration Value: {df['days_since_registration'].iloc[0]}")
    
except Exception as e:
    print(f"❌ Error: {e}")