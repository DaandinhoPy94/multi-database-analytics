"""
Simple ML Test - Fixed SQL syntax
"""

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()
connection_string = os.getenv("SUPABASE_DATABASE_URL")

print("🚀 Testing Simple ML Query")
print("=" * 30)

# Test simple customer analysis
try:
    conn = psycopg2.connect(connection_string)
    
    # Simple query without complex date functions
    simple_query = """
    SELECT 
        c.customer_id,
        c.customer_tier,
        c.registration_date,
        COUNT(o.order_id) as total_orders,
        COALESCE(SUM(o.total_amount), 0) as total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id 
        AND o.order_status IN ('confirmed', 'shipped', 'delivered')
    GROUP BY c.customer_id, c.customer_tier, c.registration_date
    ORDER BY total_spent DESC
    LIMIT 10
    """
    
    df = pd.read_sql_query(simple_query, conn)
    conn.close()
    
    print("✅ Query successful!")
    print(f"📊 Top customers:")
    print(df[['customer_tier', 'total_orders', 'total_spent']].to_string())
    
    print(f"\n🎯 Summary:")
    print(f"   👥 Total customers analyzed: {len(df)}")
    print(f"   💰 Highest spender: €{df['total_spent'].max():,.2f}")
    print(f"   🛒 Most orders: {df['total_orders'].max()}")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🎉 Simple ML test complete!")