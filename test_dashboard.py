"""
Simple Test Dashboard
Minimal version to test basic functionality
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

st.set_page_config(page_title="Test Dashboard", layout="wide")

st.title("🧪 Test Analytics Dashboard")

# Test 1: Basic Streamlit
st.write("✅ Streamlit is working!")

# Test 2: Environment variables
connection_string = os.getenv("SUPABASE_DATABASE_URL")
if connection_string:
    st.write("✅ Environment variables loaded")
    st.write(f"Connection string found: {connection_string[:30]}...")
else:
    st.error("❌ No SUPABASE_DATABASE_URL found in environment")
    st.stop()

# Test 3: Database connection
try:
    st.write("🔄 Testing database connection...")
    conn = psycopg2.connect(connection_string, connect_timeout=10)
    st.write("✅ Database connection successful!")
    
    # Test 4: Simple query
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM customers")
    customer_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM orders WHERE order_status IN ('confirmed', 'shipped', 'delivered')")
    order_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(total_amount) FROM orders WHERE order_status IN ('confirmed', 'shipped', 'delivered')")
    total_revenue = cursor.fetchone()[0]
    
    conn.close()
    
    st.write("✅ Database queries successful!")
    
    # Test 5: Display basic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", customer_count)
    
    with col2:
        st.metric("Total Orders", order_count)
    
    with col3:
        st.metric("Total Revenue", f"€{total_revenue:,.2f}")
    
    # Test 6: Simple chart
    st.subheader("📊 Test Chart")
    
    # Get simple data for chart
    conn = psycopg2.connect(connection_string, connect_timeout=10)
    df = pd.read_sql_query("""
        SELECT 
            customer_tier,
            COUNT(*) as customer_count
        FROM customers 
        GROUP BY customer_tier
        ORDER BY customer_count DESC
    """, conn)
    conn.close()
    
    if not df.empty:
        fig = px.bar(df, x='customer_tier', y='customer_count', 
                    title='Customer Distribution by Tier')
        st.plotly_chart(fig, use_container_width=True)
        st.write("✅ Charts working!")
    else:
        st.warning("No data found for chart")
    
    st.success("🎉 All tests passed! Dashboard components are working.")
    
except psycopg2.Error as e:
    st.error(f"❌ Database error: {e}")
    
except Exception as e:
    st.error(f"❌ Unexpected error: {e}")
    st.write("Error details:", str(e))

# Test 7: Session state
if 'test_counter' not in st.session_state:
    st.session_state.test_counter = 0

if st.button("Test Session State"):
    st.session_state.test_counter += 1

st.write(f"Button clicked {st.session_state.test_counter} times")

st.write("---")
st.write("If you see this message, basic Streamlit functionality is working! 🚀")