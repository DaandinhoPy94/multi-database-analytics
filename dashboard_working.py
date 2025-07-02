"""
Working Analytics Dashboard
Simplified version based on successful test
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="Analytics Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_database_connection():
    """Get database connection string"""
    return os.getenv("SUPABASE_DATABASE_URL")

@st.cache_data
def load_key_metrics():
    """Load key business metrics"""
    conn_string = get_database_connection()
    conn = psycopg2.connect(conn_string, connect_timeout=30)
    
    cursor = conn.cursor()
    
    # Key metrics
    cursor.execute("SELECT COUNT(*) FROM customers")
    total_customers = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM orders WHERE order_status IN ('confirmed', 'shipped', 'delivered')")
    total_orders = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM products WHERE is_active = true")
    active_products = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(total_amount) FROM orders WHERE order_status IN ('confirmed', 'shipped', 'delivered')")
    total_revenue = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(total_amount) FROM orders WHERE order_status IN ('confirmed', 'shipped', 'delivered')")
    avg_order_value = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT customer_id) FROM orders WHERE order_status IN ('confirmed', 'shipped', 'delivered')")
    active_customers = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_customers': total_customers,
        'total_orders': total_orders,
        'active_products': active_products,
        'total_revenue': total_revenue,
        'avg_order_value': avg_order_value,
        'active_customers': active_customers
    }

@st.cache_data
def load_customer_data():
    """Load customer analytics data"""
    conn_string = get_database_connection()
    conn = psycopg2.connect(conn_string, connect_timeout=30)
    
    query = """
    WITH customer_spending AS (
        SELECT 
            c.customer_id,
            c.first_name,
            c.last_name,
            c.customer_tier,
            COUNT(o.order_id) as total_orders,
            COALESCE(SUM(o.total_amount), 0) as total_spent
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id 
            AND o.order_status IN ('confirmed', 'shipped', 'delivered')
        GROUP BY c.customer_id, c.first_name, c.last_name, c.customer_tier
    )
    SELECT 
        customer_id,
        first_name,
        last_name,
        customer_tier,
        total_orders,
        ROUND(total_spent, 2) as total_spent,
        ROW_NUMBER() OVER (ORDER BY total_spent DESC) as overall_rank,
        RANK() OVER (PARTITION BY customer_tier ORDER BY total_spent DESC) as rank_in_tier,
        ROUND(total_spent * 100.0 / SUM(total_spent) OVER (), 2) as pct_of_total_revenue
    FROM customer_spending
    WHERE total_spent > 0
    ORDER BY total_spent DESC;
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data 
def load_revenue_data():
    """Load revenue trends data"""
    conn_string = get_database_connection()
    conn = psycopg2.connect(conn_string, connect_timeout=30)
    
    query = """
    WITH monthly_revenue AS (
        SELECT 
            DATE_TRUNC('month', order_date) as month,
            COUNT(*) as orders_count,
            SUM(total_amount) as monthly_revenue
        FROM orders
        WHERE order_status IN ('confirmed', 'shipped', 'delivered')
        GROUP BY DATE_TRUNC('month', order_date)
    )
    SELECT 
        month,
        orders_count,
        ROUND(monthly_revenue, 2) as monthly_revenue,
        SUM(monthly_revenue) OVER (ORDER BY month) as cumulative_revenue,
        LAG(monthly_revenue) OVER (ORDER BY month) as prev_month_revenue,
        ROUND(
            (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY month)) * 100.0 / 
            NULLIF(LAG(monthly_revenue) OVER (ORDER BY month), 0), 
            1
        ) as mom_growth_pct
    FROM monthly_revenue
    ORDER BY month;
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data
def load_product_data():
    """Load product performance data"""
    conn_string = get_database_connection()
    conn = psycopg2.connect(conn_string, connect_timeout=30)
    
    query = """
    SELECT 
        p.product_name,
        p.brand,
        cat.category_name,
        COUNT(oi.order_item_id) as times_ordered,
        SUM(oi.quantity) as total_units_sold,
        ROUND(SUM(oi.line_total), 2) as total_revenue,
        ROUND(AVG(oi.unit_price), 2) as avg_selling_price,
        ROUND((AVG(oi.unit_price) - p.unit_cost), 2) as profit_per_unit,
        ROUND((AVG(oi.unit_price) - p.unit_cost) * SUM(oi.quantity), 2) as total_profit,
        RANK() OVER (ORDER BY SUM(oi.line_total) DESC) as revenue_rank,
        ROUND(SUM(oi.line_total) * 100.0 / SUM(SUM(oi.line_total)) OVER (), 2) as pct_of_total_revenue
    FROM products p
    JOIN categories cat ON p.category_id = cat.category_id
    JOIN order_items oi ON p.product_id = oi.product_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_status IN ('confirmed', 'shipped', 'delivered')
    GROUP BY p.product_id, p.product_name, p.brand, cat.category_name, p.unit_cost
    ORDER BY total_revenue DESC;
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<div class="main-header">📊 Executive Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading analytics data..."):
        metrics = load_key_metrics()
        customers_df = load_customer_data()
        revenue_df = load_revenue_data()
        products_df = load_product_data()
    
    # Executive Summary
    st.subheader("📈 Executive Summary")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Customers", f"{metrics['total_customers']:,}")
    
    with col2:
        st.metric("Total Orders", f"{metrics['total_orders']:,}")
    
    with col3:
        st.metric("Active Products", f"{metrics['active_products']:,}")
    
    with col4:
        st.metric("Total Revenue", f"€{metrics['total_revenue']:,.2f}")
    
    with col5:
        st.metric("Avg Order Value", f"€{metrics['avg_order_value']:,.2f}")
    
    with col6:
        st.metric("Active Customers", f"{metrics['active_customers']:,}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Revenue Analysis", "👥 Customer Analytics", "🛍️ Product Performance"])
    
    with tab1:
        st.header("📈 Revenue Trends & Growth Analysis")
        
        if not revenue_df.empty:
            revenue_df['month'] = pd.to_datetime(revenue_df['month'])
            
            # Revenue trend
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Revenue', 'Cumulative Revenue', 'Month-over-Month Growth %', 'Orders Count'),
                specs=[[{}, {}], [{}, {}]]
            )
            
            # Monthly revenue
            fig.add_trace(
                go.Scatter(x=revenue_df['month'], y=revenue_df['monthly_revenue'], 
                          mode='lines+markers', name='Monthly Revenue', line=dict(color='#1f77b4', width=3)),
                row=1, col=1
            )
            
            # Cumulative revenue
            fig.add_trace(
                go.Scatter(x=revenue_df['month'], y=revenue_df['cumulative_revenue'], 
                          mode='lines+markers', name='Cumulative Revenue', line=dict(color='#ff7f0e', width=3)),
                row=1, col=2
            )
            
            # Growth rate
            fig.add_trace(
                go.Bar(x=revenue_df['month'], y=revenue_df['mom_growth_pct'], 
                       name='MoM Growth %', marker_color='#2ca02c'),
                row=2, col=1
            )
            
            # Orders count
            fig.add_trace(
                go.Bar(x=revenue_df['month'], y=revenue_df['orders_count'], 
                       name='Orders Count', marker_color='#d62728'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True, title_text="Revenue Analysis Dashboard")
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            latest = revenue_df.iloc[-1]
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**📊 Key Insights:**")
            st.markdown(f"• Latest month revenue: **€{latest['monthly_revenue']:,.2f}**")
            st.markdown(f"• Month-over-month growth: **{latest['mom_growth_pct']:.1f}%**")
            st.markdown(f"• Total business revenue: **€{latest['cumulative_revenue']:,.2f}**")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("👥 Customer Analytics & Rankings")
        
        if not customers_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Top customers
                top_customers = customers_df.head(20)
                
                fig = px.bar(
                    top_customers, 
                    x='total_spent', 
                    y=top_customers['first_name'] + ' ' + top_customers['last_name'],
                    color='customer_tier',
                    title='Top 20 Customers by Revenue',
                    labels={'total_spent': 'Total Spent (€)', 'y': 'Customer'},
                    color_discrete_map={
                        'Platinum': '#FFD700',
                        'Gold': '#FFA500', 
                        'Silver': '#C0C0C0',
                        'Bronze': '#CD7F32'
                    }
                )
                fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Tier summary
                tier_summary = customers_df.groupby('customer_tier').agg({
                    'customer_id': 'count',
                    'total_spent': 'sum',
                    'total_orders': 'sum'
                }).round(2)
                
                st.subheader("Customer Tier Performance")
                st.dataframe(tier_summary, use_container_width=True)
                
                # Pie chart
                fig_pie = px.pie(
                    values=tier_summary['total_spent'], 
                    names=tier_summary.index,
                    title='Revenue by Customer Tier',
                    color_discrete_map={
                        'Platinum': '#FFD700',
                        'Gold': '#FFA500',
                        'Silver': '#C0C0C0', 
                        'Bronze': '#CD7F32'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Customer insights
            top_customer = customers_df.iloc[0]
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**🎯 Customer Insights:**")
            st.markdown(f"• Top customer: **{top_customer['first_name']} {top_customer['last_name']}** (€{top_customer['total_spent']:,.2f})")
            st.markdown(f"• Top 10 customers: **{customers_df.head(10)['pct_of_total_revenue'].sum():.1f}%** of total revenue")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.header("🛍️ Product Performance & Profitability")
        
        if not products_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Top products
                top_products = products_df.head(15)
                
                fig = px.bar(
                    top_products,
                    x='total_revenue',
                    y='product_name',
                    color='category_name',
                    title='Top 15 Products by Revenue',
                    labels={'total_revenue': 'Total Revenue (€)', 'product_name': 'Product'}
                )
                fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Profitability scatter
                fig_profit = px.scatter(
                    products_df.head(20),
                    x='total_units_sold',
                    y='profit_per_unit', 
                    size='total_revenue',
                    color='category_name',
                    hover_name='product_name',
                    title='Product Profitability Matrix',
                    labels={'total_units_sold': 'Units Sold', 'profit_per_unit': 'Profit per Unit (€)'}
                )
                st.plotly_chart(fig_profit, use_container_width=True)
            
            # Category performance
            category_summary = products_df.groupby('category_name').agg({
                'total_revenue': 'sum',
                'total_units_sold': 'sum',
                'total_profit': 'sum'
            }).round(2)
            
            st.subheader("Category Performance Summary")
            st.dataframe(category_summary, use_container_width=True)
            
            # Product insights
            top_product = products_df.iloc[0]
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**💡 Product Insights:**")
            st.markdown(f"• Top product: **{top_product['product_name']}** (€{top_product['total_revenue']:,.2f})")
            apple_revenue = products_df[products_df['brand'] == 'Apple']['pct_of_total_revenue'].sum()
            st.markdown(f"• Apple products: **{apple_revenue:.1f}%** of total revenue")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with ❤️ using advanced SQL analytics and Streamlit*")

if __name__ == "__main__":
    main()