"""
Advanced Dashboard Components
Interactive filters, date pickers, and dynamic visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class DashboardFilters:
    """Advanced filtering components for dashboard"""
    
    @staticmethod
    def date_range_selector(revenue_data):
        """Interactive date range selector"""
        if not revenue_data.empty:
            revenue_data['month'] = pd.to_datetime(revenue_data['month'])
            
            min_date = revenue_data['month'].min()
            max_date = revenue_data['month'].max()
            
            st.sidebar.subheader("📅 Date Range Filter")
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_data = revenue_data[
                    (revenue_data['month'].dt.date >= start_date) & 
                    (revenue_data['month'].dt.date <= end_date)
                ]
                return filtered_data
            
        return revenue_data
    
    @staticmethod
    def customer_tier_filter(customers_data):
        """Customer tier multi-select filter"""
        if not customers_data.empty:
            st.sidebar.subheader("👥 Customer Tier Filter")
            
            available_tiers = customers_data['customer_tier'].unique()
            selected_tiers = st.sidebar.multiselect(
                "Select Customer Tiers",
                options=available_tiers,
                default=available_tiers
            )
            
            if selected_tiers:
                filtered_data = customers_data[customers_data['customer_tier'].isin(selected_tiers)]
                return filtered_data
            
        return customers_data
    
    @staticmethod
    def product_category_filter(products_data):
        """Product category filter"""
        if not products_data.empty:
            st.sidebar.subheader("🛍️ Product Category Filter")
            
            available_categories = products_data['category_name'].unique()
            selected_categories = st.sidebar.multiselect(
                "Select Product Categories", 
                options=available_categories,
                default=available_categories
            )
            
            if selected_categories:
                filtered_data = products_data[products_data['category_name'].isin(selected_categories)]
                return filtered_data
                
        return products_data
    
    @staticmethod
    def top_n_selector(label, default=20, max_value=100):
        """Generic top N selector"""
        return st.sidebar.slider(
            f"Top {label} to Display",
            min_value=5,
            max_value=max_value, 
            value=default,
            step=5
        )

class AdvancedCharts:
    """Advanced chart components with interactivity"""
    
    @staticmethod
    def revenue_forecast_chart(revenue_data):
        """Revenue forecast with trend lines"""
        if len(revenue_data) < 3:
            st.warning("Need at least 3 months of data for forecasting")
            return
            
        df = revenue_data.copy()
        df['month'] = pd.to_datetime(df['month'])
        
        # Simple linear trend for demonstration
        from scipy import stats
        x_numeric = (df['month'] - df['month'].min()).dt.days
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, df['monthly_revenue'])
        
        # Forecast next 3 months
        last_date = df['month'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=30), periods=3, freq='M')
        future_x = (future_dates - df['month'].min()).total_seconds() / (24*3600)
        future_revenue = slope * future_x + intercept
        
        # Create forecast chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df['month'], 
            y=df['monthly_revenue'],
            mode='lines+markers',
            name='Actual Revenue',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Trend line
        trend_y = slope * x_numeric + intercept
        fig.add_trace(go.Scatter(
            x=df['month'],
            y=trend_y,
            mode='lines',
            name='Trend Line',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_revenue,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#2ca02c', dash='dot', width=2),
            marker=dict(symbol='diamond', size=8)
        ))
        
        fig.update_layout(
            title=f'Revenue Forecast (R² = {r_value**2:.3f})',
            xaxis_title='Month',
            yaxis_title='Revenue (€)',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def customer_cohort_heatmap(db_connection):
        """Customer cohort retention heatmap"""
        cohort_query = """
        WITH customer_cohorts AS (
            SELECT 
                customer_id,
                DATE_TRUNC('month', MIN(order_date)) as cohort_month
            FROM orders
            WHERE order_status IN ('confirmed', 'shipped', 'delivered')
            GROUP BY customer_id
        ),
        customer_monthly_activity AS (
            SELECT 
                cc.cohort_month,
                DATE_TRUNC('month', o.order_date) as activity_month,
                EXTRACT(YEAR FROM AGE(DATE_TRUNC('month', o.order_date), cc.cohort_month)) * 12 + 
                EXTRACT(MONTH FROM AGE(DATE_TRUNC('month', o.order_date), cc.cohort_month)) as period_number,
                COUNT(DISTINCT cc.customer_id) as active_customers
            FROM customer_cohorts cc
            JOIN orders o ON cc.customer_id = o.customer_id
            WHERE o.order_status IN ('confirmed', 'shipped', 'delivered')
            GROUP BY cc.cohort_month, DATE_TRUNC('month', o.order_date)
        )
        SELECT 
            cohort_month,
            period_number,
            active_customers,
            ROUND(
                active_customers * 100.0 / 
                FIRST_VALUE(active_customers) OVER (
                    PARTITION BY cohort_month 
                    ORDER BY period_number 
                    ROWS UNBOUNDED PRECEDING
                ), 1
            ) as retention_rate
        FROM customer_monthly_activity
        ORDER BY cohort_month, period_number;
        """
        
        cohort_data = db_connection.execute_query(cohort_query)
        
        if not cohort_data.empty:
            # Pivot for heatmap
            heatmap_data = cohort_data.pivot(
                index='cohort_month', 
                columns='period_number', 
                values='retention_rate'
            )
            
            fig = px.imshow(
                heatmap_data,
                title='Customer Cohort Retention Rates (%)',
                labels=dict(x="Period (Months)", y="Cohort Month", color="Retention %"),
                aspect="auto",
                color_continuous_scale="RdYlGn"
            )
            
            return fig
        
        return None
    
    @staticmethod
    def product_bubble_chart(products_data):
        """Interactive product bubble chart"""
        if products_data.empty:
            return None
            
        fig = px.scatter(
            products_data.head(30),
            x='total_units_sold',
            y='profit_per_unit',
            size='total_revenue',
            color='category_name',
            hover_name='product_name',
            hover_data={
                'brand': True,
                'total_revenue': ':,.2f',
                'total_profit': ':,.2f'
            },
            title='Product Performance Matrix',
            labels={
                'total_units_sold': 'Units Sold',
                'profit_per_unit': 'Profit per Unit (€)',
                'total_revenue': 'Total Revenue (€)'
            }
        )
        
        # Add quadrant lines
        median_units = products_data['total_units_sold'].median()
        median_profit = products_data['profit_per_unit'].median()
        
        fig.add_hline(y=median_profit, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=median_units, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant annotations (CORRECTED)
        fig.add_annotation(x=median_units * 1.5, y=median_profit * 1.5, 
                        text="High Volume<br>High Profit", showarrow=False)
        fig.add_annotation(x=median_units * 0.5, y=median_profit * 1.5, 
                        text="Low Volume<br>High Profit", showarrow=False)
        fig.add_annotation(x=median_units * 1.5, y=median_profit * 0.5, 
                        text="High Volume<br>Low Profit", showarrow=False)
        fig.add_annotation(x=median_units * 0.5, y=median_profit * 0.5, 
                        text="Low Volume<br>Low Profit", showarrow=False)

        return fig