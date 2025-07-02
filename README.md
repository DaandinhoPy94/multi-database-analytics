# Multi-Database Analytics Platform

## 🎯 Project Overview
A comprehensive e-commerce analytics platform showcasing advanced SQL, database design, and business intelligence capabilities. Built as part of an advanced developer learning pathway from beginner to professional full-stack AI developer.

## 🚀 What's Built

### **Phase 1: Database Architecture & Data Generation** ✅ COMPLETED
- **PostgreSQL Schema**: Enterprise-level e-commerce database with 9 interconnected tables
- **Business Logic**: Constraints, triggers, and validation rules
- **Realistic Data Generator**: Professional Python script generating 200+ customers, 500+ orders, 1600+ order items
- **Cloud Integration**: Supabase PostgreSQL with session pooler for reliable connections
- **Data Quality**: Business-realistic patterns with customer tiers, seasonal trends, and pricing logic

### **Technical Stack**
- **Database**: PostgreSQL (Supabase cloud)
- **Backend**: Python 3.11+ with SQLAlchemy, psycopg2
- **Data Generation**: Faker library with statistical distributions
- **Development**: Virtual environments, Git workflows, professional project structure

## 📊 Database Schema

### **Core Tables**
- **customers**: Demographics, tiers (Bronze/Silver/Gold/Platinum), registration patterns
- **products**: Multi-category catalog (Electronics, Clothing, Home & Garden) with realistic pricing
- **orders**: Complete transaction data with tax, shipping, discounts
- **order_items**: Line-item details with business validation
- **customer_addresses**: Multi-address support per customer
- **categories**: Product categorization hierarchy

### **Business Intelligence Features**
- Customer Lifetime Value (CLV) calculations
- Product performance analytics
- Revenue trend analysis
- Customer segmentation
- Geographic distribution
- Payment method analysis

## 🛠️ Setup Instructions

### **Prerequisites**
- Python 3.11+
- Git
- Supabase account (free tier sufficient)

### **Installation**
```bash
# Clone repository
git clone git@github.com:YOUR-USERNAME/multi-database-analytics.git
cd multi-database-analytics

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Supabase credentials
```

### **Database Setup**
1. Create Supabase project at supabase.com
2. Copy connection string (use Session Pooler for reliability)
3. Execute schema: `sql/schemas/01_supabase_schema.sql` in Supabase SQL Editor
4. Generate sample data: `python src/data_generator.py`

## 📈 Sample Analytics Queries

### **Customer Lifetime Value Analysis**
```sql
SELECT 
    customer_tier,
    COUNT(*) as customers,
    AVG(total_spent) as avg_clv,
    MAX(total_spent) as max_clv
FROM customers c
LEFT JOIN customer_order_totals cot ON c.customer_id = cot.customer_id
GROUP BY customer_tier
ORDER BY avg_clv DESC;
```

### **Product Performance Metrics**
```sql
SELECT 
    p.product_name,
    p.brand,
    SUM(oi.quantity) as units_sold,
    SUM(oi.line_total) as revenue,
    ROUND((AVG(oi.unit_price) - p.unit_cost) / AVG(oi.unit_price) * 100, 1) as margin_pct
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.brand, p.unit_cost
ORDER BY revenue DESC;
```

## 🎓 Learning Outcomes

### **Database Design Mastery**
- ✅ Normalized schema design with proper relationships
- ✅ Business rule implementation via constraints
- ✅ Index optimization for analytical queries
- ✅ Cloud database deployment and management

### **Professional Python Development**
- ✅ Object-oriented programming with business logic
- ✅ Statistical data generation with realistic distributions
- ✅ Database connectivity and bulk operations
- ✅ Error handling and transaction management

### **Business Intelligence Skills**
- ✅ Customer segmentation and CLV analysis
- ✅ Product performance and profitability metrics
- ✅ Revenue trend analysis and forecasting foundations
- ✅ Data-driven decision making frameworks

## 🗂️ Project Structure
```
multi-database-analytics/
├── src/
│   ├── data_generator.py          # Realistic e-commerce data generation
│   └── database_connection.py     # Multi-database connection manager
├── sql/
│   ├── schemas/
│   │   └── 01_supabase_schema.sql # Complete PostgreSQL schema
│   └── queries/
│       └── analytics_queries.sql  # Business intelligence queries
├── data/                          # Generated datasets (gitignored)
├── docs/                          # Documentation and learning notes
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
└── README.md
```

## 🚀 Next Phase: Advanced Analytics & Dashboards

### **Coming Soon**
- **Advanced SQL**: Window functions, CTEs, recursive queries
- **Performance Optimization**: Query tuning, indexing strategies
- **Interactive Dashboards**: Streamlit/Dash visualization platform
- **ETL Pipelines**: Automated data processing workflows
- **Machine Learning Integration**: Predictive analytics and recommendations

## 🔗 Related Projects
Part of the **Advanced Developer Learning Roadmap**:
1. ✅ **Personal Finance ML Dashboard** - Streamlit app with ML categorization
2. ✅ **Multi-Database Analytics Platform** - This project
3. 🔄 **iOS SwiftUI Task Manager** - Native mobile development
4. 🔄 **Blockchain Certification Platform** - Web3 development
5. 🔄 **Production ML Recommendation Engine** - Enterprise AI systems

## 📝 License
MIT License - Feel free to use this project for learning and portfolio purposes.

---

*Built with ❤️ as part of an intensive full-stack AI developer learning journey*