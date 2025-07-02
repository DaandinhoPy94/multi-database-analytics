"""
E-commerce Sample Data Generator
Creates realistic business data for analytics testing
Professional approach: Faker for realistic data, business logic constraints
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

class EcommerceDataGenerator:
    """
    Professional data generator for e-commerce analytics
    Creates realistic, interconnected business data
    """
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducible data"""
        self.fake = Faker(['nl_NL', 'en_US'])  # Dutch and English locales
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Business constants
        self.CUSTOMER_TIERS = ['Bronze', 'Silver', 'Gold', 'Platinum']
        self.TIER_WEIGHTS = [0.6, 0.25, 0.12, 0.03]  # Most customers are Bronze
        
        self.ORDER_STATUSES = ['pending', 'confirmed', 'shipped', 'delivered', 'cancelled']
        self.STATUS_WEIGHTS = [0.05, 0.10, 0.15, 0.65, 0.05]  # Most are delivered
        
        # Product data templates
        self.ELECTRONICS_PRODUCTS = [
            {"name": "iPhone 15 Pro", "brand": "Apple", "cost": 800, "price": 1199},
            {"name": "Samsung Galaxy S24", "brand": "Samsung", "cost": 600, "price": 899},
            {"name": "MacBook Air M3", "brand": "Apple", "cost": 900, "price": 1399},
            {"name": "Dell XPS 13", "brand": "Dell", "cost": 700, "price": 1099},
            {"name": "Sony WH-1000XM5", "brand": "Sony", "cost": 200, "price": 349},
            {"name": "iPad Pro 12.9", "brand": "Apple", "cost": 650, "price": 1099},
            {"name": "AirPods Pro", "brand": "Apple", "cost": 150, "price": 279},
            {"name": "Nintendo Switch", "brand": "Nintendo", "cost": 200, "price": 299}
        ]
        
        self.CLOTHING_PRODUCTS = [
            {"name": "Levi's 501 Jeans", "brand": "Levi's", "cost": 45, "price": 89},
            {"name": "Nike Air Max 90", "brand": "Nike", "cost": 65, "price": 129},
            {"name": "Adidas Ultraboost", "brand": "Adidas", "cost": 80, "price": 179},
            {"name": "Zara Wool Coat", "brand": "Zara", "cost": 75, "price": 149},
            {"name": "H&M Basic T-Shirt", "brand": "H&M", "cost": 8, "price": 19},
            {"name": "Tommy Hilfiger Polo", "brand": "Tommy Hilfiger", "cost": 35, "price": 79},
            {"name": "G-Star Raw Jeans", "brand": "G-Star", "cost": 55, "price": 119},
            {"name": "Converse Chuck Taylor", "brand": "Converse", "cost": 40, "price": 69}
        ]
        
        self.HOME_PRODUCTS = [
            {"name": "IKEA LACK Table", "brand": "IKEA", "cost": 15, "price": 29},
            {"name": "Philips Hue Bulbs", "brand": "Philips", "cost": 25, "price": 49},
            {"name": "Dyson V15 Vacuum", "brand": "Dyson", "cost": 350, "price": 599},
            {"name": "Weber BBQ Grill", "brand": "Weber", "cost": 180, "price": 349},
            {"name": "Nespresso Coffee Machine", "brand": "Nespresso", "cost": 120, "price": 199},
            {"name": "Tefal Cookware Set", "brand": "Tefal", "cost": 85, "price": 159},
            {"name": "LEGO Architecture Set", "brand": "LEGO", "cost": 60, "price": 99},
            {"name": "Sonos One Speaker", "brand": "Sonos", "cost": 150, "price": 229}
        ]
        
    def connect_to_database(self):
        """Connect to Supabase database with timeout settings"""
        connection_string = os.getenv("SUPABASE_DATABASE_URL")
        if not connection_string:
            raise ValueError("SUPABASE_DATABASE_URL not found in .env file")
        
        # Add connection parameters for better reliability
        self.engine = create_engine(
            connection_string,
            pool_timeout=30,
            pool_recycle=300,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 30,
                "application_name": "ecommerce_data_generator"
            }
        )
        print("✅ Connected to Supabase database")
        
    def generate_customers(self, num_customers=500):
        """Generate realistic customer data"""
        print(f"👥 Generating {num_customers} customers...")
        
        customers = []
        
        for i in range(num_customers):
            # Generate realistic demographics
            gender = random.choice(['Male', 'Female', 'Other'])
            first_name = self.fake.first_name_male() if gender == 'Male' else self.fake.first_name_female()
            last_name = self.fake.last_name()
            
            # Age distribution: mostly 25-55 years old
            age = int(np.random.normal(40, 12))
            age = max(18, min(80, age))  # Clamp between 18-80
            birth_date = datetime.now().date() - timedelta(days=age*365 + random.randint(0, 365))
            
            # Customer tier based on realistic distribution
            tier = np.random.choice(self.CUSTOMER_TIERS, p=self.TIER_WEIGHTS)
            
            # Registration date: mostly in last 2 years
            reg_days_ago = int(np.random.exponential(200))  # Exponential distribution
            reg_days_ago = min(reg_days_ago, 730)  # Max 2 years ago
            registration_date = datetime.now().date() - timedelta(days=reg_days_ago)
            
            customer = {
                'email': f"{first_name.lower()}.{last_name.lower()}{random.randint(1,999)}@{self.fake.free_email_domain()}",
                'first_name': first_name,
                'last_name': last_name,
                'phone': self.fake.phone_number(),
                'date_of_birth': birth_date,
                'gender': gender,
                'registration_date': registration_date,
                'customer_tier': tier,
                'is_active': random.choice([True, True, True, False])  # 75% active
            }
            
            customers.append(customer)
        
        # Convert to DataFrame and save to database
        df_customers = pd.DataFrame(customers)
        df_customers.to_sql('customers', self.engine, if_exists='append', index=False)
        print(f"✅ Generated {len(customers)} customers")
        return df_customers
    
    def generate_products(self):
        """Generate product catalog with realistic pricing"""
        print("🛍️ Generating product catalog...")
        
        products = []
        
        # Get category IDs from database
        with self.engine.connect() as conn:
            categories_result = conn.execute(text("SELECT category_id, category_name FROM categories"))
            categories = {row[1]: row[0] for row in categories_result.fetchall()}
        
        # Generate Electronics
        for product_template in self.ELECTRONICS_PRODUCTS:
            product = self._create_product(product_template, categories['Electronics'])
            products.append(product)
        
        # Generate Clothing
        for product_template in self.CLOTHING_PRODUCTS:
            product = self._create_product(product_template, categories['Clothing'])
            products.append(product)
        
        # Generate Home & Garden
        for product_template in self.HOME_PRODUCTS:
            product = self._create_product(product_template, categories['Home & Garden'])
            products.append(product)
        
        # Convert to DataFrame and save
        df_products = pd.DataFrame(products)
        df_products.to_sql('products', self.engine, if_exists='append', index=False)
        print(f"✅ Generated {len(products)} products")
        return df_products
    
    def _create_product(self, template, category_id):
        """Create individual product with variations"""
        # Add some price variation (±10%)
        cost_variation = random.uniform(0.9, 1.1)
        price_variation = random.uniform(0.95, 1.05)
        
        return {
            'product_name': template['name'],
            'sku': f"{template['brand'][:3].upper()}-{random.randint(10000, 99999)}",
            'category_id': category_id,
            'brand': template['brand'],
            'description': f"High-quality {template['name']} from {template['brand']}",
            'unit_cost': round(template['cost'] * cost_variation, 2),
            'selling_price': round(template['price'] * price_variation, 2),
            'weight_kg': round(random.uniform(0.1, 5.0), 3),
            'dimensions_cm': f"{random.randint(10,50)}x{random.randint(10,40)}x{random.randint(5,20)}",
            'is_active': random.choice([True, True, True, False])  # 75% active
        }
    
    def generate_addresses(self, customer_ids):
        """Generate customer addresses"""
        print(f"🏠 Generating addresses for {len(customer_ids)} customers...")
        
        addresses = []
        
        for customer_id in customer_ids:
            # 60% have 1 address, 35% have 2, 5% have 3+
            num_addresses = np.random.choice([1, 2, 3], p=[0.6, 0.35, 0.05])
            
            for i in range(num_addresses):
                address = {
                    'customer_id': customer_id,
                    'address_type': 'shipping' if i == 0 else random.choice(['shipping', 'billing']),
                    'street_address': self.fake.street_address(),
                    'city': self.fake.city(),
                    'state_province': self.fake.state(),
                    'postal_code': self.fake.postcode(),
                    'country': 'Netherlands',
                    'is_primary': i == 0  # First address is primary
                }
                addresses.append(address)
        
        df_addresses = pd.DataFrame(addresses)
        df_addresses.to_sql('customer_addresses', self.engine, if_exists='append', index=False)
        print(f"✅ Generated {len(addresses)} addresses")
        return df_addresses
    
    def generate_orders_and_items(self, customer_ids, product_ids, num_orders=1000):
        """Generate realistic orders with business logic"""
        print(f"🛒 Generating {num_orders} orders with items...")
        
        orders = []
        order_items = []
        
        # Get customer tiers for spending behavior
        with self.engine.connect() as conn:
            customers_result = conn.execute(text("SELECT customer_id, customer_tier FROM customers"))
            customer_tiers = {row[0]: row[1] for row in customers_result.fetchall()}
            
            products_result = conn.execute(text("SELECT product_id, selling_price FROM products WHERE is_active = true"))
            product_prices = {row[0]: row[1] for row in products_result.fetchall()}
        
        for order_id in range(1, num_orders + 1):
            # Select random customer
            customer_id = random.choice(customer_ids)
            tier = customer_tiers.get(customer_id, 'Bronze')
            
            # Order date: mostly in last 6 months, some seasonal patterns
            days_ago = int(np.random.exponential(60))  # Average 2 months ago
            days_ago = min(days_ago, 365)  # Max 1 year ago
            order_date = datetime.now() - timedelta(days=days_ago)
            
            # Order status
            status = np.random.choice(self.ORDER_STATUSES, p=self.STATUS_WEIGHTS)
            
            # Number of items based on customer tier
            tier_multipliers = {'Bronze': 1.0, 'Silver': 1.3, 'Gold': 1.6, 'Platinum': 2.0}
            base_items = np.random.poisson(2) + 1  # Average 2-3 items
            num_items = max(1, int(base_items * tier_multipliers[tier]))
            num_items = min(num_items, 8)  # Max 8 items per order
            
            # Select products for this order
            selected_products = random.sample(list(product_prices.keys()), 
                                            min(num_items, len(product_prices)))
            
            subtotal = 0
            order_items_for_this_order = []
            
            for product_id in selected_products:
                quantity = random.randint(1, 3)
                unit_price = product_prices[product_id]
                
                # Apply small random discount occasionally
                if random.random() < 0.1:  # 10% chance of discount
                    discount = round(unit_price * random.uniform(0.05, 0.15), 2)
                else:
                    discount = 0
                
                line_total = (quantity * unit_price) - discount
                subtotal += line_total
                
                order_item = {
                    'order_id': order_id,
                    'product_id': product_id,
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'line_total': line_total,
                    'discount_applied': discount
                }
                order_items_for_this_order.append(order_item)
            
            # Calculate order totals
            tax_rate = 0.21  # Dutch VAT
            tax_amount = round(subtotal * tax_rate, 2)
            
            # Shipping cost based on order size
            if subtotal > 50:
                shipping_cost = 0  # Free shipping over €50
            else:
                shipping_cost = 5.95
            
            # Order-level discount for high-tier customers
            order_discount = 0
            if tier == 'Gold' and random.random() < 0.2:
                order_discount = round(subtotal * 0.05, 2)  # 5% discount
            elif tier == 'Platinum' and random.random() < 0.3:
                order_discount = round(subtotal * 0.10, 2)  # 10% discount
            
            total_amount = subtotal + tax_amount + shipping_cost - order_discount
            
            order = {
                'order_id': order_id,
                'customer_id': customer_id,
                'order_date': order_date,
                'order_status': status,
                'payment_method': random.choice(['Credit Card', 'PayPal', 'Bank Transfer', 'iDEAL']),
                'shipping_method': random.choice(['Standard', 'Express', 'Next Day']),
                'subtotal': subtotal,
                'tax_amount': tax_amount,
                'shipping_cost': shipping_cost,
                'discount_amount': order_discount,
                'total_amount': total_amount,
                'notes': self.fake.sentence() if random.random() < 0.1 else None
            }
            
            orders.append(order)
            order_items.extend(order_items_for_this_order)
        
        # Save to database
        df_orders = pd.DataFrame(orders)
        df_order_items = pd.DataFrame(order_items)
        
        df_orders.to_sql('orders', self.engine, if_exists='append', index=False)
        df_order_items.to_sql('order_items', self.engine, if_exists='append', index=False)
        
        print(f"✅ Generated {len(orders)} orders with {len(order_items)} items")
        return df_orders, df_order_items
    
    def generate_all_data(self, num_customers=500, num_orders=1000):
        """Generate complete dataset"""
        print("🚀 Starting complete data generation...\n")
        
        # Connect to database
        self.connect_to_database()
        
        # Generate in order (respecting foreign key constraints)
        df_customers = self.generate_customers(num_customers)
        customer_ids = list(range(1, num_customers + 1))
        
        df_products = self.generate_products()
        product_ids = list(range(1, len(df_products) + 1))
        
        df_addresses = self.generate_addresses(customer_ids)
        
        df_orders, df_order_items = self.generate_orders_and_items(
            customer_ids, product_ids, num_orders
        )
        
        print("\n🎉 Data generation completed successfully!")
        print(f"📊 Summary:")
        print(f"   👥 Customers: {len(df_customers)}")
        print(f"   🛍️  Products: {len(df_products)}")
        print(f"   🏠 Addresses: {len(df_addresses)}")
        print(f"   🛒 Orders: {len(df_orders)}")
        print(f"   📦 Order Items: {len(df_order_items)}")
        
        return {
            'customers': df_customers,
            'products': df_products,
            'addresses': df_addresses,
            'orders': df_orders,
            'order_items': df_order_items
        }

def main():
    """Main function to run data generation"""
    generator = EcommerceDataGenerator()
    
    # Generate data
    data = generator.generate_all_data(
        num_customers=300,  # Start smaller for testing
        num_orders=800
    )
    
    print("\n✅ Ready for analytics queries!")

if __name__ == "__main__":
    main()