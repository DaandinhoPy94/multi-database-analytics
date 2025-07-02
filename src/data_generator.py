"""
E-commerce Sample Data Generator - Fixed Version
Uses direct psycopg2 connection instead of SQLAlchemy for bulk inserts
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

class EcommerceDataGenerator:
    """
    Professional data generator for e-commerce analytics
    Uses direct psycopg2 for reliable cloud database connections
    """
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducible data"""
        self.fake = Faker(['nl_NL', 'en_US'])
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Business constants
        self.CUSTOMER_TIERS = ['Bronze', 'Silver', 'Gold', 'Platinum']
        self.TIER_WEIGHTS = [0.6, 0.25, 0.12, 0.03]
        
        self.ORDER_STATUSES = ['pending', 'confirmed', 'shipped', 'delivered', 'cancelled']
        self.STATUS_WEIGHTS = [0.05, 0.10, 0.15, 0.65, 0.05]
        
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
        """Connect to Supabase database using direct psycopg2"""
        connection_string = os.getenv("SUPABASE_DATABASE_URL")
        if not connection_string:
            raise ValueError("SUPABASE_DATABASE_URL not found in .env file")
        
        try:
            self.conn = psycopg2.connect(
                connection_string,
                connect_timeout=30,
                application_name="ecommerce_data_generator"
            )
            self.conn.autocommit = False  # Use transactions for safety
            print("✅ Connected to Supabase database via psycopg2")
        except Exception as e:
            raise Exception(f"Database connection failed: {e}")
        
    def execute_query(self, query, params=None):
        """Execute a query and return results"""
        with self.conn.cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description:  # SELECT query
                return cursor.fetchall()
            return cursor.rowcount  # INSERT/UPDATE/DELETE
    
    def bulk_insert(self, table_name, data_list, columns):
        """Bulk insert data using psycopg2.extras.execute_values"""
        if not data_list:
            return
            
        # Create placeholders for SQL
        placeholders = ','.join(['%s'] * len(columns))
        query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES %s"
        
        # Convert list of dicts to list of tuples
        values = []
        for row in data_list:
            values.append(tuple(row[col] for col in columns))
        
        with self.conn.cursor() as cursor:
            psycopg2.extras.execute_values(
                cursor, query, values,
                template=f"({placeholders})",
                page_size=100  # Insert 100 rows at a time
            )
        
        self.conn.commit()
        print(f"✅ Inserted {len(data_list)} rows into {table_name}")
        
    def generate_customers(self, num_customers=300):
        """Generate realistic customer data"""
        print(f"👥 Generating {num_customers} customers...")
        
        customers = []
        
        for i in range(num_customers):
            gender = random.choice(['Male', 'Female', 'Other'])
            first_name = self.fake.first_name_male() if gender == 'Male' else self.fake.first_name_female()
            last_name = self.fake.last_name()
            
            age = int(np.random.normal(40, 12))
            age = max(18, min(80, age))
            birth_date = datetime.now().date() - timedelta(days=age*365 + random.randint(0, 365))
            
            tier = np.random.choice(self.CUSTOMER_TIERS, p=self.TIER_WEIGHTS)
            
            reg_days_ago = int(np.random.exponential(200))
            reg_days_ago = min(reg_days_ago, 730)
            registration_date = datetime.now().date() - timedelta(days=reg_days_ago)
            
            customer = {
                'email': f"{first_name.lower()}.{last_name.lower()}{random.randint(1,999)}@{self.fake.free_email_domain()}",
                'first_name': first_name[:100],  # Limit to 100 chars
                'last_name': last_name[:100],   # Limit to 100 chars  
                'phone': self.fake.phone_number()[:20],  # Limit to 20 chars
                'date_of_birth': birth_date,
                'gender': gender[:20] if gender else None,  # Limit to 20 chars
                'registration_date': registration_date,
                'customer_tier': tier,
                'is_active': random.choice([True, True, True, False])
            }
            
            customers.append(customer)
        
        # Bulk insert using psycopg2
        columns = ['email', 'first_name', 'last_name', 'phone', 'date_of_birth', 
                  'gender', 'registration_date', 'customer_tier', 'is_active']
        self.bulk_insert('customers', customers, columns)
        
        return customers
    
    def generate_products(self):
        """Generate product catalog"""
        print("🛍️ Generating product catalog...")
        
        # Get category IDs
        categories_result = self.execute_query("SELECT category_id, category_name FROM categories")
        categories = {row[1]: row[0] for row in categories_result}
        
        products = []
        
        # Generate all products
        for product_template in (self.ELECTRONICS_PRODUCTS + self.CLOTHING_PRODUCTS + self.HOME_PRODUCTS):
            category_name = 'Electronics' if product_template in self.ELECTRONICS_PRODUCTS else \
                          'Clothing' if product_template in self.CLOTHING_PRODUCTS else 'Home & Garden'
            
            cost_variation = random.uniform(0.9, 1.1)
            price_variation = random.uniform(0.95, 1.05)
            
            product = {
                'product_name': product_template['name'],
                'sku': f"{product_template['brand'][:3].upper()}-{random.randint(10000, 99999)}",
                'category_id': categories[category_name],
                'brand': product_template['brand'],
                'description': f"High-quality {product_template['name']} from {product_template['brand']}",
                'unit_cost': round(product_template['cost'] * cost_variation, 2),
                'selling_price': round(product_template['price'] * price_variation, 2),
                'weight_kg': round(random.uniform(0.1, 5.0), 3),
                'dimensions_cm': f"{random.randint(10,50)}x{random.randint(10,40)}x{random.randint(5,20)}",
                'is_active': random.choice([True, True, True, False])
            }
            products.append(product)
        
        # Bulk insert products
        columns = ['product_name', 'sku', 'category_id', 'brand', 'description',
                  'unit_cost', 'selling_price', 'weight_kg', 'dimensions_cm', 'is_active']
        self.bulk_insert('products', products, columns)
        
        return products
    
    def generate_addresses(self, num_customers):
        """Generate customer addresses"""
        print(f"🏠 Generating addresses...")
        
        addresses = []
        
        for customer_id in range(1, num_customers + 1):
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
                    'is_primary': i == 0
                }
                addresses.append(address)
        
        columns = ['customer_id', 'address_type', 'street_address', 'city', 
                  'state_province', 'postal_code', 'country', 'is_primary']
        self.bulk_insert('customer_addresses', addresses, columns)
        
        return addresses
    
    def generate_orders_and_items(self, num_customers, num_orders=800):
        """Generate orders and order items"""
        print(f"🛒 Generating {num_orders} orders with items...")
        
        # Get data for order generation
        customers_result = self.execute_query("SELECT customer_id, customer_tier FROM customers")
        customer_tiers = {row[0]: row[1] for row in customers_result}
        
        products_result = self.execute_query("SELECT product_id, selling_price FROM products WHERE is_active = true")
        product_prices = {row[0]: row[1] for row in products_result}
        
        orders = []
        order_items = []
        
        tier_multipliers = {'Bronze': 1.0, 'Silver': 1.3, 'Gold': 1.6, 'Platinum': 2.0}
        
        for order_id in range(1, num_orders + 1):
            customer_id = random.randint(1, num_customers)
            tier = customer_tiers.get(customer_id, 'Bronze')
            
            days_ago = int(np.random.exponential(60))
            days_ago = min(days_ago, 365)
            order_date = datetime.now() - timedelta(days=days_ago)
            
            status = np.random.choice(self.ORDER_STATUSES, p=self.STATUS_WEIGHTS)
            
            base_items = np.random.poisson(2) + 1
            num_items = max(1, int(base_items * tier_multipliers[tier]))
            num_items = min(num_items, 8)
            
            selected_products = random.sample(list(product_prices.keys()), 
                                            min(num_items, len(product_prices)))
            
            subtotal = 0
            
            for product_id in selected_products:
                quantity = random.randint(1, 3)
                unit_price = float(product_prices[product_id])
                
                discount = 0
                if random.random() < 0.1:
                    discount = round(unit_price * random.uniform(0.05, 0.15), 2)
                
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
                order_items.append(order_item)
            
            # Calculate order totals
            tax_rate = 0.21
            tax_amount = round(subtotal * tax_rate, 2)
            
            shipping_cost = 0 if subtotal > 50 else 5.95
            
            order_discount = 0
            if tier == 'Gold' and random.random() < 0.2:
                order_discount = round(subtotal * 0.05, 2)
            elif tier == 'Platinum' and random.random() < 0.3:
                order_discount = round(subtotal * 0.10, 2)
            
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
        
        # Bulk insert orders
        order_columns = ['order_id', 'customer_id', 'order_date', 'order_status', 
                        'payment_method', 'shipping_method', 'subtotal', 'tax_amount',
                        'shipping_cost', 'discount_amount', 'total_amount', 'notes']
        self.bulk_insert('orders', orders, order_columns)
        
        # Bulk insert order items
        item_columns = ['order_id', 'product_id', 'quantity', 'unit_price', 
                       'line_total', 'discount_applied']
        self.bulk_insert('order_items', order_items, item_columns)
        
        return orders, order_items
    
    def generate_all_data(self, num_customers=300, num_orders=800):
        """Generate complete dataset"""
        print("🚀 Starting complete data generation...\n")
        
        try:
            # Connect to database
            self.connect_to_database()
            
            # Generate data in order
            customers = self.generate_customers(num_customers)
            products = self.generate_products()
            addresses = self.generate_addresses(num_customers)
            orders, order_items = self.generate_orders_and_items(num_customers, num_orders)
            
            print("\n🎉 Data generation completed successfully!")
            print(f"📊 Summary:")
            print(f"   👥 Customers: {len(customers)}")
            print(f"   🛍️  Products: {len(products)}")
            print(f"   🏠 Addresses: {len(addresses)}")
            print(f"   🛒 Orders: {len(orders)}")
            print(f"   📦 Order Items: {len(order_items)}")
            
        except Exception as e:
            print(f"❌ Error during data generation: {e}")
            if hasattr(self, 'conn'):
                self.conn.rollback()
            raise
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()
                print("✅ Database connection closed")

def main():
    """Main function to run data generation"""
    generator = EcommerceDataGenerator()
    
    # Generate smaller dataset for testing
    generator.generate_all_data(
        num_customers=200,  # Smaller for testing
        num_orders=500
    )
    
    print("\n✅ Ready for analytics queries!")

if __name__ == "__main__":
    main()