"""
Script to generate Phase 2 sample datasets:
1. sample_data/sample_sales.csv (Clean Dataset)
2. sample_data/sample_dirty_data.xlsx (Realistic Dirty Dataset)
"""
import os
import random
from datetime import datetime, timedelta
import pandas as pd

# Set fixed seed for reproducibility
random.seed(42)

# Ensure sample_data directory exists
os.makedirs("sample_data", exist_ok=True)

# -------------------------------------------------------------
# MASTER DATA DEFINITIONS (Indian context, INR prices)
# -------------------------------------------------------------
CUSTOMERS = [
    "Aarav Sharma", "Priya Patel", "Rohan Verma", "Ananya Iyer", "Vikram Malhotra",
    "Neha Gupta", "Rajesh Kumar", "Sneha Nair", "Aditya Joshi", "Pooja Singh",
    "Rahul Deshmukh", "Kavita Reddy", "Amit Shah", "Meera Sundaram", "Karthik Raman",
    "Divya Chawla", "Suresh Menon", "Ritu Sen", "Varun Kapoor", "Sunita Rao",
    "Manish Tiwari", "Deepika Padukone", "Siddharth Roy", "Ishaan Bhat", "Nandini Mukherjee"
]

PRODUCTS_CATALOG = [
    ("Noise Cancelling Headphones", "Electronics", 4999.0),
    ("Ergonomic Office Chair", "Furniture", 8500.0),
    ("Mechanical Keyboard", "Electronics", 3200.0),
    ("Ceramic Coffee Mug Set", "Home & Kitchen", 799.0),
    ("Stainless Steel Water Bottle", "Home & Kitchen", 549.0),
    ("Smart Fitness Band", "Electronics", 2499.0),
    ("Cotton Double Bedspread", "Home & Kitchen", 1899.0),
    ("Adjustable Laptop Stand", "Furniture", 1299.0),
    ("Wireless Gaming Mouse", "Electronics", 1699.0),
    ("Desk Bookshelf Organizer", "Furniture", 1499.0),
]

REGIONS = ["North", "South", "East", "West", "Central"]
PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"]

START_DATE = datetime(2024, 1, 1)

# -------------------------------------------------------------
# TASK 2: GENERATE CLEAN DATASET (sample_sales.csv)
# -------------------------------------------------------------
clean_rows = []
for i in range(1, 101):
    order_id = f"ORD-{1000 + i}"
    # Uniformly spaced dates over ~180 days
    order_date = (START_DATE + timedelta(days=int(i * 1.8))).strftime("%Y-%m-%d")
    customer = random.choice(CUSTOMERS)
    product, category, unit_price = random.choice(PRODUCTS_CATALOG)
    region = random.choice(REGIONS)
    quantity = random.randint(1, 5)
    revenue = round(quantity * unit_price, 2)
    payment = random.choice(PAYMENT_METHODS)

    clean_rows.append({
        "order_id": order_id,
        "order_date": order_date,
        "customer_name": customer,
        "product": product,
        "category": category,
        "region": region,
        "quantity": quantity,
        "unit_price": unit_price,
        "revenue": revenue,
        "payment_method": payment
    })

df_clean = pd.DataFrame(clean_rows)
df_clean.to_csv("sample_data/sample_sales.csv", index=False)
print(f"[OK] Created sample_data/sample_sales.csv with {len(df_clean)} rows.")

# -------------------------------------------------------------
# TASK 3: GENERATE DIRTY DATASET (sample_dirty_data.xlsx)
# -------------------------------------------------------------
dirty_rows = []
for i in range(1, 101):
    order_id = f"ORD-{2000 + i}"
    order_date = (START_DATE + timedelta(days=int(i * 1.8))).strftime("%Y-%m-%d")
    customer = random.choice(CUSTOMERS)
    product, category, unit_price = random.choice(PRODUCTS_CATALOG)
    region = random.choice(REGIONS)
    quantity = random.randint(1, 5)
    revenue = round(quantity * unit_price, 2)
    payment = random.choice(PAYMENT_METHODS)

    dirty_rows.append({
        "order_id": order_id,
        "order_date": order_date,
        "customer_name": customer,
        "product": product,
        "category": category,
        "region": region,
        "quantity": quantity,
        "unit_price": unit_price,
        "revenue": revenue,
        "payment_method": payment
    })

# Now intentionally introduce controlled realistic dirty anomalies into ~15-20 rows:

# 1. Missing values (None/NaN)
dirty_rows[7]["customer_name"] = None         # Row 8: Missing customer
dirty_rows[21]["category"] = None              # Row 22: Missing category
dirty_rows[44]["payment_method"] = None        # Row 45: Missing payment method
dirty_rows[67]["revenue"] = None               # Row 68: Missing revenue
dirty_rows[84]["region"] = None                # Row 85: Missing region

# 2. Duplicate rows
# Row 14 is an exact duplicate of Row 13
dirty_rows[13] = dict(dirty_rows[12])
# Row 39 has duplicate order_id ("ORD-2035")
dirty_rows[38]["order_id"] = dirty_rows[34]["order_id"]

# 3. Extra leading/trailing whitespace
dirty_rows[4]["customer_name"] = "  Rohan Verma "      # Row 5
dirty_rows[30]["product"] = "Mechanical Keyboard  "    # Row 31
dirty_rows[56]["region"] = " West "                    # Row 57

# 4. Inconsistent capitalization
dirty_rows[11]["category"] = "electronics"             # Row 12: lowercase
dirty_rows[37]["category"] = "FURNITURE"               # Row 38: uppercase
dirty_rows[61]["product"] = "smart fitness band"       # Row 62: lowercase

# 5. Inconsistent product / category names (typos / variations)
dirty_rows[17]["product"] = "Wireless Mouse"           # Row 18: missing 'Gaming'
dirty_rows[51]["category"] = "Home and Kitchen"        # Row 52: 'and' instead of '&'
dirty_rows[73]["product"] = "Laptop Stand (Adjustable)" # Row 74: inverted format

# 6. Inconsistent region naming
dirty_rows[24]["region"] = "South Region"              # Row 25
dirty_rows[48]["region"] = "S. India"                  # Row 49
dirty_rows[80]["region"] = "north"                     # Row 81: lowercase

# 7. Different date formats
dirty_rows[8]["order_date"] = "15/02/2024"             # Row 9: DD/MM/YYYY
dirty_rows[26]["order_date"] = "03-22-2024"            # Row 27: MM-DD-YYYY
dirty_rows[64]["order_date"] = "Apr 12, 2024"          # Row 65: Month Name

# 8. Invalid dates
dirty_rows[34]["order_date"] = "2024-02-31"            # Row 35: Feb 31 does not exist
dirty_rows[76]["order_date"] = "2024-13-45"            # Row 77: Month 13, Day 45
dirty_rows[91]["order_date"] = "invalid_date"          # Row 92: Text string

# 9. Numeric values stored as text / strings with symbols
dirty_rows[15]["unit_price"] = "4999"                  # Row 16: string number
dirty_rows[41]["unit_price"] = "INR 8500"              # Row 42: string with currency prefix

# 10. Negative quantity
dirty_rows[32]["quantity"] = -2                        # Row 33: negative qty

# 11. Zero quantity
dirty_rows[58]["quantity"] = 0                         # Row 59: zero qty

# 12. Outlier / Unrealistic revenue values
dirty_rows[49]["revenue"] = 9999999.0                  # Row 50: extreme outlier (data entry error)
dirty_rows[87]["revenue"] = -1500.0                    # Row 88: negative revenue

# 13. Inconsistent payment method spelling
dirty_rows[9]["payment_method"] = "credit_card"        # Row 10
dirty_rows[27]["payment_method"] = "CC"                # Row 28
dirty_rows[46]["payment_method"] = "U.P.I."            # Row 47
dirty_rows[70]["payment_method"] = "Cod"               # Row 71
dirty_rows[94]["payment_method"] = "NetBanking"        # Row 95

# 14. Mismatched revenue calculation
dirty_rows[59]["quantity"] = 3
dirty_rows[59]["unit_price"] = 1299.0
dirty_rows[59]["revenue"] = 1299.0                     # Row 60: revenue doesn't equal qty * price

df_dirty = pd.DataFrame(dirty_rows)
df_dirty.to_excel("sample_data/sample_dirty_data.xlsx", index=False)
print(f"[OK] Created sample_data/sample_dirty_data.xlsx with {len(df_dirty)} rows.")
