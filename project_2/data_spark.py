import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import mysql.connector

def load_data():
    # Extract the data from csv
    products = pd.read_csv("Products.csv")
    sales = pd.read_csv("Sales.csv")
    stores = pd.read_csv("Stores.csv")
    customers = pd.read_csv("customers.csv", encoding='unicode_escape')
    exchange_rates = pd.read_csv("Exchange_Rates.csv")
    return products, sales, stores, customers, exchange_rates

def preprocess_data(products, sales, stores, customers, exchange_rates):
    # Handling missing values
    sales['Delivery Date'] = sales['Delivery Date'].fillna(sales['Delivery Date'].mode()[0])
    customers['State Code'] = customers['State Code'].fillna(customers['State Code'].mode()[0])

    # Converting date columns to datetime format
    sales['Order Date'] = pd.to_datetime(sales['Order Date']).dt.date
    sales['Delivery Date'] = pd.to_datetime(sales['Delivery Date']).dt.date
    exchange_rates['Date'] = pd.to_datetime(exchange_rates['Date']).dt.date
    customers['Birthday'] = pd.to_datetime(customers['Birthday']).dt.date
    
    # Cleaning and converting 'Unit Cost USD' and 'Unit Price USD'
    products = products.dropna(subset=['Unit Cost USD'])
    products['Unit Cost USD'] = products['Unit Cost USD'].str.replace('$', '').str.replace(',', '').str.strip().astype(float)
    products = products.dropna(subset=['Unit Price USD'])
    products['Unit Price USD'] = products['Unit Price USD'].str.replace('$', '').str.replace(',', '').str.strip().astype(float)
    
    # Converting key columns to integer type
    products['ProductKey'] = products['ProductKey'].astype(int)
    products['SubcategoryKey'] = products['SubcategoryKey'].astype(int)
    products['CategoryKey'] = products['CategoryKey'].astype(int)
    customers['Zip Code'] = pd.to_numeric(customers['Zip Code'], errors='coerce')
    customers = customers.dropna(subset=['Zip Code'])
    customers['Zip Code'] = customers['Zip Code'].astype(int)
    customers['CustomerKey'] = customers['CustomerKey'].astype(int)
   
    # Cleaning 'Open Date' in stores
    stores['Open Date'] = pd.to_datetime(stores['Open Date']).dt.date
    stores = stores.dropna()

    return products, sales, stores, customers, exchange_rates

def merge_data(products, sales, stores, customers, exchange_rates):
    # Merge sales and products data on the 'ProductKey' column
    sales_products = pd.merge(sales, products, on='ProductKey', how='left')
    # Merge the result with stores data on the 'StoreKey' column
    sales_products_stores = pd.merge(sales_products, stores, on='StoreKey', how='left')
     # Merge the result with customers data on the 'CustomerKey' column
    full_data = pd.merge(sales_products_stores, customers, on='CustomerKey', how='left')
    # Merge the result with exchange rates data on the 'Order Date' column
    full_data = pd.merge(full_data, exchange_rates, left_on='Order Date', right_on='Date', how='left')
    # Return the merged DataFrame
    return full_data

def find_outliers(data):
    sns.boxplot(data['Quantity'])
    plt.show()

    sns.boxplot(data['Unit Price USD'])
    plt.show()

    sns.boxplot(data['Unit Cost USD'])
    plt.show()

def standardize_and_reduce_outliers(data):
    # Standardize 'Unit Cost USD' and reduce outliers
    data['Unit Cost USD'] = (data['Unit Cost USD'] - data['Unit Cost USD'].mean()) / data['Unit Cost USD'].std()
    data = data[(data['Unit Cost USD'] > -3) & (data['Unit Cost USD'] < 3)]
    sns.boxplot(data['Unit Cost USD'])
    plt.show()
    
    # Standardize 'Unit Price USD' and reduce outliers
    data['Unit Price USD'] = (data['Unit Price USD'] - data['Unit Price USD'].mean()) / data['Unit Price USD'].std()
    data = data[(data['Unit Price USD'] > -3) & (data['Unit Price USD'] < 3)]
    sns.boxplot(data['Unit Price USD'])
    plt.show()
    
    # Standardize 'Quantity' and reduce outliers
    data['Quantity'] = (data['Quantity'] - data['Quantity'].mean()) / data['Quantity'].std()
    data = data[(data['Quantity'] > -3) & (data['Quantity'] < 3)]
    sns.boxplot(data['Quantity'])
    plt.show()
    
    return data

def normalize_data(data):
    # Normalize 'Unit Price USD' using min-max normalization
    data['Unit Price USD'] = (data['Unit Price USD'] - data['Unit Price USD'].min()) / (data['Unit Price USD'].max() - data['Unit Price USD'].min())
    # Calculate and print skewness of the normalized data
    print("Skew:", data['Unit Price USD'].skew())
    # Apply log transformation to the normalized data
    skew = np.log(data['Unit Price USD'])
    # Plot the histogram of the log-transformed data
    sns.histplot(skew)
    plt.show()

    return data

def get_kurtosis(value):
    if value > 3:
        return "Leptokurtic"
    elif value < 3:
        return "Platykurtic"
    else:
        return "Mesokurtic"

def distribution_analysis(data):
    value = data['Unit Cost USD'].kurtosis()
    name = get_kurtosis(value)
    sns.histplot(data['Unit Cost USD'])
    plt.show()
    print(name, value)

def frequency_plot(data):
    data['Product Name'].value_counts()[0:30].plot(kind="bar")
    plt.show()

    data['Brand'].value_counts()[0:40].plot(kind="bar")
    plt.show()

    data['Category'].value_counts()[0:40].plot(kind="bar")
    plt.show()

def correlation_analysis(data):
    corr = data[['Unit Cost USD', 'Unit Price USD', 'Quantity', 'Square Meters']].corr()
    sns.heatmap(corr, annot=True)
    plt.show()

# Connect to the MySQL database
def create_database():
    con = mysql.connector.connect(
        host='localhost',
        user='root',
        password='12345678',
        database="company"
    )
    cursor = con.cursor()
     
    # Create the table products if it doesn't already exist
    query = """CREATE TABLE IF NOT EXISTS products(
                    ProductKey INT PRIMARY KEY,
                    ProductName VARCHAR(255),
                    Brand VARCHAR(255),
                    Color VARCHAR(255),
                    UnitCostUSD FLOAT,
                    UnitPriceUSD FLOAT,
                    SubcategoryKey INT,
                    Subcategory VARCHAR(255),
                    CategoryKey INT,
                    Category VARCHAR(255)
                )"""
    cursor.execute(query)
    
    # products table insert into values
    query = "INSERT INTO products VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    for index, row in products.iterrows():
        val = (
            int(row['ProductKey']),
            str(row['Product Name']),
            str(row['Brand']),
            str(row['Color']),
            float(row['Unit Cost USD']),
            float(row['Unit Price USD']),
            int(row['SubcategoryKey']),
            str(row['Subcategory']),
            int(row['CategoryKey']),
            str(row['Category'])
        )
        cursor.execute(query, val)
    
    # Create the table sales if it doesn't already exist
    query = """CREATE TABLE IF NOT EXISTS sales(
                    OrderNumber INT,
                    LineItem INT,
                    OrderDate DATE,
                    DeliveryDate DATE,
                    CustomerKey INT,
                    StoreKey INT,
                    ProductKey INT,
                    Quantity INT,
                    CurrencyCode VARCHAR(255)
                )"""
    cursor.execute(query)

    # sales table insert into values
    query = "INSERT INTO sales VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    for index, row in sales.iterrows():
        val = (
            int(row['Order Number']),
            int(row['Line Item']),
            row['Order Date'],
            row['Delivery Date'],
            int(row['CustomerKey']),
            int(row['StoreKey']),
            int(row['ProductKey']),
            int(row['Quantity']),
            str(row['Currency Code'])
        )
        cursor.execute(query, val)
    
    # Create the table stores if it doesn't already exist
    query = """CREATE TABLE IF NOT EXISTS stores(
                    StoreKey INT PRIMARY KEY,
                    Country VARCHAR(255),
                    State VARCHAR(255),
                    SquareMeters FLOAT,
                    OpenDate DATE
                )"""
    cursor.execute(query)
    
    # stores table insert into values
    query = "INSERT INTO stores VALUES (%s, %s, %s, %s, %s)"
    for index, row in stores.iterrows():
        val = (
            int(row['StoreKey']),
            str(row['Country']),
            str(row['State']),
            float(row['Square Meters']),
            row['Open Date']
        )
        cursor.execute(query, val)
    
    # Create the table customers if it doesn't already exist
    query = """CREATE TABLE IF NOT EXISTS customers(
                    CustomerKey INT,
                    Gender VARCHAR(255),
                    Name VARCHAR(255),
                    City VARCHAR(255),
                    StateCode VARCHAR(255),
                    State VARCHAR(255),
                    ZipCode INT,
                    Country VARCHAR(255),
                    Continent VARCHAR(255),
                    Birthday DATE
                )"""
    cursor.execute(query)
    
    # customers table insert into values
    query = "INSERT INTO customers VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    for index, row in customers.iterrows():
        val = (
            int(row['CustomerKey']),
            str(row['Gender']),
            str(row['Name']),
            str(row['City']),
            str(row['State Code']),
            str(row['State']),
            int(row['Zip Code']),
            str(row['Country']),
            str(row['Continent']),
            row['Birthday']
        )
        cursor.execute(query, val)
     
    # Create the table exchannge_rates if it doesn't already exist
    query = """CREATE TABLE IF NOT EXISTS exchange_rates(
                    Date DATE,
                    Currency VARCHAR(255),
                    Exchange FLOAT
                )"""
    cursor.execute(query)

    # exchange_rates table insert into values
    query = "INSERT INTO exchange_rates VALUES (%s, %s, %s)"
    for index, row in exchange_rates.iterrows():
        val = (
            row['Date'],
            str(row['Currency']),
            float(row['Exchange'])
        )
        cursor.execute(query, val)

    con.commit()
    return con, cursor
    
# Used exist table query for create analytics table
def create_analytics_tables(cursor):
    # This query analysis the total sales percentage
    query = """
            CREATE TABLE IF NOT EXISTS sales_percentage AS
            WITH SalesTotals AS (
            SELECT st.StoreKey, st.State, SUM(s.Quantity) AS TotalSales
            FROM sales s
            JOIN stores st ON s.StoreKey = st.StoreKey
            GROUP BY st.StoreKey, st.State
        ),
        GrandTotal AS (
            SELECT SUM(TotalSales) AS GrandTotalSales
            FROM SalesTotals
        )
        SELECT st.StoreKey, st.State, st.TotalSales, 
               (st.TotalSales * 100.0 / gt.GrandTotalSales) AS PercentageOfTotalSales
        FROM SalesTotals st, GrandTotal gt
        ORDER BY st.TotalSales DESC;
    """
    cursor.execute(query)

    # This query analysis the gender count
    query = """
    CREATE TABLE IF NOT EXISTS customer_gender AS
    SELECT Gender, COUNT(*) AS NumberOfCustomers
    FROM customers
    GROUP BY Gender;
    """
    cursor.execute(query)

    # This query analysis the currency based sales
    query = """
    CREATE TABLE IF NOT EXISTS currency_sales AS
    SELECT s.CurrencyCode, SUM(s.Quantity) AS TotalSales
    FROM sales s
    GROUP BY s.CurrencyCode
    ORDER BY TotalSales DESC;
    """
    cursor.execute(query)

    # This query analysis the Total quantity sold based on product name
    query = """
    CREATE TABLE IF NOT EXISTS total_quantity_sold AS
    SELECT p.ProductName, SUM(s.Quantity) AS TotalQuantitySold
    FROM sales s
    JOIN products p ON s.ProductKey = p.ProductKey
    GROUP BY p.ProductName
    ORDER BY TotalQuantitySold DESC;
    """
    cursor.execute(query)

   # This query analysis the total sales based on category
    query = """
    CREATE TABLE IF NOT EXISTS category_total_sales AS
    SELECT p.Category, SUM(s.Quantity) AS TotalSales
    FROM sales s
    JOIN products p ON s.ProductKey = p.ProductKey
    GROUP BY p.Category
    ORDER BY TotalSales DESC;
    """
    cursor.execute(query)

    # This query analysis the total profit based on total quantity unit price and unit cost
    query = """
    CREATE TABLE IF NOT EXISTS total_profit AS
    SELECT p.ProductName, SUM(s.Quantity * (p.UnitPriceUSD - p.UnitCostUSD)) AS TotalProfit
    FROM sales s
    JOIN products p ON s.ProductKey = p.ProductKey
    GROUP BY p.ProductName
    ORDER BY TotalProfit DESC;
    """
    cursor.execute(query)

    # This query analysis the total sales based on country and state
    query = """
    CREATE TABLE IF NOT EXISTS country_state_total_sales AS
    SELECT st.Country, st.State, SUM(s.Quantity) AS TotalSales
    FROM sales s
    JOIN stores st ON s.StoreKey = st.StoreKey
    GROUP BY st.Country, st.State
    ORDER BY TotalSales DESC;
    """
    cursor.execute(query)

    # This query analysis Total Sales and Average Sales per Square Meter for Each Store
    query="""
    CREATE TABLE if not exists sales_per_sqm
    SELECT s.StoreKey,s.Country,s.State,s.SquareMeters,
    COALESCE(SUM(sl.Quantity), 0) AS total_sales,
    (COALESCE(SUM(sl.Quantity), 0) / s.SquareMeters) AS sales_per_sqm
    FROM stores s
    LEFT JOIN sales sl ON s.StoreKey = sl.StoreKey
    GROUP BY s.StoreKey, s.Country, s.State, s.SquareMeters;
    """
    cursor.execute(query)

    # This query to analysis Identify High-Performing Regions Based on Total and Average Sales
    query = """
    CREATE TABLE if not exists store_location_sales
    SELECT s.Country,s.State,
    COALESCE(SUM(sl.total_sales), 0) AS total_sales,
    COALESCE(AVG(sl.total_sales), 0) AS average_sales_per_store,
    COUNT(s.StoreKey) AS number_of_stores
    FROM stores s
    LEFT JOIN (SELECT StoreKey, SUM(Quantity) AS total_sales FROM sales GROUP BY StoreKey) sl ON s.StoreKey = sl.StoreKey
    GROUP BY s.Country, s.State
    ORDER BY total_sales DESC, average_sales_per_store DESC;
    """
    cursor.execute(query)

    # Query to analysis the age distribution
    query1="""
    SELECT TIMESTAMPDIFF(YEAR, Birthday, CURDATE()) AS age,
    COUNT(*) AS number_of_customers
    FROM customers
    GROUP BY age
    ORDER BY age;
    """
    # Query to analyze the age range distribution
    query="""
    CREATE TABLE if not exists age_group
    SELECT 
        CASE
            WHEN TIMESTAMPDIFF(YEAR, Birthday, CURDATE()) BETWEEN 0 AND 10 THEN '0-10'
            WHEN TIMESTAMPDIFF(YEAR, Birthday, CURDATE()) BETWEEN 11 AND 20 THEN '11-20'
            WHEN TIMESTAMPDIFF(YEAR, Birthday, CURDATE()) BETWEEN 21 AND 30 THEN '21-30'
            WHEN TIMESTAMPDIFF(YEAR, Birthday, CURDATE()) BETWEEN 31 AND 40 THEN '31-40'
            WHEN TIMESTAMPDIFF(YEAR, Birthday, CURDATE()) BETWEEN 41 AND 50 THEN '41-50'
            WHEN TIMESTAMPDIFF(YEAR, Birthday, CURDATE()) BETWEEN 51 AND 60 THEN '51-60'
            ELSE '61+'
        END AS age_range,
        COUNT(*) AS number_of_customers
    FROM customers
    GROUP BY age_range
    ORDER BY age_range;
    """   
    cursor.execute(query)
        
# The main function return all the values
def main():
    products, sales, stores, customers, exchange_rates = load_data()
    products, sales, stores, customers, exchange_rates = preprocess_data(products, sales, stores, customers, exchange_rates)
    full_data = merge_data(products, sales, stores, customers, exchange_rates)
    find_outliers(full_data)
    full_data = standardize_and_reduce_outliers(full_data)
    full_data = normalize_data(full_data)
    distribution_analysis(full_data)
    frequency_plot(full_data)
    correlation_analysis(full_data)

    con, cursor = create_database()
    create_analytics_tables(cursor)


if __name__ == "__main__":
    main()
