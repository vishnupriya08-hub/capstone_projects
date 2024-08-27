 **DataSpark: Illuminating Insights for Global Electronics**
**Overview**
This project involves a comprehensive data analysis of Global Electronics. 
It integrates multiple datasets, preprocesses and cleans the data, performs exploratory data analysis (EDA), and stores the data in a MySQL database. Additionally, several analytical tables are created to provide insights into sales, customer demographics, product performance, and store operations.

**Features**
Data Loading from CSV files
Data Preprocessing and Cleaning
Merging Multiple Datasets
Identifying and Handling Outliers
Data Normalization
Exploratory Data Analysis (EDA)
Storing Data in a MySQL Database
Creating Analytical Tables for Insights

**Datasets**
**The following datasets are used in this project:**

Products.csv: Information about products including unit cost and price.
Sales.csv: Sales data including order and delivery dates.
Stores.csv: Details about store locations and sizes.
Customers.csv: Customer demographics including location and birthdates.
Exchange_Rates.csv: Historical exchange rates data.
Prerequisites
Python 3.x
MySQL
Required Python libraries: pandas, seaborn, matplotlib, numpy, scipy, mysql-connector-python

**Usage**
**Load and Preprocess Data:**

The load_data function loads data from the CSV files.
The preprocess_data function handles missing values, converts date columns, cleans unit cost and price columns, and converts key columns to integer types.
Merge Data:

The merge_data function merges sales, products, stores, customers, and exchange rates data into a single DataFrame.

**Outlier Detection and Handling:**

The find_outliers function visualizes outliers using boxplots.
The standardize_and_reduce_outliers function standardizes data and reduces outliers.
Normalization:

The normalize_data function normalizes unit price using min-max normalization and applies log transformation.

**Exploratory Data Analysis (EDA):**

The distribution_analysis function performs kurtosis analysis and plots histograms.
The frequency_plot function visualizes the frequency of product names, brands, and categories.
The correlation_analysis function calculates and visualizes correlations using a heatmap.

**Database Integration:**

The create_database function connects to MySQL, creates tables, and inserts data into the database.
The create_analytics_tables function creates analytical tables for insights into sales, customer demographics, and store performance.

**Run the main function:**

The main function manages the entire process from data loading to creating analytical tables.
**python spark.py**


