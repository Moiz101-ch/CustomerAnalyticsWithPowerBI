# Customer Segmentation and Dashboarding  

This repository showcases a project that segments customers based on their purchasing behavior, using clustering techniques and data visualization. The goal is to identify target groups and provide actionable insights through an interactive Power BI dashboard.  

---

## Features  
- **Data Cleaning & Preparation**: Handle missing values, standardize features, and prepare data for analysis.  
- **Exploratory Data Analysis (EDA)**: Visualize key features such as spending scores and demographic distributions to uncover patterns.  
- **Clustering**: Apply K-Means clustering to group customers based on age and spending behavior.  
- **Dashboard Creation**: Build an interactive Power BI dashboard with visuals like bar charts, pie charts, and slicers for filtering insights.  
- **Actionable Insights**: Provide strategic recommendations for targeting customer segments effectively.  

---

## Workflow  

### 1. Data Collection  
The dataset used in this project is sourced from Kaggle: [Customer Segmentation Dataset](https://www.kaggle.com/code/fabiendaniel/customer-segmentation/input).  
It contains customer demographic and purchasing information, including:  
- InvoiceNo  
- StockCode  
- Description  
- Quantity  
- InvoiceDate
- UnitPrice  
- CustomerID
- Country
  
### 2. Data Exploration & Cleaning  
- Analyzed the dataset for trends and anomalies.  
- Handled missing values and standardized features to improve clustering results.  

### 3. Clustering with K-Means  
- Grouped customers into distinct segments based on age and spending score.  
- Evaluated clustering performance through visualizations and metrics.  

### 4. Data Visualization with Power BI  
- Exported processed data to CSV for use in Power BI.  
- Created an interactive dashboard with the following features:  
  - Bar charts for customer segment distribution.  
  - Pie charts for gender breakdown across segments.  
  - Slicers for filtering data by demographics.  

### 5. Insights & Recommendations  
- Interpreted the clusters and summarized customer segment characteristics.  
- Provided business strategies for targeting and engaging each group effectively.  

---

## Installation & Usage  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Moiz101-ch/CustomerAnalyticsWithPowerBI.git
   cd CustomerAnalyticsWithPowerBI
   ```
2. **Install Required Libraries**
   
   Install the Python libraries listed in `requirements.txt`.
     
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Scripts**
   
   Use the python script provided to perform data cleaning, clustering, and exporting data.
   
6. **Power BI Dashboard**

   Import the exported CSV file into Power BI to explore the interactive dashboard.

---

## Tools & Technologies

- **Python**: Data cleaning, clustering, and preprocessing.
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn.
- **Power BI**: Dashboard creation and data visualization.

---

## Dataset Source

The dataset used in this project is sourced from Kaggle:

[Customer Segmentation Dataset](https://www.kaggle.com/code/fabiendaniel/customer-segmentation/input)
  
