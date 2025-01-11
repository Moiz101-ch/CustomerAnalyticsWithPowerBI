# Data-Driven Sales Analysis and Revenue Forecasting  

Welcome to the **Data-Driven Sales Analysis and Revenue Forecasting** project repository. This project combines data transformation, visualization, and predictive modeling to analyze sales trends and forecast future revenue. Leveraging tools like Power BI for dashboards and Python for advanced analytics, this repository demonstrates end-to-end data analysis and forecasting workflows.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Data Overview](#data-overview)
- [Power BI Dashboard](#power-bi-dashboard)
  - [Features](#features)
- [Revenue Forecasting](#revenue-forecasting)
  - [ARIMA Model](#arima-model)
  - [Prophet Model](#prophet-model)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This repository focuses on analyzing and forecasting sales data for a retail business. The project consists of:
1. **Data Visualization**: A Power BI dashboard to visualize revenue, quantity, and country-level sales trends.
2. **Revenue Forecasting**: Predictive modeling using ARIMA and Prophet to forecast future revenue and evaluate model performance.

The goal is to provide actionable insights into sales performance while demonstrating proficiency in data analysis, visualization, and machine learning techniques.

---

## Data Overview

The dataset contains retail sales data with the following attributes:
- **InvoiceNo**: Invoice number (object)
- **StockCode**: Product code (object)
- **Description**: Product description (object)
- **Quantity**: Quantity of items purchased (integer)
- **InvoiceDate**: Date of purchase (datetime)
- **UnitPrice**: Price per unit (float)
- **CustomerID**: Customer identifier (float)
- **Country**: Customer's country (object)
- **Revenue**: Calculated revenue for each transaction (float)

---

## Power BI Dashboard

### Features
The **Data-Driven Sales Analysis Dashboard** includes:
1. **Sum of Revenue by Month**: Visualizing revenue trends over time.
2. **Sum of Quantity by Month**: Analyzing product sales volume trends.
3. **Sum of Quantity and Revenue by Country**: Understanding geographic distribution of sales.
   
---

## Revenue Forecasting

### ARIMA Model
The ARIMA (AutoRegressive Integrated Moving Average) model was used to forecast revenue trends. 
- Model Parameters: `(4, 0, 4)` and `(4, 0, 2)` were tested.
- Performance Metric: **Root Mean Square Error (RMSE)** of the ARIMA model was calculated to evaluate accuracy.

#### Key Observations:
- ARIMA models performed well with stationary data after log transformations and scaling.
- Limitations: High variance in sales data slightly impacted accuracy.

### Prophet Model
The **Prophet** model was employed for robust and flexible revenue forecasting. 
- Advantages: Handles seasonality, trends, and missing data effectively.
- Performance Metric: **RMSE** for the Prophet model was calculated to evaluate accuracy.
- Key Feature: Prophet provides interpretability for seasonality and trends.

---

## How to Run

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sales-analysis-forecasting.git
   cd sales-analysis-forecasting
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Python scripts for ARIMA and Prophet models:
   ```
   python arima_forecasting.py
   python prophet_forecasting.py
   ```
4. Open the Power BI dashboard:
- Locate the file: Data-Driven Sales Analysis Dashboard.pbix.
- Open it in Power BI Desktop.
  
5. View the visualizations and evaluate the forecasting results.

## Repository Structure

```
├── data/                         # Dataset files
├── visuals/                      # Power BI dashboard and images
│   └── Data-Driven Sales Analysis Dashboard.pbix
├── scripts/                      # Python scripts for modeling
│   ├── arima_forecasting.py      # ARIMA model implementation
│   ├── prophet_forecasting.py    # Prophet model implementation
│   ├── k_means_clustering.py     # K means implementation
├── results/                      # Output results and model performance
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

---

## Future Enhancements
**Dashboard Improvements:** Add more granular visualizations (e.g., revenue by product category).
**Model Optimization:** Experiment with additional forecasting models like LSTM or XGBoost.
**Real-Time Analysis:** Incorporate real-time data streams for dynamic updates.

---

## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request. For significant changes, open an issue first to discuss your proposal.

---

## Dataset Source

The dataset used in this project is sourced from:

[Customer Segmentation Dataset](https://www.kaggle.com/code/fabiendaniel/customer-segmentation/input)
  
