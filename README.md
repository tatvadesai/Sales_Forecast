Prit Enterprise Sales & Forecasting Tool
Overview
The Prit Enterprise Sales & Forecasting Tool is an interactive Streamlit-based web application designed for analyzing historical sales data, generating sales forecasts, and providing inventory planning suggestions. Built with the Prophet forecasting library, it enables users to upload sales data in Excel format, visualize trends, and plan inventory for up to 36 months ahead. The tool supports multi-city analysis, seasonality decomposition, and customizable forecasting parameters.
Features

Data Upload & Processing: Upload Excel files with sales data organized by fiscal year sheets (e.g., "2020-2021") containing monthly sales per city and agency.
Interactive Visualizations: View sales trends, city/agency performance, and year-over-year comparisons using Plotly charts.
Prophet Forecasting: Generate sales forecasts for selected cities with trend and seasonality decomposition.
Inventory Planning: Calculate suggested stock levels and sales targets based on historical data, forecasts, and user-defined risk levels.
Customizable Filters: Filter data by cities, fiscal years, and months for tailored analysis.
Export Options: Download filtered data and detailed forecast outputs as CSV files.

Requirements
To run the application locally, ensure you have Python installed along with the following dependencies:
streamlit
pandas
numpy
prophet
plotly
openpyxl

Install them using pip:
pip install -r requirements.txt

Sample requirements.txt
streamlit>=1.0.0
pandas>=1.5.0
numpy>=1.22.0
prophet>=1.1.0
plotly>=5.0.0
openpyxl>=3.0.0

Installation

Clone or download the repository containing the code.

Navigate to the project directory:
cd path/to/project


Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required dependencies:
pip install -r requirements.txt


Run the Streamlit app:
streamlit run app.py

Replace app.py with the name of your Python script containing the provided code.

Access the app in your browser at http://localhost:8501.


Usage
1. Preparing Your Data

File Format: Excel (.xlsx) with sheets named in the format YYYY-YYYY (e.g., 2020-2021 for fiscal year 2020-2021).
Required Columns:
AGENCY NAME: Name of the agency.
CITY: City where sales occurred.
Monthly columns: April, May, June, July, August, September, October, November, December, January, February, March (in this order, representing fiscal year months starting from April).
Sales values should be numeric (e.g., number of boxes sold).


Example Data Structure:AGENCY NAME,CITY,April,May,June,...,March
Agency A,New York,100,120,130,...,110
Agency B,Chicago,80,90,100,...,85


Save the Excel file with one sheet per fiscal year.

2. Running the App

Upload Data: Use the sidebar to upload your Excel file under "Upload Your Data".
Configure Settings:
Forecast Horizon: Select the number of months to forecast (1–36) using the "Months to Forecast Ahead" slider.
Inventory Risk Level: Choose Low (90% Service Level), Medium (95% Service Level), or High (99% Service Level) to adjust safety stock calculations.
Inventory Planning: Specify the target year and months for inventory suggestions.
Display Filters: Select cities, fiscal years, and months to filter the displayed data.
Forecast Cities: Choose cities for Prophet forecasting.


Explore Tabs:
Main Dashboard: View overall sales metrics, filtered data summaries, and aggregated forecasts.
Advanced Insights & Planning: Analyze forecast decomposition, inventory suggestions, and actionable insights for a selected city.
Forecast Calculations: Compare Prophet forecasts with historical benchmarks and view detailed forecast outputs.
Dynamic Visualizations: Explore interactive charts for monthly trends, city/agency sales, and year-over-year comparisons.



3. Key Functionalities

Data Loading: The load_data function processes Excel sheets, validates column names, and transforms data into a time-series format with columns ds (date), city, agency, y (sales), etc.
Forecasting: The run_forecast_for_cities function uses Prophet to generate forecasts for cities with at least 12 months of data, ensuring robust seasonality modeling.
Inventory Suggestions: The generate_inventory_suggestions function calculates suggested stock and sales targets using historical medians, standard deviations, and Prophet forecasts, adjusted by the selected risk level (Z-score).
Visualizations: Plotly charts provide interactive views of sales trends, forecasts, and distributions, with options to filter by city, year, and month.
Caching: Streamlit’s @st.cache_data decorator optimizes performance for data loading, forecasting, and calculations.

4. Example Workflow

Upload an Excel file with sales data for multiple fiscal years.
Set the forecast horizon to 12 months and select Medium (95% Service Level) for inventory planning.
Choose target months (e.g., April, May, June) for inventory planning in the next year.
Filter data to focus on specific cities (e.g., New York, Chicago) and fiscal years (e.g., 2020-2021, 2021-2022).
Select cities for forecasting (e.g., New York, Chicago).
Navigate to the Advanced Insights & Planning tab to view forecast decomposition and inventory suggestions for a specific city.
Download the detailed Prophet forecast CSV from the Forecast Calculations tab for further analysis.

5. Output Examples

Main Dashboard:
Total sales: 10,000 boxes
Number of cities: 5
Bar chart of yearly sales and forecast for the selected period.


Inventory Suggestions (for New York, April 2026):
Suggested Stock: 150 boxes
Sales Target: 140 boxes
Last Year Sales: 130 boxes
Historical Median: 125 boxes


Forecast Calculations:
Historical benchmarks (e.g., Simple Avg: 100 boxes/month, 3-Month MA: 105 boxes/month).
Prophet forecast summary (e.g., Total Sales for 2026: 1,800 boxes).



Notes

Data Quality: Ensure Excel sheets have consistent column names and numeric sales values. Missing or invalid data may result in warnings or skipped sheets.
Forecast Limitations: Prophet requires at least 12 months of data per city for reliable forecasting. Cities with insufficient data are skipped.
Performance: Large datasets or many cities may increase processing time. Caching is used to minimize delays.
Customization: Modify the code to adjust Prophet parameters (e.g., changepoint_prior_scale) or add new visualizations as needed.

Troubleshooting

Error: "No valid data could be processed": Check that sheet names follow the YYYY-YYYY format and include all required columns.
Warning: "Prophet forecast failed for [City]": Ensure the city has at least 12 months of non-zero sales data.
Charts not displaying: Verify that filters (cities, years, months) result in non-empty data.
Slow performance: Reduce the number of forecast cities or use a smaller dataset.

License
This project is provided for educational and professional use. Ensure compliance with any organizational data usage policies when deploying.
Contact
For support or feature requests, contact the developer or refer to the Streamlit and Prophet documentation:

Streamlit Docs
Prophet Docs

