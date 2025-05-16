import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Constants for Months ---
MONTH_NAME_TO_NUM = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
MONTH_NUM_TO_NAME = {v: k for k, v in MONTH_NAME_TO_NUM.items()}
ALL_MONTH_NAMES = list(MONTH_NAME_TO_NUM.keys())
MONTH_SORTER = {name: num for num, name in enumerate(ALL_MONTH_NAMES)}

# Streamlit app configuration
st.set_page_config(page_title="Prit Enterprise Sales & Forecasting Tool", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Prit Enterprise Sales & Forecasting Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555555;'>Interactive analysis, planning, and forecasting (Forecasts up to 36 months ahead)</p>", unsafe_allow_html=True)

# --- Helper Functions ---
def get_forecast_date_range(last_historical_date, months_ahead):
    if last_historical_date is None or not isinstance(last_historical_date, pd.Timestamp) or months_ahead <= 0:
        return None, None
    # Forecast starts the first day of the month following the last historical data month
    forecast_start = (last_historical_date.replace(day=1) + pd.DateOffset(months=1))
    
    forecast_end = forecast_start + pd.DateOffset(months=months_ahead - 1)
    forecast_end = forecast_end + pd.offsets.MonthEnd(0) # Ensure it's end of month
    return forecast_start, forecast_end

def get_forecast_period_name(forecast_start, forecast_end):
    if forecast_start is None or forecast_end is None: return "N/A"
    if forecast_start.month == forecast_end.month and forecast_start.year == forecast_end.year:
        return forecast_start.strftime("%B %Y")
    return f"{forecast_start.strftime('%b %Y')} - {forecast_end.strftime('%b %Y')}"

@st.cache_data
def calculate_basic_forecasts(_df_hist_for_city_input):
    if _df_hist_for_city_input.empty or 'y' not in _df_hist_for_city_input.columns or 'ds' not in _df_hist_for_city_input.columns:
        return {}
    df_calc = _df_hist_for_city_input.copy()
    df_calc['ds'] = pd.to_datetime(df_calc['ds'])
    df_calc = df_calc.set_index('ds').sort_index()
    monthly_sales = df_calc['y'].resample('M').sum().fillna(0)

    if monthly_sales.empty:
        return {
            'Simple Avg (Monthly)': np.nan, '3-Month Moving Avg (Monthly)': np.nan,
            '6-Month Moving Avg (Monthly)': np.nan, '12-Month Moving Avg (Monthly)': np.nan,
            'Weighted Moving Avg (Monthly)': np.nan, 'Last Year Same Month (LYSM)': np.nan,
            'Latest Full Year Avg (Monthly)': np.nan, 'Annual Growth Rate (YoY %)': 'N/A'
        }
    yearly_totals = monthly_sales.groupby(monthly_sales.index.year).sum()
    yearly_counts = monthly_sales.groupby(monthly_sales.index.year).count()
    complete_years = yearly_counts[yearly_counts == 12].index
    latest_full_year_avg_monthly = np.nan
    if not complete_years.empty:
        latest_complete_year = complete_years.max()
        latest_full_year_avg_monthly = monthly_sales[monthly_sales.index.year == latest_complete_year].mean()

    simple_avg = monthly_sales.mean() if not monthly_sales.empty else np.nan
    moving_avg_3m = monthly_sales.rolling(window=3, min_periods=1).mean().iloc[-1] if len(monthly_sales) >= 1 else np.nan
    moving_avg_6m = monthly_sales.rolling(window=6, min_periods=1).mean().iloc[-1] if len(monthly_sales) >= 1 else np.nan
    moving_avg_12m = monthly_sales.rolling(window=12, min_periods=1).mean().iloc[-1] if len(monthly_sales) >= 1 else np.nan

    wma = np.nan
    if len(monthly_sales) >= 1:
        window_size = min(max(1,len(monthly_sales)), 12)
        weights = np.arange(1, window_size + 1)
        relevant_sales = monthly_sales.tail(window_size)
        if not relevant_sales.empty and np.sum(weights[:len(relevant_sales)]) != 0:
             wma = np.sum(weights[:len(relevant_sales)] * relevant_sales.to_numpy()) / np.sum(weights[:len(relevant_sales)])


    lysm = np.nan
    if len(monthly_sales) >= 13:
        last_month_hist_series = monthly_sales.index[-1]
        lysm_val_series = monthly_sales[
            (monthly_sales.index.month == last_month_hist_series.month) &
            (monthly_sales.index.year == last_month_hist_series.year - 1)
        ]
        if not lysm_val_series.empty: lysm = lysm_val_series.iloc[0]

    yearly_growth = 'N/A'
    if len(yearly_totals) >= 2:
        last_year_total = yearly_totals.iloc[-1]
        prev_year_total = yearly_totals.iloc[-2]
        if prev_year_total != 0:
            yearly_growth = (last_year_total / prev_year_total - 1) * 100
        elif last_year_total > 0: yearly_growth = np.inf
        elif last_year_total == 0 and prev_year_total == 0: yearly_growth = 0.0

    seasonal_factors = {}
    if len(monthly_sales) >= 12:
        month_avg_sales = monthly_sales.groupby(monthly_sales.index.month).mean()
        overall_monthly_avg = monthly_sales.mean()
        if overall_monthly_avg != 0:
            for month_num, avg_val in month_avg_sales.items():
                month_name_sf = pd.Timestamp(f'2000-{month_num}-01').strftime('%b')
                seasonal_factors[f'Seasonal Factor ({month_name_sf})'] = avg_val / overall_monthly_avg
        else:
             for month_num in range(1,13):
                month_name_sf = pd.Timestamp(f'2000-{month_num}-01').strftime('%b')
                seasonal_factors[f'Seasonal Factor ({month_name_sf})'] = np.nan
    else:
        for month_num in range(1,13):
            month_name_sf = pd.Timestamp(f'2000-{month_num}-01').strftime('%b')
            seasonal_factors[f'Seasonal Factor ({month_name_sf})'] = np.nan
    result = {
        'Simple Avg (Monthly)': simple_avg, '3-Month Moving Avg (Monthly)': moving_avg_3m,
        '6-Month Moving Avg (Monthly)': moving_avg_6m, '12-Month Moving Avg (Monthly)': moving_avg_12m,
        'Weighted Moving Avg (Monthly)': wma, 'Last Year Same Month (LYSM)': lysm,
        'Latest Full Year Avg (Monthly)': latest_full_year_avg_monthly, 'Annual Growth Rate (YoY %)': yearly_growth
    }
    result.update(seasonal_factors)
    return result

@st.cache_data
def load_data(uploaded_file_obj):
    if uploaded_file_obj is None: return None
    try:
        sheets = pd.read_excel(uploaded_file_obj, sheet_name=None)
        all_data = []
        for sheet_name, df_sheet in sheets.items():
            try:
                fiscal_year = int(sheet_name.split('-')[0])
            except (ValueError, IndexError):
                st.warning(f"Sheet name '{sheet_name}' not in 'YYYY-YYYY' format. Skipping.")
                continue
            required_cols = ['AGENCY NAME', 'CITY', 'April', 'May', 'June', 'July', 'August',
                             'September', 'October', 'November', 'December', 'January', 'February', 'March']
            if not all(col in df_sheet.columns for col in required_cols):
                st.error(f"Sheet '{sheet_name}' missing required columns. Check: {', '.join(required_cols)}. Skipping.")
                continue
            df_sheet.dropna(subset=['CITY'], inplace=True)
            df_melted = df_sheet.melt(id_vars=['AGENCY NAME', 'CITY'], value_vars=required_cols[2:],
                                      var_name='month_name', value_name='sales')
            df_melted['fiscal_year'] = fiscal_year
            df_melted['calendar_year'] = df_melted.apply(
                lambda row: fiscal_year if row['month_name'] not in ['January', 'February', 'March'] else fiscal_year + 1, axis=1)
            df_melted['month_num'] = df_melted['month_name'].map(MONTH_NAME_TO_NUM)
            df_melted['ds'] = pd.to_datetime(df_melted['calendar_year'].astype(str) + '-' +
                                             df_melted['month_num'].astype(str) + '-01', errors='coerce')
            df_melted = df_melted.rename(columns={'AGENCY NAME': 'agency', 'CITY': 'city', 'sales': 'y'})
            df_melted = df_melted.dropna(subset=['ds', 'city'])
            df_melted['y'] = pd.to_numeric(df_melted['y'], errors='coerce').fillna(0)
            all_data.append(df_melted[['ds', 'city', 'agency', 'month_name', 'month_num', 'fiscal_year', 'calendar_year', 'y']])

        if not all_data:
            st.error("No valid data could be processed from the uploaded file. Please check sheet names and column headers.")
            return None
        df_combined = pd.concat(all_data, ignore_index=True)
        df_combined['y'] = pd.to_numeric(df_combined['y'], errors='coerce').fillna(0)
        df_combined['y'] = df_combined.groupby(['city', 'fiscal_year'])['y'].transform(
            lambda x: x.replace(0, x[x!=0].median() if pd.notna(x[x!=0].median()) and x[x!=0].median() > 0 else 0)
        )
        df_combined['y'] = df_combined['y'].fillna(0)
        return df_combined.sort_values(by=['city', 'ds']).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def generate_inventory_suggestions(
        _current_df_for_hist, _forecast_data_df_for_city_components,
        _z_val, _target_city, _target_year, _target_months_nums):
    city_specific_forecast_components = pd.DataFrame()
    if _forecast_data_df_for_city_components is not None and not _forecast_data_df_for_city_components.empty:
        if all(col in _forecast_data_df_for_city_components.columns for col in ['city', 'ds', 'yhat']):
            city_specific_forecast_components = _forecast_data_df_for_city_components[_forecast_data_df_for_city_components['city'] == _target_city].copy()
            if not city_specific_forecast_components.empty:
                 city_specific_forecast_components['ds'] = pd.to_datetime(city_specific_forecast_components['ds'])

    city_hist_df = _current_df_for_hist[_current_df_for_hist['city'] == _target_city].copy()
    if city_hist_df.empty or not _target_months_nums:
        return pd.DataFrame(columns=['Month-Year', 'Suggested Stock (Boxes)', 'Sales Target (Boxes)', 'Last Year Sales (Boxes)', 'Historical Median (Boxes)'])

    suggestions = []
    last_full_fiscal_year_in_data = city_hist_df['fiscal_year'].max() if 'fiscal_year' in city_hist_df.columns and not city_hist_df['fiscal_year'].empty else None

    for month_num_inv in _target_months_nums:
        month_name_inv = MONTH_NUM_TO_NAME[month_num_inv]
        city_month_hist_sales = city_hist_df[city_hist_df['month_name'] == month_name_inv]['y']
        last_year_month_sales = 0
        if last_full_fiscal_year_in_data is not None:
             ly_sales_series = city_hist_df[
                (city_hist_df['fiscal_year'] == last_full_fiscal_year_in_data) & (city_hist_df['month_name'] == month_name_inv)
            ]['y']
             if not ly_sales_series.empty: last_year_month_sales = ly_sales_series.iloc[0]

        median_sales = city_month_hist_sales.median() if not city_month_hist_sales.empty else 0
        std_sales = city_month_hist_sales.std() if len(city_month_hist_sales) >= 2 else 0
        median_sales = 0 if pd.isna(median_sales) else median_sales
        std_sales = 0 if pd.isna(std_sales) else std_sales
        base_stock = median_sales + _z_val * std_sales
        forecast_value_this_month = 0
        if not city_specific_forecast_components.empty:
            month_forecast = city_specific_forecast_components[
                (city_specific_forecast_components['ds'].dt.month == month_num_inv) &
                (city_specific_forecast_components['ds'].dt.year == _target_year)
            ]
            if not month_forecast.empty: forecast_value_this_month = month_forecast['yhat'].mean()

        suggested_stock = base_stock
        if pd.notna(forecast_value_this_month) and forecast_value_this_month > 0:
            if median_sales > 0:
                suggested_stock = base_stock * (forecast_value_this_month / median_sales)
            else:
                suggested_stock = forecast_value_this_month * (1 + _z_val * 0.1)
        elif pd.notna(forecast_value_this_month) and forecast_value_this_month <= 0 and median_sales > 0:
            suggested_stock = base_stock
        elif pd.notna(forecast_value_this_month) and forecast_value_this_month <= 0 and median_sales <=0:
             suggested_stock = 0
        sales_target = max(last_year_month_sales * 1.1, forecast_value_this_month if pd.notna(forecast_value_this_month) else 0)
        suggestions.append({
            'Month-Year': f"{month_name_inv} {_target_year}",
            'Suggested Stock (Boxes)': int(round(suggested_stock)),
            'Sales Target (Boxes)': int(round(sales_target)),
            'Last Year Sales (Boxes)': int(round(last_year_month_sales)),
            'Historical Median (Boxes)': int(round(median_sales))
        })
    return pd.DataFrame(suggestions)

@st.cache_data
def run_forecast_for_cities(_df_full_hist_for_prophet, _cities_list_for_prophet, _horizon_months_actual_prophet):
    models_dict_fc = {}
    forecasts_components_list_fc = []
    city_full_forecast_objects_fc = {}
    successful_fc_cities_list = []

    if not isinstance(_horizon_months_actual_prophet, int) or _horizon_months_actual_prophet <= 0:
        # st.warning("Invalid forecast horizon in run_forecast_for_cities. Must be a positive integer.") # Avoid st calls in cached func
        return models_dict_fc, pd.DataFrame(), city_full_forecast_objects_fc, successful_fc_cities_list

    for city_name_fc_run in _cities_list_for_prophet:
        city_hist_data_fc_run = _df_full_hist_for_prophet[_df_full_hist_for_prophet['city'] == city_name_fc_run][['ds', 'y']].copy()
        city_hist_data_fc_run = city_hist_data_fc_run.groupby('ds')['y'].sum().reset_index().sort_values('ds')
        city_hist_data_fc_run['ds'] = pd.to_datetime(city_hist_data_fc_run['ds'])

        if len(city_hist_data_fc_run['ds'].unique()) >= 12:
            model_prophet_fc = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
            try:
                model_prophet_fc.fit(city_hist_data_fc_run)
                future_dates_fc_run = model_prophet_fc.make_future_dataframe(periods=_horizon_months_actual_prophet, freq='MS')
                forecast_output_fc_run = model_prophet_fc.predict(future_dates_fc_run)
                forecast_output_fc_run['yhat'] = np.maximum(0, forecast_output_fc_run['yhat'])
                forecast_output_fc_run['yhat_lower'] = np.maximum(0, forecast_output_fc_run['yhat_lower'])
                forecast_output_fc_run['yhat_upper'] = np.maximum(0, forecast_output_fc_run['yhat_upper'])
                forecast_output_fc_run['city'] = city_name_fc_run
                models_dict_fc[city_name_fc_run] = model_prophet_fc
                city_full_forecast_objects_fc[city_name_fc_run] = forecast_output_fc_run
                components_to_add = {'ds': forecast_output_fc_run['ds'], 'yhat': forecast_output_fc_run['yhat'],
                                        'yhat_lower': forecast_output_fc_run['yhat_lower'], 'yhat_upper': forecast_output_fc_run['yhat_upper'],
                                        'city': city_name_fc_run, 'trend': forecast_output_fc_run['trend']}
                if 'yearly' in forecast_output_fc_run.columns:
                    components_to_add['yearly'] = forecast_output_fc_run['yearly']
                forecasts_components_list_fc.append(pd.DataFrame(components_to_add))
                successful_fc_cities_list.append(city_name_fc_run)
            except Exception: # as e_fc_run:
                # st.warning(f"Prophet forecast failed for {city_name_fc_run}: {e_fc_run}") # Avoid st calls
                print(f"Prophet forecast failed for {city_name_fc_run}") # Log to console for debugging
        else:
            # st.warning(f"Skipping forecast for {city_name_fc_run}: needs 12+ data points. Found {len(city_hist_data_fc_run['ds'].unique())}.")
            print(f"Skipping forecast for {city_name_fc_run}: needs 12+ data points. Found {len(city_hist_data_fc_run['ds'].unique())}.")
    
    if forecasts_components_list_fc:
        all_components_df = pd.concat(forecasts_components_list_fc, ignore_index=True)
        return models_dict_fc, all_components_df, city_full_forecast_objects_fc, successful_fc_cities_list
    return models_dict_fc, pd.DataFrame(), city_full_forecast_objects_fc, successful_fc_cities_list

# --- Initialize Session State ---
def init_session_state():
    ss = st.session_state
    if 'forecast_horizon' not in ss: ss.forecast_horizon = 12
    if 'risk_level' not in ss: ss.risk_level = "Medium (95% Service Level)"
    current_year = datetime.now().year
    if 'inventory_target_year' not in ss:
        ss.inventory_target_year = current_year + 1 if datetime.now().month > 6 else current_year
    default_inv_months = [MONTH_NUM_TO_NAME.get(m) for m in [3, 4, 5]]
    if 'inventory_target_months' not in ss: ss.inventory_target_months = [m for m in default_inv_months if m]
    if 'selected_display_months' not in ss: ss.selected_display_months = ALL_MONTH_NAMES
    if 'display_cities_ms' not in ss: ss.display_cities_ms = []
    if 'display_years_ms' not in ss: ss.display_years_ms = []
    if 'forecast_cities_ms' not in ss: ss.forecast_cities_ms = []
    if 'tab3_focus_city' not in ss: ss.tab3_focus_city = None
    if 'adv_city_select' not in ss: ss.adv_city_select = None

init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Upload Sales Data (Excel)", type="xlsx",
                                     help="Excel file with sheets 'YYYY-YYYY'. Ensure 'CITY', 'AGENCY NAME' and all month columns are present.")
    st.markdown("---")
    st.header("Global Forecast Settings")
    st.session_state.forecast_horizon = st.slider("Months to Forecast Ahead", 1, 36,
                                                  st.session_state.forecast_horizon, key="forecast_horizon_slider",
                                                  help="Select how many months into the future to forecast from the end of historical data.")
    
    risk_level_options = ["Low (90% Service Level)", "Medium (95% Service Level)", "High (99% Service Level)"]
    st.session_state.risk_level = st.selectbox("Inventory Risk Level (Safety Stock)", risk_level_options,
                                                index=risk_level_options.index(st.session_state.risk_level),
                                                key="risk_level_sb")
    z_score_map = {"Low (90% Service Level)": 1.28, "Medium (95% Service Level)": 1.65, "High (99% Service Level)": 2.33}
    z_score_val = z_score_map[st.session_state.risk_level]

    st.markdown("---")
    st.header("Inventory Planning Settings")
    current_year_inv_planning = datetime.now().year
    st.session_state.inventory_target_year = st.number_input("Target Year for Inventory Plan", min_value=current_year_inv_planning, max_value=current_year_inv_planning + 5,
                                       value=st.session_state.inventory_target_year, key="inv_target_year_ni")
    st.session_state.inventory_target_months = st.multiselect("Target Months for Inventory Plan", options=ALL_MONTH_NAMES,
                                             default=st.session_state.inventory_target_months, key="inv_target_months_ms",
                                             help="Select months for inventory suggestions.")
    target_inv_months_nums_sb = [MONTH_NAME_TO_NUM[m] for m in st.session_state.inventory_target_months]

    st.markdown("---")
    st.header("Display Filters")

# --- Main App Logic ---
df_original = None
if uploaded_file is not None:
    df_original = load_data(uploaded_file)

if df_original is not None and not df_original.empty:
    original_ds_min = df_original['ds'].min()
    original_ds_max = df_original['ds'].max()
    available_cities_list = sorted(df_original['city'].unique())
    available_fiscal_years_list = sorted(df_original['fiscal_year'].unique(), reverse=True)
    
    if not st.session_state.display_cities_ms or not all(c in available_cities_list for c in st.session_state.display_cities_ms):
        st.session_state.display_cities_ms = available_cities_list[:]
    if not st.session_state.display_years_ms or not all(y in available_fiscal_years_list for y in st.session_state.display_years_ms):
        st.session_state.display_years_ms = available_fiscal_years_list[:]
    if not st.session_state.forecast_cities_ms or not all(c in available_cities_list for c in st.session_state.forecast_cities_ms):
        st.session_state.forecast_cities_ms = available_cities_list[:]

    actual_forecast_months = st.session_state.forecast_horizon
    forecast_start_calc, forecast_end_calc = get_forecast_date_range(original_ds_max, actual_forecast_months)
    
    if actual_forecast_months > 0 and (forecast_start_calc is None or forecast_end_calc is None):
        st.sidebar.warning(f"Cannot create a valid {actual_forecast_months}-month forecast range from historical data ending {original_ds_max.strftime('%b %Y') if original_ds_max else 'N/A'}. Forecast disabled.")
        actual_forecast_months = 0
    forecast_period_name_global = get_forecast_period_name(forecast_start_calc, forecast_end_calc)

    original_total_sales = df_original['y'].sum()
    original_num_cities = df_original['city'].nunique()

    with st.sidebar:
        st.session_state.display_cities_ms = st.multiselect("Filter Cities (Display)", options=available_cities_list, default=st.session_state.display_cities_ms, key="display_cities_widget")
        st.session_state.display_years_ms = st.multiselect("Filter Fiscal Years (Display)", options=available_fiscal_years_list, default=st.session_state.display_years_ms, key="display_years_widget")
        st.session_state.selected_display_months = st.multiselect("Filter Display by Months", options=ALL_MONTH_NAMES,
                                                     default=st.session_state.selected_display_months, key="display_months_widget")
        st.markdown("---")
        st.header("Forecast Target Cities")
        st.session_state.forecast_cities_ms = st.multiselect("Select Cities for Prophet Forecast", options=available_cities_list, default=st.session_state.forecast_cities_ms, key="forecast_cities_widget")

    df_filtered_for_display = df_original.copy()
    if st.session_state.display_cities_ms:
        df_filtered_for_display = df_filtered_for_display[df_filtered_for_display['city'].isin(st.session_state.display_cities_ms)]
    if st.session_state.display_years_ms:
        df_filtered_for_display = df_filtered_for_display[df_filtered_for_display['fiscal_year'].isin(st.session_state.display_years_ms)]
    if st.session_state.selected_display_months:
        df_filtered_for_display = df_filtered_for_display[df_filtered_for_display['month_name'].isin(st.session_state.selected_display_months)]

    prophet_models = {}
    prophet_forecasts_full = {}
    all_cities_forecast_components_df = pd.DataFrame()
    successfully_forecasted_cities = []
    basic_forecasts_master_df = pd.DataFrame()

    if actual_forecast_months > 0 and st.session_state.forecast_cities_ms:
        training_data_for_prophet_run = df_original[df_original['city'].isin(st.session_state.forecast_cities_ms)].copy()
        if not training_data_for_prophet_run.empty:
            prophet_models, all_cities_forecast_components_df, prophet_forecasts_full, successfully_forecasted_cities = run_forecast_for_cities(
                training_data_for_prophet_run, st.session_state.forecast_cities_ms, actual_forecast_months
            )
            if successfully_forecasted_cities:
                basic_forecasts_data_list_main = []
                for city_for_basic_main in successfully_forecasted_cities:
                    city_hist_data_for_basic_main = df_original[df_original['city'] == city_for_basic_main][['ds', 'y']].copy()
                    if not city_hist_data_for_basic_main.empty:
                        basics_main = calculate_basic_forecasts(city_hist_data_for_basic_main)
                        if basics_main:
                            basics_main['City'] = city_for_basic_main
                            basic_forecasts_data_list_main.append(basics_main)
                if basic_forecasts_data_list_main:
                    basic_forecasts_master_df = pd.DataFrame(basic_forecasts_data_list_main)

            if not successfully_forecasted_cities:
                st.session_state.tab3_focus_city = None
                st.session_state.adv_city_select = None
            else:
                if st.session_state.tab3_focus_city not in successfully_forecasted_cities or st.session_state.tab3_focus_city is None:
                    st.session_state.tab3_focus_city = successfully_forecasted_cities[0]
                if st.session_state.adv_city_select not in successfully_forecasted_cities or st.session_state.adv_city_select is None:
                    st.session_state.adv_city_select = successfully_forecasted_cities[0]
        else:
            st.sidebar.info("No historical data available for the cities selected for Prophet forecast.")
            successfully_forecasted_cities, basic_forecasts_master_df = [], pd.DataFrame()
            st.session_state.tab3_focus_city, st.session_state.adv_city_select = None, None
    else:
        successfully_forecasted_cities, basic_forecasts_master_df = [], pd.DataFrame()
        st.session_state.tab3_focus_city, st.session_state.adv_city_select = None, None
        if actual_forecast_months <= 0 and st.session_state.forecast_cities_ms:
             st.sidebar.warning("Forecast horizon is 0 (or became 0 due to data proximity/settings). No Prophet forecast generated.")
        elif not st.session_state.forecast_cities_ms:
             st.sidebar.warning("Select cities under 'Forecast Target Cities' to generate Prophet forecasts.")

    total_forecast_aggregated = pd.DataFrame()
    if not all_cities_forecast_components_df.empty and forecast_start_calc:
        forecast_period_data_agg = all_cities_forecast_components_df[all_cities_forecast_components_df['ds'] >= forecast_start_calc]
        if not forecast_period_data_agg.empty:
            total_forecast_aggregated = forecast_period_data_agg.groupby('ds')[['yhat', 'yhat_lower', 'yhat_upper']].sum().reset_index()

    # --- Tabs ---
    tab_titles = ["📈 Main Dashboard", "💡 Advanced Insights & Planning", "🧮 Forecast Calculations", "📊 Dynamic Visualizations"]
    tab1, tab2, tab3, tab4_viz = st.tabs(tab_titles)

    with tab1: # Main Dashboard
        # ... (Tab 1 content as in previous robust version)
        st.markdown("### Data Snapshot")
        st.subheader("Overall Dataset Summary (All Uploaded Data)")
        overall_col1, overall_col2, overall_col3 = st.columns(3)
        with overall_col1: st.metric("Grand Total Sales", f"{int(original_total_sales):,}")
        with overall_col2: st.metric("Total Cities in Dataset", original_num_cities)
        with overall_col3: st.metric("Full Data Period", f"{original_ds_min.strftime('%b %Y')} - {original_ds_max.strftime('%b %Y')}")

        st.markdown("---")
        st.subheader("Current Filtered View Summary (Based on Sidebar Display Filters)")
        filtered_col1, filtered_col2, filtered_col3 = st.columns(3)
        total_filtered_sales_val = df_filtered_for_display['y'].sum() if not df_filtered_for_display.empty else 0
        num_filtered_cities_val = df_filtered_for_display['city'].nunique() if not df_filtered_for_display.empty else 0
        filtered_period_str = "N/A"
        if not df_filtered_for_display.empty:
            min_f_ds = df_filtered_for_display['ds'].min().strftime('%b %Y')
            max_f_ds = df_filtered_for_display['ds'].max().strftime('%b %Y')
            filtered_period_str = f"{min_f_ds} - {max_f_ds}" if min_f_ds != max_f_ds else min_f_ds
        with filtered_col1: st.metric("Filtered Sales (Boxes)", f"{int(total_filtered_sales_val):,}")
        with filtered_col2: st.metric("Filtered Cities", num_filtered_cities_val)
        with filtered_col3: st.metric("Filtered Period", filtered_period_str)

        st.markdown("---")
        st.markdown("### Sales Visualizations (Current Filtered View)")
        viz_col1_main, viz_col2_main = st.columns([3, 2])
        with viz_col1_main:
            st.markdown("#### Yearly Sales Trend")
            if not df_filtered_for_display.empty:
                yearly_sales_filtered_df = df_filtered_for_display.groupby(df_filtered_for_display['ds'].dt.year)['y'].sum().reset_index()
                yearly_sales_filtered_df.rename(columns={'y': 'Sales', 'ds': 'Year'}, inplace=True)
                if not yearly_sales_filtered_df.empty:
                    fig_yearly = px.bar(yearly_sales_filtered_df, x='Year', y='Sales', title="Total Sales by Year (Filtered View)", labels={'Sales': 'Sales (Boxes)'}, height=400, text_auto=True)
                    fig_yearly.update_layout(xaxis_type='category')
                    st.plotly_chart(fig_yearly, use_container_width=True)
                else: st.info("No yearly sales data for current filter selection.")
            else: st.info("Apply display filters or upload data to see yearly sales.")

        with viz_col2_main:
            title_selected_months = "Selected Months" if len(st.session_state.selected_display_months) != 1 else st.session_state.selected_display_months[0]
            st.markdown(f"#### Sales by City for {title_selected_months} (Filtered View, Top 15)")
            if not df_filtered_for_display.empty and st.session_state.selected_display_months:
                city_sel_months_totals_df = df_filtered_for_display.groupby('city')['y'].sum().reset_index()
                city_sel_months_totals_df.rename(columns={'y': 'Sales for Selected Months', 'city':'City'}, inplace=True)
                city_sel_months_totals_df = city_sel_months_totals_df.sort_values(by="Sales for Selected Months", ascending=False).head(15)
                if not city_sel_months_totals_df.empty:
                    fig_sel_months_city = px.bar(city_sel_months_totals_df, x='City', y='Sales for Selected Months', title=f"Sales by City for {title_selected_months} (Filtered View)", labels={'Sales for Selected Months': 'Sales (Boxes)'}, height=400, text_auto=True)
                    st.plotly_chart(fig_sel_months_city, use_container_width=True)
                else: st.info(f"No sales data for {title_selected_months} by city in current filter selection.")
            elif not st.session_state.selected_display_months:
                st.info("Select months in the sidebar under 'Display Filters' to see this chart.")
            else: st.info("Apply display filters or upload data to see this chart.")
        
        st.markdown("---")
        st.markdown(f"### Prophet Sales Forecast (For Cities in Forecast Target Cities)")
        fc_viz_col1, fc_viz_col2 = st.columns([3,2])
        with fc_viz_col1:
            st.markdown(f"#### Historical vs. Forecasted Sales ({forecast_period_name_global})")
            if not st.session_state.forecast_cities_ms or actual_forecast_months <= 0:
                st.info("Select cities for forecast and ensure forecast horizon > 0 in the sidebar.")
            elif not total_forecast_aggregated.empty and forecast_start_calc and forecast_end_calc:
                hist_data_for_fc_cities = df_original[df_original['city'].isin(successfully_forecasted_cities)]
                if not hist_data_for_fc_cities.empty:
                    hist_annual_sales_fc_cities_df = hist_data_for_fc_cities.groupby(hist_data_for_fc_cities['ds'].dt.year)['y'].sum().reset_index()
                    hist_annual_sales_fc_cities_df.rename(columns={'y': 'Sales', 'ds': 'Year'}, inplace=True)
                else:
                    hist_annual_sales_fc_cities_df = pd.DataFrame(columns=['Year', 'Sales'])

                fig_hist_fc_go = go.Figure()
                if not hist_annual_sales_fc_cities_df.empty:
                    fig_hist_fc_go.add_trace(go.Bar(x=hist_annual_sales_fc_cities_df['Year'], y=hist_annual_sales_fc_cities_df['Sales'], name='Historical Annual Sales (Forecasted Cities)', marker_color='#2E86C1'))
                
                forecast_period_sum_plot_tab1 = total_forecast_aggregated[
                    (total_forecast_aggregated['ds'] >= forecast_start_calc) &
                    (total_forecast_aggregated['ds'] <= forecast_end_calc)
                ]
                if not forecast_period_sum_plot_tab1.empty:
                    yhat_val = forecast_period_sum_plot_tab1['yhat'].sum()
                    yhat_lower_val = forecast_period_sum_plot_tab1['yhat_lower'].sum()
                    yhat_upper_val = forecast_period_sum_plot_tab1['yhat_upper'].sum()
                    fig_hist_fc_go.add_trace(go.Bar(x=[forecast_period_name_global], y=[yhat_val], name=f'Forecast: {forecast_period_name_global}', marker_color='#28B463',
                                                    error_y=dict(type='data', symmetric=False, array=[max(0,yhat_upper_val - yhat_val)], arrayminus=[max(0,yhat_val - yhat_lower_val)], color='gray', thickness=1.5, width=3)))
                
                fig_hist_fc_go.update_layout(title=f"Total Sales: Past Years vs. {forecast_period_name_global} Forecast", xaxis_title="Period", yaxis_title="Sales (Boxes)", height=400, xaxis_type='category', legend_title_text='Data Type', barmode='group')
                if not hist_annual_sales_fc_cities_df.empty or not forecast_period_sum_plot_tab1.empty:
                    st.plotly_chart(fig_hist_fc_go, use_container_width=True)
                    if not forecast_period_sum_plot_tab1.empty: st.caption(f"Forecast for {forecast_period_name_global} includes a 95% confidence interval based on the sum of individual city intervals.")
                else: st.info("Not enough historical or forecast data to plot for the selected forecast cities and period.")
            elif st.session_state.forecast_cities_ms and not successfully_forecasted_cities:
                 st.warning("Prophet forecasts could not be generated for any of the selected cities. Check warnings in sidebar or console.")
            else: st.info("No aggregated forecast data available. Check forecast settings and data.")

        with fc_viz_col2:
            st.markdown(f"#### Forecast ({forecast_period_name_global}) by City (Top 15)")
            if not all_cities_forecast_components_df.empty and forecast_start_calc and forecast_end_calc :
                horizon_fcst_data_df = all_cities_forecast_components_df[
                    (all_cities_forecast_components_df['ds'] >= forecast_start_calc) &
                    (all_cities_forecast_components_df['ds'] <= forecast_end_calc)
                ]
                if not horizon_fcst_data_df.empty:
                    city_fcst_horizon_totals_df = horizon_fcst_data_df.groupby('city')['yhat'].sum().reset_index()
                    forecast_col_name = f'Forecasted Sales ({forecast_period_name_global})'
                    city_fcst_horizon_totals_df.rename(columns={'yhat': forecast_col_name, 'city': 'City'}, inplace=True)
                    city_fcst_horizon_totals_df = city_fcst_horizon_totals_df.sort_values(by=forecast_col_name, ascending=False).head(15)
                    if not city_fcst_horizon_totals_df.empty:
                        fig_horizon_fc_city = px.bar(city_fcst_horizon_totals_df, x='City', y=forecast_col_name, title=f"Forecast ({forecast_period_name_global}) by City", labels={forecast_col_name: 'Sales (Boxes)'}, height=400, text_auto=True)
                        st.plotly_chart(fig_horizon_fc_city, use_container_width=True)
                    else: st.info(f"No forecast totals by city to display for the period {forecast_period_name_global}.")
                else: st.info(f"No forecast data available for any city within the period: {forecast_period_name_global}.")
            elif st.session_state.forecast_cities_ms and not successfully_forecasted_cities and actual_forecast_months > 0:
                 st.warning("Prophet forecasts failed for all selected cities. Check warnings.")
            elif actual_forecast_months <= 0: st.info("Forecast horizon is 0, cannot display forecast by city.")
            else: st.info("Select cities for forecast (Sidebar > Forecast Target Cities).")
        
        st.markdown("---")
        st.subheader("Sales Data Table (Current Filtered View)")
        if not df_filtered_for_display.empty:
            df_display_table = df_filtered_for_display[['fiscal_year', 'month_name', 'city', 'agency', 'y']].copy()
            df_display_table.rename(columns={'y': 'Sales (Boxes)', 'agency': 'Agency Name', 'city': 'City', 'month_name': 'Month', 'fiscal_year': 'Fiscal Year'}, inplace=True)
            df_display_table = df_display_table.sort_values(by=['Fiscal Year', 'City', 'Month'])
            st.info("Sort table by clicking headers. Expand by dragging bottom-right corner.")
            st.data_editor(df_display_table, column_config={"Sales (Boxes)": st.column_config.NumberColumn(format="%d")}, use_container_width=True, height=400, num_rows="dynamic")
            csv_filtered = df_display_table.to_csv(index=False).encode('utf-8')
            st.download_button("Download Filtered Data as CSV", csv_filtered, 'filtered_sales_data.csv', 'text/csv', key='download_filtered_csv')
        else: st.info("No data in table based on current filters.")


    with tab2: # Advanced Insights & Planning
        st.header("Advanced Forecast Analysis & Inventory Planning")
        if not st.session_state.forecast_cities_ms or actual_forecast_months <= 0:
            st.warning("Please select cities for forecast and ensure forecast horizon > 0 in the sidebar.")
        elif not successfully_forecasted_cities:
            st.error("Prophet forecasts could not be generated for any selected cities. This tab requires successful forecasts.")
        else:
            if st.session_state.adv_city_select not in successfully_forecasted_cities or st.session_state.adv_city_select is None:
                 st.session_state.adv_city_select = successfully_forecasted_cities[0] if successfully_forecasted_cities else None
            
            if st.session_state.adv_city_select:
                st.session_state.adv_city_select = st.selectbox(
                    "Select City for Deep Dive Analysis:", 
                    options=successfully_forecasted_cities, 
                    index=successfully_forecasted_cities.index(st.session_state.adv_city_select),
                    key="adv_city_selector_widget_tab2_final" 
                )
                selected_adv_city_tab2 = st.session_state.adv_city_select

                if selected_adv_city_tab2:
                    st.subheader(f"Prophet Forecast Decomposition for: {selected_adv_city_tab2}")
                    if selected_adv_city_tab2 in prophet_models and selected_adv_city_tab2 in prophet_forecasts_full:
                        fig_components_tab2 = prophet_models[selected_adv_city_tab2].plot_components(prophet_forecasts_full[selected_adv_city_tab2])
                        st.pyplot(fig_components_tab2)
                        st.caption("Trend: Estimated long-term sales direction. Yearly Seasonality: Typical sales fluctuations within a year.")
                    else: st.error(f"Could not retrieve forecast component data for {selected_adv_city_tab2}.")

                    st.markdown("---")
                    inv_months_str_tab2 = ", ".join(st.session_state.inventory_target_months) if st.session_state.inventory_target_months else "selected months"
                    st.subheader(f"Inventory Planning for {inv_months_str_tab2} {st.session_state.inventory_target_year}: {selected_adv_city_tab2}")
                    if not st.session_state.inventory_target_months:
                        st.warning("Please select target months for inventory planning in the sidebar.")
                    else:
                        hist_for_inventory_city_tab2 = df_original[df_original['city'] == selected_adv_city_tab2]
                        inventory_suggestions_df_tab2 = generate_inventory_suggestions(
                            hist_for_inventory_city_tab2, all_cities_forecast_components_df,
                            z_score_val, selected_adv_city_tab2, st.session_state.inventory_target_year, target_inv_months_nums_sb
                        )
                        if not inventory_suggestions_df_tab2.empty:
                            st.dataframe(inventory_suggestions_df_tab2.style.format({
                                'Suggested Stock (Boxes)': '{:,.0f}', 'Sales Target (Boxes)': '{:,.0f}',
                                'Last Year Sales (Boxes)': '{:,.0f}', 'Historical Median (Boxes)': '{:,.0f}'
                            }), use_container_width=True)
                        else: st.info(f"No inventory suggestions for {selected_adv_city_tab2} for selected period. Check historical data and forecast availability.")
                    with st.expander("How are Inventory Suggestions Calculated?", expanded=False):
                        st.markdown(f"""
                        Inventory suggestions aim to balance stock availability with the risk of overstocking. Here's a simplified breakdown for **{selected_adv_city_tab2}**:
                        1.  **Historical Baseline (per target month):** Historical Median & Std Dev for that month.
                        2.  **Safety Stock:** `Historical Median + (Z-score * Historical Std Dev)`. (Z-score: {z_score_val} for {st.session_state.risk_level})
                        3.  **Prophet Forecast Integration:** If forecast (`yhat`) is available, `Base Stock` is scaled by `(Forecast / Median)` or a heuristic is used if median is zero.
                        4.  **Sales Target:** Higher of: `(Last Year's Sales * 1.1)` OR `Prophet Forecast`.
                        *Disclaimer: Data-driven suggestions. Always apply business judgment.*
                        """)

                    st.markdown("---")
                    st.subheader(f"Actionable Insights for: {selected_adv_city_tab2}")
                    if selected_adv_city_tab2 in prophet_forecasts_full and not prophet_forecasts_full[selected_adv_city_tab2].empty and forecast_start_calc:
                        city_full_fc_data_insight_tab2 = prophet_forecasts_full[selected_adv_city_tab2]
                        insights_text_tab2 = ""
                        latest_trend_val_tab2 = city_full_fc_data_insight_tab2['trend'].iloc[-1]
                        insights_text_tab2 += f"- **Overall Trend:** De-seasonalized sales trend for **{selected_adv_city_tab2}** is ~**{latest_trend_val_tab2:,.0f}** units/month towards forecast end.\n"
                        if 'yearly' in city_full_fc_data_insight_tab2.columns:
                            yearly_data_insight_tab2 = city_full_fc_data_insight_tab2[['ds', 'yearly']].copy()
                            peak_season_effect_tab2 = yearly_data_insight_tab2['yearly'].max()
                            peak_ds_obj_tab2 = yearly_data_insight_tab2.loc[yearly_data_insight_tab2['yearly'].idxmax(), 'ds'] if pd.notna(peak_season_effect_tab2) else None
                            peak_month_str_tab2 = peak_ds_obj_tab2.strftime('%B') if peak_ds_obj_tab2 else "N/A"
                            
                            low_season_effect_tab2 = yearly_data_insight_tab2['yearly'].min()
                            low_ds_obj_tab2 = yearly_data_insight_tab2.loc[yearly_data_insight_tab2['yearly'].idxmin(), 'ds'] if pd.notna(low_season_effect_tab2) else None
                            low_month_str_tab2 = low_ds_obj_tab2.strftime('%B') if low_ds_obj_tab2 else "N/A"
                            insights_text_tab2 += (f"- **Seasonal Impact:** Peak around **{peak_month_str_tab2}** (+{peak_season_effect_tab2:,.0f} units). Low around **{low_month_str_tab2}** ({low_season_effect_tab2:,.0f} units) vs trend.\n")
                        
                        city_fc_comp_data_insight_tab2 = all_cities_forecast_components_df[
                            (all_cities_forecast_components_df['city'] == selected_adv_city_tab2) &
                            (all_cities_forecast_components_df['ds'] >= forecast_start_calc)
                        ].copy()
                        if not city_fc_comp_data_insight_tab2.empty:
                            first_fc_month_details_tab2 = city_fc_comp_data_insight_tab2.sort_values('ds').iloc[0]
                            fc_m_name_tab2, fc_y_tab2, fc_l_tab2, fc_u_tab2 = first_fc_month_details_tab2['ds'].strftime("%B %Y"), first_fc_month_details_tab2['yhat'], first_fc_month_details_tab2['yhat_lower'], first_fc_month_details_tab2['yhat_upper']
                            unc_range_tab2 = fc_u_tab2 - fc_l_tab2
                            unc_pct_tab2 = (unc_range_tab2 / fc_y_tab2 * 100) if fc_y_tab2 != 0 else 0
                            insights_text_tab2 += (f"- **Initial Forecast Confidence ({fc_m_name_tab2}):** Central: **{fc_y_tab2:,.0f}**. Range: **{fc_l_tab2:,.0f} to {fc_u_tab2:,.0f}**. Uncertainty: **{unc_range_tab2:,.0f} boxes** ({unc_pct_tab2:.1f}% of forecast).")
                        st.markdown(insights_text_tab2)
                    else: st.error(f"Could not generate insights for {selected_adv_city_tab2}. Ensure forecast data exists.")
            else:
                 st.info("No cities were successfully forecasted. Please check forecast settings or data quality.")

    with tab3: # Forecast Calculations
        # ... (Tab 3 content as in previous robust version, using basic_forecasts_master_df)
        st.header("🧮 Forecast Calculations & Methodology")
        st.markdown("Explore Prophet model details, compare with historical benchmarks, and analyze forecast outputs. "
                    "All data reflects cities selected for forecasting in the sidebar.")
        st.markdown("---")

        col_methodology, col_params = st.columns(2)
        with col_methodology:
            st.markdown("#### How the Prophet Forecast is Calculated")
            st.markdown("""
            Prophet decomposes time series into: Trend, Seasonality, Holidays & Error.
            It fits these components to historical sales for each city, then projects them.
            """)
        with col_params:
            st.markdown("#### Prophet Model Parameters (Defaults)")
            st.json({
                'growth': 'linear', 'changepoint_prior_scale': 0.05, 
                'seasonality_prior_scale': 10.0, 'yearly_seasonality': True,
                'weekly_seasonality': False, 'daily_seasonality': False, 'interval_width': 0.95
            })
        st.markdown("---")

        st.subheader("Historical Benchmarks vs. Prophet")
        st.markdown("Compare Prophet's performance potential against simpler historical methods for **successfully Prophet-forecasted cities**.")

        if not successfully_forecasted_cities or basic_forecasts_master_df.empty:
            st.info("No data for historical benchmarks. Ensure Prophet forecasts were successfully generated for some cities in the sidebar ('Forecast Target Cities').")
        else:
            st.markdown("##### Key Historical Metrics (All Forecasted Cities)")
            comp_df_display_tab3 = basic_forecasts_master_df.copy()
            metrics_to_show_monthly_tab3 = [
                'Simple Avg (Monthly)', '3-Month Moving Avg (Monthly)', 
                '6-Month Moving Avg (Monthly)', '12-Month Moving Avg (Monthly)',
                'Weighted Moving Avg (Monthly)', 'Last Year Same Month (LYSM)',
                'Latest Full Year Avg (Monthly)'
            ]
            other_metrics_to_show_tab3 = ['Annual Growth Rate (YoY %)']
            
            fmt_dict_display_tab3 = {'City': '{}'}
            for col_name in metrics_to_show_monthly_tab3:
                if col_name in comp_df_display_tab3.columns:
                    fmt_dict_display_tab3[col_name] = '{:,.0f}'
                    annual_col_tab3 = col_name.replace("(Monthly)", "(Annualized)").replace("LYSM", "LYSM (Annualized)")
                    if col_name in comp_df_display_tab3 and pd.api.types.is_numeric_dtype(comp_df_display_tab3[col_name]):
                        comp_df_display_tab3[annual_col_tab3] = comp_df_display_tab3[col_name] * 12
                        fmt_dict_display_tab3[annual_col_tab3] = '{:,.0f}'
            
            for col_name in other_metrics_to_show_tab3:
                 if col_name in comp_df_display_tab3.columns:
                    fmt_dict_display_tab3[col_name] = '{:.1f}%' if '%' in col_name else '{:,.0f}'

            ordered_cols_display_tab3 = ['City'] + \
                                   [m for m in metrics_to_show_monthly_tab3 if m in comp_df_display_tab3.columns] + \
                                   [m.replace("(Monthly)", "(Annualized)").replace("LYSM", "LYSM (Annualized)") for m in metrics_to_show_monthly_tab3 if m.replace("(Monthly)", "(Annualized)").replace("LYSM", "LYSM (Annualized)") in comp_df_display_tab3.columns] + \
                                   [o for o in other_metrics_to_show_tab3 if o in comp_df_display_tab3.columns]
            
            comp_df_display_final_tab3 = comp_df_display_tab3[[col for col in ordered_cols_display_tab3 if col in comp_df_display_tab3.columns]]
            
            st.dataframe(comp_df_display_final_tab3.style.format(fmt_dict_display_tab3, na_rep='N/A'),
                         use_container_width=True, height=min(400, 50 + len(comp_df_display_final_tab3) * 35))
            st.caption("LYSM is based on sales of the corresponding month in the year prior to the last historical data month. Annualized figures are monthly values * 12.")
            st.markdown("---")

            st.markdown("##### Detailed Historical Analysis (Focus City)")
            if st.session_state.tab3_focus_city not in successfully_forecasted_cities or st.session_state.tab3_focus_city is None:
                 st.session_state.tab3_focus_city = successfully_forecasted_cities[0] if successfully_forecasted_cities else None
            
            if st.session_state.tab3_focus_city:
                focus_city_selected_tab3 = st.selectbox(
                    "Select Focus City for Detailed Plots:",
                    options=successfully_forecasted_cities,
                    index=successfully_forecasted_cities.index(st.session_state.tab3_focus_city),
                    key="tab3_focus_city_selector_final_v2"
                )
                st.session_state.tab3_focus_city = focus_city_selected_tab3

                if focus_city_selected_tab3:
                    focus_city_basic_data_plot_tab3 = basic_forecasts_master_df[basic_forecasts_master_df['City'] == focus_city_selected_tab3].iloc[0]
                    plot_col1_tab3, plot_col2_tab3 = st.columns(2)
                    with plot_col1_tab3:
                        chart_data_annualized_plot_tab3 = []
                        for method_col_name in metrics_to_show_monthly_tab3: # Use the same list for consistency
                            if method_col_name in focus_city_basic_data_plot_tab3 and pd.notna(focus_city_basic_data_plot_tab3[method_col_name]):
                                short_name = method_col_name.replace(" (Monthly)", "").replace(" Moving Avg"," MA")
                                chart_data_annualized_plot_tab3.append({'Method': short_name, 'Annualized Value': focus_city_basic_data_plot_tab3[method_col_name] * 12})
                        if chart_data_annualized_plot_tab3:
                            df_chart_ann_plot_tab3 = pd.DataFrame(chart_data_annualized_plot_tab3)
                            fig_basic_ann_plot_tab3 = px.bar(df_chart_ann_plot_tab3, x='Method', y='Annualized Value',
                                                title=f"Annualized Historical Benchmarks: {focus_city_selected_tab3}",
                                                labels={'Annualized Value': 'Sales (Boxes)'}, text_auto='.2s', height=450)
                            fig_basic_ann_plot_tab3.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_basic_ann_plot_tab3, use_container_width=True)

                    with plot_col2_tab3:
                        seasonal_factors_data_plot_tab3 = []
                        for col, val in focus_city_basic_data_plot_tab3.items():
                            if 'Seasonal Factor' in col and pd.notna(val):
                                month_sf_plot_tab3 = col.replace("Seasonal Factor (", "").replace(")", "")
                                seasonal_factors_data_plot_tab3.append({'Month': month_sf_plot_tab3, 'Factor': val})
                        if seasonal_factors_data_plot_tab3:
                            df_seasonal_factors_plot_tab3 = pd.DataFrame(seasonal_factors_data_plot_tab3)
                            month_order_map_short_tab3 = {pd.Timestamp(f'2000-{MONTH_NAME_TO_NUM[name]}-01').strftime('%b'): i for i, name in enumerate(ALL_MONTH_NAMES)}
                            df_seasonal_factors_plot_tab3['MonthOrder'] = df_seasonal_factors_plot_tab3['Month'].apply(lambda x: month_order_map_short_tab3.get(x, -1))
                            df_seasonal_factors_plot_tab3 = df_seasonal_factors_plot_tab3[df_seasonal_factors_plot_tab3['MonthOrder'] != -1].sort_values('MonthOrder').drop(columns='MonthOrder')
                            fig_seasonal_plot_tab3 = px.bar(df_seasonal_factors_plot_tab3, x='Month', y='Factor',
                                                  title=f"Monthly Seasonal Factors: {focus_city_selected_tab3}",
                                                  labels={'Factor': 'Seasonal Factor (vs. Avg)'}, text_auto='.2f', height=450)
                            fig_seasonal_plot_tab3.add_hline(y=1.0, line_dash="dot", line_color="grey", annotation_text="Average")
                            st.plotly_chart(fig_seasonal_plot_tab3, use_container_width=True)
                        else: st.info(f"Seasonal factors not available or insufficient data for {focus_city_selected_tab3}.")

                    focus_city_hist_plot_df_final = df_original[df_original['city'] == focus_city_selected_tab3].copy()
                    if not focus_city_hist_plot_df_final.empty:
                        city_m_sales_plot_series_final = focus_city_hist_plot_df_final.set_index(pd.to_datetime(focus_city_hist_plot_df_final['ds'])).sort_index()['y'].resample('M').sum().fillna(0)
                        if not city_m_sales_plot_series_final.empty:
                            ma3_plot_final = city_m_sales_plot_series_final.rolling(3, min_periods=1).mean()
                            ma6_plot_final = city_m_sales_plot_series_final.rolling(6, min_periods=1).mean()
                            ma12_plot_final = city_m_sales_plot_series_final.rolling(12, min_periods=1).mean()
                            fig_ma_p_final = go.Figure()
                            fig_ma_p_final.add_trace(go.Scatter(x=city_m_sales_plot_series_final.index, y=city_m_sales_plot_series_final, mode='lines+markers', name='Monthly Sales', line=dict(width=2), marker=dict(size=5)))
                            if len(ma3_plot_final.dropna()) > 0: fig_ma_p_final.add_trace(go.Scatter(x=ma3_plot_final.index, y=ma3_plot_final, mode='lines', name='3-Month MA', line=dict(width=1.5)))
                            if len(ma6_plot_final.dropna()) > 0: fig_ma_p_final.add_trace(go.Scatter(x=ma6_plot_final.index, y=ma6_plot_final, mode='lines', name='6-Month MA', line=dict(width=1.5)))
                            if len(ma12_plot_final.dropna()) > 0: fig_ma_p_final.add_trace(go.Scatter(x=ma12_plot_final.index, y=ma12_plot_final, mode='lines', name='12-Month MA', line=dict(width=1.5)))
                            fig_ma_p_final.update_layout(title=f"Historical Sales & Moving Averages: {focus_city_selected_tab3}", xaxis_title='Date', yaxis_title='Sales (Boxes)', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), height=500)
                            st.plotly_chart(fig_ma_p_final, use_container_width=True)
            else:
                st.info("No cities were successfully forecasted. Detailed plots require at least one successful forecast.")
        st.markdown("---")

        st.subheader("Prophet Forecast Summary (Aggregated & Detailed)")
        if not all_cities_forecast_components_df.empty and forecast_start_calc and forecast_end_calc:
            prophet_summary_data_fc_period_tab3 = all_cities_forecast_components_df[
                (all_cities_forecast_components_df['ds'] >= forecast_start_calc) &
                (all_cities_forecast_components_df['ds'] <= forecast_end_calc)
            ]
            if not prophet_summary_data_fc_period_tab3.empty:
                forecast_summary_agg_df_tab3 = prophet_summary_data_fc_period_tab3.groupby('city')['yhat'].agg(
                    total_forecasted_sales='sum', average_monthly_forecast='mean', std_dev_monthly_forecast='std'
                ).reset_index()
                forecast_summary_agg_df_tab3.rename(columns={
                    'city': 'City', 'total_forecasted_sales': f'Total Sales ({forecast_period_name_global})',
                    'average_monthly_forecast': 'Avg Monthly Sales (Forecast)', 'std_dev_monthly_forecast': 'Std Dev of Monthly Sales (Forecast)'
                }, inplace=True)

                agg_metrics_col1_tab3, agg_metrics_col2_tab3 = st.columns(2)
                with agg_metrics_col1_tab3:
                    st.metric("Number of Cities Successfully Forecasted by Prophet", len(forecast_summary_agg_df_tab3))
                with agg_metrics_col2_tab3:
                    overall_avg_monthly_fc_val_tab3 = forecast_summary_agg_df_tab3['Avg Monthly Sales (Forecast)'].mean() if not forecast_summary_agg_df_tab3.empty else 0
                    st.metric("Overall Avg Monthly Forecast (Per City)", f"{overall_avg_monthly_fc_val_tab3:,.0f} boxes")
                st.metric("Total Forecast Period Length", f"{actual_forecast_months} months, ending {forecast_end_calc.strftime('%b %Y') if forecast_end_calc else 'N/A'}")
                
                st.markdown("##### Forecast Summary per City (Full Horizon)")
                st.dataframe(forecast_summary_agg_df_tab3.style.format({
                    f'Total Sales ({forecast_period_name_global})': '{:,.0f}',
                    'Avg Monthly Sales (Forecast)': '{:,.0f}',
                    'Std Dev of Monthly Sales (Forecast)': '{:,.1f}'
                }), use_container_width=True, height=min(400, 100 + len(forecast_summary_agg_df_tab3) * 35))

                cols_to_download_prophet_final = ['ds', 'city', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
                downloadable_df_prophet_final_tab3 = all_cities_forecast_components_df[all_cities_forecast_components_df['ds'] >= forecast_start_calc].copy()
                
                final_cols_for_download = cols_to_download_prophet_final[:] # Create a copy
                if 'yearly' in downloadable_df_prophet_final_tab3.columns:
                    final_cols_for_download.append('yearly')
                
                # Ensure only existing columns are selected
                final_cols_for_download = [col for col in final_cols_for_download if col in downloadable_df_prophet_final_tab3.columns]
                downloadable_df_prophet_final_tab3 = downloadable_df_prophet_final_tab3[final_cols_for_download]
                
                downloadable_df_prophet_final_tab3.loc[:, 'ds'] = pd.to_datetime(downloadable_df_prophet_final_tab3['ds']).dt.strftime('%Y-%m-%d')
                
                csv_prophet_details_final_tab3 = downloadable_df_prophet_final_tab3.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Detailed Prophet Forecast Data (Month-by-Month, All Forecasted Cities)",
                    data=csv_prophet_details_final_tab3,
                    file_name=f"prophet_forecast_details_{forecast_start_calc.strftime('%Y%m%d') if forecast_start_calc else 'nodate'}_{forecast_end_calc.strftime('%Y%m%d') if forecast_end_calc else 'nodate'}.csv",
                    mime='text/csv',
                    key='download_prophet_details_csv_final_v2'
                )
                st.caption("Provides month-by-month `yhat` (forecast), confidence intervals, trend, and seasonality (if applicable) for all successfully forecasted cities over the forecast horizon.")
            else:
                st.info("No Prophet forecast data within the defined forecast horizon to summarize.")
        elif actual_forecast_months <= 0: st.info("No Prophet forecast summary as forecast horizon is 0.")
        else: st.info("No Prophet forecast data available. Ensure cities are selected and Prophet forecast is run from the sidebar.")


    with tab4_viz: # Dynamic Visualizations Tab
        # ... (Tab 4 content as in previous robust version)
        st.header("📊 Dynamic Data Visualizations")
        st.markdown("These visualizations are based on the **Display Filters** selected in the sidebar (Cities, Fiscal Years, Months).")

        if df_filtered_for_display.empty:
            st.info("No data to display based on current filters. Please adjust filters in the sidebar or upload data.")
        else:
            st.subheader("Monthly Sales Trend (Filtered View)")
            monthly_trend_df = df_filtered_for_display.groupby('ds')['y'].sum().reset_index().sort_values('ds')
            if not monthly_trend_df.empty:
                fig_monthly_trend = px.line(monthly_trend_df, x='ds', y='y', title="Total Monthly Sales Trend", labels={'ds': 'Date', 'y': 'Total Sales (Boxes)'}, markers=True)
                st.plotly_chart(fig_monthly_trend, use_container_width=True)
            else: st.info("No monthly sales data for the current filter selection.")
            st.markdown("---")

            st.subheader("Sales by City (Filtered View)")
            sales_by_city_df = df_filtered_for_display.groupby('city')['y'].sum().reset_index().sort_values('y', ascending=False)
            if not sales_by_city_df.empty:
                fig_sales_city = px.bar(sales_by_city_df.head(20), x='city', y='y', title="Top 20 Cities by Sales", labels={'city': 'City', 'y': 'Total Sales (Boxes)'}, text_auto=True)
                st.plotly_chart(fig_sales_city, use_container_width=True)
            else: st.info("No sales data by city for the current filter selection.")
            st.markdown("---")

            st.subheader("Sales by Agency (Filtered View)")
            sales_by_agency_df = df_filtered_for_display.groupby('agency')['y'].sum().reset_index().sort_values('y', ascending=False)
            if not sales_by_agency_df.empty:
                fig_sales_agency = px.bar(sales_by_agency_df.head(20), x='agency', y='y', title="Top 20 Agencies by Sales", labels={'agency': 'Agency Name', 'y': 'Total Sales (Boxes)'}, text_auto=True)
                st.plotly_chart(fig_sales_agency, use_container_width=True)
            else: st.info("No sales data by agency for the current filter selection.")
            st.markdown("---")

            st.subheader("Sales Distribution by Month (Filtered View)")
            sales_by_month_df = df_filtered_for_display.groupby('month_name')['y'].sum().reset_index()
            if not sales_by_month_df.empty:
                sales_by_month_df['month_order'] = sales_by_month_df['month_name'].map(MONTH_SORTER)
                sales_by_month_df = sales_by_month_df.sort_values('month_order')
                fig_dist_month = px.bar(sales_by_month_df, x='month_name', y='y', title="Sales Distribution by Month", labels={'month_name': 'Month', 'y': 'Total Sales (Boxes)'}, text_auto=True)
                fig_dist_month.update_xaxes(categoryorder='array', categoryarray=ALL_MONTH_NAMES)
                st.plotly_chart(fig_dist_month, use_container_width=True)
                fig_pie_month = px.pie(sales_by_month_df, names='month_name', values='y', title="Sales Proportion by Month", labels={'month_name': 'Month', 'y': 'Total Sales (Boxes)'})
                st.plotly_chart(fig_pie_month, use_container_width=True)
            else: st.info("No sales distribution data by month for the current filter selection.")
            st.markdown("---")

            st.subheader("Year-over-Year Monthly Sales Comparison (Filtered View)")
            yoy_df = df_filtered_for_display.groupby(['calendar_year', 'month_name'])['y'].sum().reset_index()
            selected_cal_years_plot_tab4 = sorted(yoy_df['calendar_year'].unique())
            if len(selected_cal_years_plot_tab4) > 1:
                yoy_df['month_order'] = yoy_df['month_name'].map(MONTH_SORTER)
                yoy_df = yoy_df.sort_values(['calendar_year', 'month_order'])
                fig_yoy = px.line(yoy_df, x='month_name', y='y', color='calendar_year', title="Year-over-Year Monthly Sales", markers=True, labels={'month_name': 'Month', 'y': 'Total Sales (Boxes)', 'calendar_year': 'Year'})
                fig_yoy.update_xaxes(categoryorder='array', categoryarray=ALL_MONTH_NAMES)
                st.plotly_chart(fig_yoy, use_container_width=True)
            elif len(selected_cal_years_plot_tab4) == 1: st.info("Only one year of data in filtered view. Select multiple years for YoY comparison.")
            else: st.info("No data or insufficient years selected for Year-over-Year comparison.")


elif df_original is None and uploaded_file is not None :
    pass # Error message already shown by load_data
else:
    st.info("👈 Please upload your sales data Excel file from the sidebar to start.")