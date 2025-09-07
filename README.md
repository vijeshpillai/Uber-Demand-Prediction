# Project Overview: Uber Demand Prediction System

## What is the Project About?
This project focuses on predicting Uber ride demand in New York City using a dataset of Yellow Taxi trips from January to March 2016. The dataset, sourced from the [NYC Taxi & Limousine Commission](https://www.kaggle.com/datasets/elemento/nyc-yellow-taxi-trip-data), includes trip details such as pickup and drop-off locations, timestamps, distances, fares, and passenger counts.

"Demand" is defined as the number of ride bookings at a specific location and time, determined by combining latitude-longitude coordinates with timestamps. For example, at JFK Airport around 8 PM, demand spikes due to arriving travelers. The system predicts how many cabs will be needed at a given location and time (e.g., the next 15-minute interval), enabling drivers to move to high-demand areas proactively.

The goal is to build a regression model that takes time and location as inputs and outputs the expected number of pickups, helping drivers optimize their positioning and reduce idle time.

## Business Understanding
Uber operates a platform model with "driver-partners" who own their vehicles and choose when to work, unlike traditional employees. Uber earns a commission (e.g., 20–30%) per ride, so its revenue is tied to driver earnings. The demand prediction system provides drivers with a tool (e.g., a smartphone app) to identify high-demand areas for the next time interval (e.g., 15 minutes), based on their current location and time.

### Why It Matters
- **For Drivers**: Predicting demand for the next 15 minutes helps drivers move to high-demand areas (e.g., from 150 predicted rides to 250 in a nearby region), reducing idle time and increasing earnings.
- **For Customers**: Better driver distribution reduces wait times, cancellations, and surge pricing, improving reliability and affordability.
- **For Uber**: Higher driver earnings increase platform revenue. Enhanced driver and customer trust boosts loyalty, making Uber more competitive against rivals like Ola or Indrive. The system also positions Uber as an innovative, driver-friendly platform.

The focus on short-term (15-minute) predictions ensures actionable, fuel-efficient decisions, avoiding long trips that waste time or fuel.

## About the Data
The dataset is the NYC Yellow Taxi Trip dataset (Jan–Mar 2016), containing detailed trip records for Yellow Medallion Taxicabs. Key features include:
- **VendorID**: Taxi provider code.
- **Pickup/Drop-off Datetime**: Trip start/end timestamps.
- **Passenger_count**: Number of passengers.
- **Trip_distance**: Distance traveled.
- **Pickup/Drop-off Coordinates**: Latitude and longitude.
- **RateCodeID**: Fare type (e.g., standard, airport).
- **Payment_type**: Cash, card, etc.
- **Fare Details**: Fare, tips, tolls, taxes, total amount.

These features provide the spatial (coordinates) and temporal (timestamps) data needed for demand prediction.

## Project Goal
The objective is to frame demand prediction as a regression problem, predicting the number of pickups (a continuous value) for a given time and location. To manage millions of unique coordinates, New York City is divided into regions using clustering, with each region small enough for drivers to travel to within 15 minutes. The model accounts for temporal patterns, such as higher demand in office districts during weekday rush hours or in entertainment areas on weekends.

### Tasks
- **Spatial Forecasting**: Predict demand for each region.
- **Temporal Forecasting**: Predict demand for the next 15-minute interval.

## How to Achieve
### Spatial Analysis
To handle millions of unique latitude-longitude pairs, unsupervised clustering (e.g., Mini-Batch K-Means) groups pickup points into regions. Each region is sized so drivers can travel to it within 10–15 minutes (approximately 1–1.5 miles in NYC).

### Temporal Analysis
Demand is treated as a time series, with data aggregated into 15-minute intervals. Historical trip data is used to forecast pickups for the next interval (e.g., 8:15 PM based on data up to 8:00 PM).

### Model and Evaluation
A regression model is trained on region and time inputs to predict pickups. The evaluation metric is **Mean Absolute Percentage Error (MAPE)**, which penalizes large percentage errors, critical for low-demand regions. For example:
- Predicting 105 vs. actual 100 (5% error) is manageable.
- Predicting 15 vs. actual 10 (50% error) is significant.

MAPE ensures predictions are meaningful across regions with varying demand levels.

## Expected Flow
The machine learning pipeline includes:
1. **Exploratory Data Analysis (EDA)**: Use Dask to process the 6 GB dataset in chunks, identifying patterns and outliers.
2. **Feature Selection**: Focus on pickup time, location, and trip-related features.
3. **Clustering for Regions**: Apply clustering to group pickup points into actionable regions.
4. **Time Series Analysis**: Aggregate data into 15-minute intervals to capture temporal patterns.
5. **Data Preparation**: Engineer features (e.g., lagged demand) and structure data for spatial-temporal forecasting.
6. **Model Training**: Train regression models to predict pickups.
7. **Evaluation**: Use MAPE on a time-based train-test split (Jan–Feb for training, Mar for testing).
8. **Hyperparameter Tuning**: Optimize models using Optuna.
9. **Visualization**: Plot demand predictions on a NYC map for driver use.

## Exploratory Data Analysis (EDA)
### Why Use Dask?
The 6 GB dataset exceeds typical memory limits. Dask, a Python library for large-scale data, processes data in chunks and parallelizes computations across CPU cores. It mimics Pandas and NumPy APIs, enabling efficient analysis without loading the entire dataset into memory.

### How Dask Works
Dask uses lazy-loading and task graphs, storing metadata (e.g., column types) instead of the full dataset. Operations like filtering or grouping are executed in chunks, leveraging parallel processing to handle large data efficiently.

### Things to Avoid
- Excessive use of `.compute()`, which consumes CPU resources.
- Rely on Pandas when possible for smaller subsets to avoid overhead.

### EDA Results
- **Datetime Importance**: Pickup counts vary by time of day and day of week, making timestamps critical.
- **Outliers**: Boxplots revealed unrealistic coordinates, trip distances, and fares, which were cleaned to prevent model distortion.

## Breaking New York City into Regions
### Goal
Convert millions of pickup coordinates into manageable regions using clustering (e.g., Mini-Batch K-Means). Each region is sized for drivers to reach within 10–15 minutes (1–1.5 miles).

### Why Clustering Over Fixed Grids?
Fixed grids create uneven regions, with some overloaded (e.g., downtown) and others sparse. Clustering adapts to demand density, forming tighter clusters in high-demand areas (e.g., airports) and broader ones in low-demand areas.

### Considerations
- **Outliers**: Cleaned extreme coordinates to avoid distorted clusters.
- **Driver Mobility**: Regions are sized based on NYC travel times (10–12 minutes per mile), ensuring drivers can act on predictions within 15 minutes.

### Business Logic Meets Technical Logic
Clustering balances technical efficiency (grouping similar coordinates) and business practicality (ensuring regions are reachable). Regions are defined so drivers can move to neighboring areas within the prediction window, making forecasts actionable.

### Plan of Attack
1. **Data Cleaning**: Remove outliers in coordinates, fares, and distances using Dask.
2. **Clustering**: Apply Mini-Batch K-Means on pickup coordinates, incrementally training with `partial_fit` on chunks.
3. **Cluster Evaluation**: Test different K values (e.g., 10, 20, 30, 40), calculate distances between centroids, and ensure neighboring regions are 1–1.5 miles apart for driver feasibility.
4. **Region Assignment**: Map pickup points to clusters, creating a region-based dataset.

## Creating Historical Data
### Goal
Transform raw trip data into a spatio-temporal dataset by aggregating pickups into 15-minute intervals per region. For example, Region 12 at 8:00–8:15 PM with 143 pickups becomes one record.

### Why Smoothing Is Required
Raw pickup counts are volatile, complicating pattern detection. Smoothing (e.g., moving averages, EWMA) reduces noise, revealing trends. For example, raw counts [12, 45, 10, 50, 15] may smooth to [22, 22, 35, 25], improving model performance.

#### Moving Average Analysis
A moving average (e.g., 3-day window) calculates average pickups, sliding forward to smooth data. Larger windows increase smoothness but may obscure short-term trends.

#### Exponential Weighted Moving Average (EWMA)
EWMA assigns higher weights to recent data via alpha (α):
- High α (e.g., 0.8): Less smoothing, responsive to changes (e.g., [10, 20, 40] → [10, 18, 35.6]).
- Low α (e.g., 0.2): More smoothing, stable trends (e.g., [10, 12, 17.6]).

Alpha balances responsiveness and smoothness, critical for accurate forecasting.

## Building a Baseline Model
### Flow
The dataset is structured with:
- Datetime
- Region (cluster ID)
- Average/total pickups per interval

Lagged features (e.g., pickups at T₋₁, T₋₂, T₋₃, T₋₄) capture historical demand patterns. A time-based train-test split (Jan–Feb for training, Mar for testing) preserves temporal order.

### Model Selection and Hyperparameter Tuning
- **Framework**: Optuna with Bayesian optimization for efficient model and hyperparameter selection.
- **Metric**: MAPE to penalize large percentage errors.
- **Tracking**: MLflow with Dagshub logs trials, metrics, and models for reproducibility.

## Building the DVC Pipeline
### Part 1: Data Preparation and Clustering
1. Load and concatenate Jan–Mar datasets using Dask (~33M rows post-cleaning).
2. Remove outliers in coordinates, fares, and distances.
3. Extract pickup coordinates for clustering.
4. Scale coordinates incrementally with `partial_fit`.
5. Apply Mini-Batch K-Means to form regions, converting centroids back to original coordinates.
6. Assign regions to the dataset.

### Part 2: Feature Engineering and Modeling
1. Resample data into 15-minute intervals per region.
2. Calculate total pickups per interval.
3. Apply EWMA for smoothing.
4. Add lag features (T₋₁ to T₋₄).
5. Split data: Jan–Feb (train), Mar (test).
6. Train a Linear Regression model and evaluate with MAPE.
7. Track pipeline with DVC and log with MLflow.

## Building the Streamlit Application
The Linear Regression model is deployed in a Streamlit app for driver use:
- **Inputs**: Location (latitude/longitude or region ID), time interval.
- **Output**: Predicted pickups for the region and neighbors, visualized on a NYC map with:
  - Colored regions indicating demand.
  - A legend with region IDs and predicted demand.
- **Purpose**: Helps drivers identify high-demand areas for the next 15 minutes, reducing idle time and increasing earnings.