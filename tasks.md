# Project: Train Comfort Predictor API

## 1. Project Summary & Goals

- **Goal:** Develop a machine learning model and API to predict the "comfort level" (e.g., Quiet, Moderate, Busy) on specific train carriages for given journey segments, based on a fixed historical dataset for the MVP.
- **Purpose:** To provide passengers with an indication of how crowded a train service is likely to be, allowing them to make more informed travel choices.
- **Input (User via API):** Departure station, destination station, desired date, and time of travel.
- **Output (API to User):** Predicted comfort tier (e.g., "Quiet", "Moderate", "Busy") for Standard and First Class carriages for the requested journey (initially for the first leg, with potential for multi-leg in future iterations).
- **Core Logic:** The prediction will be based on a fixed historical train occupancy dataset (from the `train_journey_legs` table), leveraging patterns related to time of day, day of week, station, service characteristics, and historical passenger counts on arrival at the departure station of a leg.

## 2. Tech Stack

- **Programming Language:** Python (v3.9+)
- **Data Processing & Analysis (Primarily for model training data prep):**
  - Pandas: For data manipulation in Python.
  - DuckDB: For accessing the pre-existing `train_journey_legs` and other source tables (data is fixed for MVP).
- **Machine Learning:**
  - XGBoost: For the core classification model.
  - Scikit-learn: For preprocessing (e.g., `LabelEncoder`), model evaluation metrics, and potentially `train_test_split`.
- **API Framework:** FastAPI
- **Data Storage for API Lookups:** SQLite (for storing pre-aggregated historical stats and lookup tables needed by the live API, generated once from the fixed DuckDB dataset for the MVP).
- **Environment Management:** `venv`.
- **Containerization:** Docker.
- **Cloud Deployment:** Google Cloud Run.
- **Version Control:** Git (e.g., GitHub, GitLab).

## 3. Development Environment Setup

Note that the pwd of the working folder (where this file is located) is `/Users/jamessimpson/workspace/uk-rail-comfort-score`.

- [x] **Task 3.1:** Install Python (if not already installed, ensure version 3.9+).
- [x] **Task 3.2:** Set up a project directory and initialize a Git repository.
- [x] **Task 3.3:** Create a Python virtual environment (e.g., `python -m venv .venv` and `source .venv/bin/activate` or `conda create -n train_comfort python=3.9 && conda activate train_comfort`).
- [x] **Task 3.4:** Install necessary Python libraries:
  - `pip install pandas jupyterlab notebook duckdb xgboost scikit-learn fastapi uvicorn[standard] python-dotenv joblib sqlite3` (or add to a `requirements.txt` file and `pip install -r requirements.txt`).
- [x] **Task 3.5:** Ensure DuckDB CLI is available if direct DB inspection is desired.
- [x] **Task 3.6:** Set up your preferred IDE (e.g., VS Code with Python extension).
- [x] **Task 3.7:** Obtain connection details/path for the existing DuckDB database containing the fixed MVP dataset (`train_journey_legs` and other source tables).

## 4. Data Understanding & Preparation for Model Training (Python with Pandas & DuckDB)

- **Input:** Pre-existing DuckDB table `train_journey_legs` (part of the fixed MVP dataset).
- [x] **Task 4.1: Load `train_journey_legs` into Pandas DataFrame:**
  - Write a Python script to connect to the DuckDB (containing the MVP dataset) and fetch data from `train_journey_legs` into a Pandas DataFrame.
- [x] **Task 4.2: Exploratory Data Analysis (EDA) on `train_journey_legs` (Jupyter Notebook recommended):**
  - Verify structure, content, distributions, missing values, outliers.
- [x] **Task 4.3: Feature Engineering (in Pandas):**
  - **Time Features:** From `leg_departure_dt`, extract `hour_of_day`, `day_of_week`, `month`, `is_weekend`, etc.
  - **Location Features:** Parse string coordinates into numerical `latitude` and `longitude`.
  - **Categorical Feature Encoding Strategy:** Plan for encoding.
- [x] **Task 4.4: Define Target Variable (`comfort_tier`):**
  - Identify `relevant_passengers_on_leg_departure` column.
  - Calculate `occupancy_percentage_leg`.
  - Map to comfort tiers ("Quiet", "Moderate", "Busy").
  - LabelEncode `comfort_tier`. This is your `y` variable.
- [x] **Task 4.5: Select Features for Model Training (`X` variable):**
  - Include: Time features, `stationName_from`, `stationName_to` (encoded), location features, `coach_type` (encoded), `vehicle_capacity`, service identifiers (`headcode`, `rsid` - encoded), and key contextual features like `vehicle_pax_on_arrival_..._at_from`, `totalUnitPassenger_at_leg_departure`, `onUnitPassenger_at_from_station`, `offUnitPassenger_at_from_station`.

## 5. XGBoost Model Training & Evaluation

- [x] **Task 5.1: Data Splitting:**
  - Split into training/testing sets (e.g., 80/20, `stratify=y`).
- [x] **Task 5.2: Preprocessing Pipeline (Consider).**
- [x] **Task 5.3: Train XGBoost Classifier (`XGBClassifier`):**
  - Initialize, train, use `eval_set` and `early_stopping_rounds`.
- [x] **Task 5.4: Model Evaluation:**
  - Use accuracy, classification report, confusion matrix on the test set.
- [x] **Task 5.5: Hyperparameter Tuning (Iterative Improvement).**
- [x] **Task 5.6: Save Trained Model & Supporting Files:**
  - Use `joblib` for the model, target encoder, feature encoders, and feature list.

## 6. API Data Preparation (SQLite for API Lookups - Generated Once for MVP)

- **Goal:** Create a new, lightweight SQLite database (e.g., `train_comfort_api_lookups.sqlite`) containing pre-aggregated data and lookups derived _solely from the fixed MVP DuckDB dataset_. This SQLite file will be bundled with the API.
- [x] **Task 6.1: Identify Data Required by the API at Prediction Time (from the fixed MVP dataset):**
  - **Station Information:** Mapping `stationName` to `latitude`, `longitude`.
  - **Service Identification & Routing Summaries:** A way to:
    - Match a user's request (`from_station`, `datetime_api`, potentially `to_station_api`) to historical `headcode`/`rsid` patterns _present in the MVP dataset_.
    - Determine the _actual next stop_ for an identified service departing a given station, _based on routes in the MVP dataset_.
      This might be a `service_routes_summary_mvp` table.
  - **Historical Average Stats for Arrival State Estimation (from the fixed MVP dataset):** For a given `headcode`/`rsid` (or service pattern), `station_of_arrival`, `day_of_week_bucket`, `time_of_day_bucket` _as observed in the MVP dataset_:
    - `avg_vehicle_pax_on_arrival_std`
    - `avg_vehicle_pax_on_arrival_first`
    - `avg_total_unit_pax_on_arrival` (or at departure from previous stop)
    - `avg_unit_boarders_at_station`
    - `avg_unit_alighters_at_station`
- [x] **Task 6.2: Create SQLite Database and Populate Tables (One-Time Process for MVP):**
  - Write a Python script (can be run after model training or as a separate data pipeline step):
    - Connect to the main DuckDB (containing the fixed MVP `train_journey_legs` dataset).
    - Execute aggregation queries on `train_journey_legs` to calculate the averages and summaries identified in Task 6.1. These aggregations are based _only_ on the static MVP data.
    - Write these resulting Pandas DataFrames into new tables in the `train_comfort_api_lookups.sqlite` database.
  - Create necessary indexes on SQLite tables.

## 7. API Development (FastAPI/Flask)

- [x] **Task 7.1: Set up API Project Structure.**
- [x] **Task 7.2: Load Model and Supporting Files at API Startup:**
  - Load XGBoost model, encoders, feature list.
  - Establish a connection (or ensure access) to the bundled `train_comfort_api_lookups.sqlite` file.
- [x] **Task 7.3: Implement API Endpoint (e.g., `/predict_comfort_first_leg`):**
  - **Input:** `from_station`, `to_station`, `departure_datetime`.
  - **Logic:**
    1. Parse & Validate API Inputs.
    2. **Identify Relevant Service & Actual Next Stop (from bundled SQLite):**
        - Query `train_comfort_api_lookups.sqlite` (e.g., `service_routes_summary_mvp` table) to find `headcode`/`rsid` patterns and their next stops matching the user's request, using "nearest time" logic against the _historical schedule patterns in the MVP data_.
    3. **Fetch Historical Averages for Arrival State (from bundled SQLite):**
        - For the identified service and `from_station` at the relevant time bucket, query `train_comfort_api_lookups.sqlite` for the pre-calculated average arrival/flow stats.
    4. **Construct Feature Vector(s) for XGBoost Model** for Standard and First Class.
    5. **Make Predictions** using `model.predict_proba()`.
    6. **Format & Return JSON Output.**
- [x] **Task 7.4: Implement Basic Error Handling and Logging.**

## 8. Containerization & Deployment (Docker, Google Cloud Run)

- [x] **Task 8.1: Create `Dockerfile`:**
  - Include API code, model files (`.joblib`), and crucially, the `train_comfort_api_lookups.sqlite` file.
- [x] **Task 8.2: Build and Test Docker Image Locally.**
- [ ] **Task 8.3: Set up Google Cloud Project & Enable APIs.**
- [ ] **Task 8.4: Push Docker Image to Google Artifact Registry.**
- [ ] **Task 8.5: Deploy to Google Cloud Run.**
- [ ] **Task 8.6: Test the Deployed API Endpoint.**

## 9. XGBoost Usage Summary (For this Leg-Based Model with Fixed MVP Data)

- **Type of Problem:** Multi-class classification.
- **Input Data for Training:** Each row from the fixed MVP `train_journey_legs` table.
- **Key Features for Training:** Time, leg origin/destination, service IDs, vehicle type/capacity, and _observed historical_ passenger counts on arrival at leg origin & unit flows (all from the fixed MVP dataset).
- **Target Variable for Training:** Comfort tier for that historical leg (from the fixed MVP dataset).
- **Prediction Time by API:**
  - API identifies a target service and its first leg using lookups against the bundled `train_comfort_api_lookups.sqlite` (which itself is derived from the fixed MVP dataset).
  - It constructs features using _estimated historical averages_ (also from the bundled SQLite) for arrival state and passenger flows.
  - XGBoost predicts comfort for this first leg.

## 10. Iteration & Future Improvements (Post-MVP)

- Integrate live/updated schedule data (potentially from the Go app's output) for service identification and routing in the API lookup database.
- Implement a data pipeline to regularly update `train_journey_legs` and re-calculate historical averages in `train_comfort_api_lookups.sqlite` from new data.
- Automate model re-training with new data.
- Implement multi-leg journey prediction in the API.
- More advanced feature engineering.
- Enhanced monitoring and logging.

---

The main changes emphasize that the data sources for both model training and the API's lookup SQLite are all derived from the **fixed MVP dataset** in DuckDB. This simplifies the data dependencies for the initial deployment.
