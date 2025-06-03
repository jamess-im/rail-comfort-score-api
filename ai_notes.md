# AI Notes - Train Comfort Predictor API Project

## Project Overview
- **Goal:** Develop ML model and API to predict train carriage comfort levels
- **Input:** Departure station, destination, date/time
- **Output:** Comfort tier predictions (Quiet/Moderate/Busy) for Standard/First Class
- **Tech Stack:** Python, XGBoost, FastAPI, SQLite, Docker, Google Cloud Run
- **Working Directory:** `/Users/jamessimpson/workspace/uk-rail-comfort-score`

## Progress Log

### Completed: Development Environment Setup (Tasks 3.1-3.7)
- ✅ **Task 3.1:** Python 3.13.1 installed (meets 3.9+ requirement)
- ✅ **Task 3.2:** Git repository initialized 
- ✅ **Task 3.3:** Virtual environment created and activated (.venv)
- ✅ **Task 3.4:** All required packages installed via requirements.txt
- ✅ **Task 3.5:** DuckDB CLI v1.2.2 available
- ✅ **Task 3.6:** IDE setup - assumes VS Code or equivalent available
- ✅ **Task 3.7:** Sample DuckDB database created at `data/train_journey_legs.db` with 10,000 records

## Packages Installed
- pandas>=2.0.0, jupyterlab>=4.0.0, notebook>=7.0.0
- duckdb>=0.9.0, xgboost>=2.0.0, scikit-learn>=1.3.0
- fastapi>=0.100.0, uvicorn[standard]>=0.23.0
- python-dotenv>=1.0.0, joblib>=1.3.0

## Project Structure Created
- `/data/` - Contains train_journey_legs.db (sample data)
- `/src/` - Source code (create_sample_data.py)
- `/notebooks/` - For Jupyter notebooks
- `/models/` - For trained ML models
- `/api/` - For FastAPI application
- `/tests/` - For test files

## Sample Database Details
- **File:** `data/train_journey_legs.db`
- **Records:** 10,000 train journey legs
- **Schema:** Based on provided image schema
- **Stations:** 15 major UK stations
- **Time Range:** Full year 2024
- **Coach Types:** Standard, First Class, Mixed

### Completed: Data Understanding & Preparation (Tasks 4.1-4.2)
- ✅ **Task 4.1:** Loaded train_journey_legs into Pandas DataFrame (10,000 records, 24 columns)
- ✅ **Task 4.2:** Completed Exploratory Data Analysis (EDA)

## EDA Key Findings
- **Data Quality:** No missing values, no duplicates, clean dataset
- **Target Variable:** relevant_passengers_on_leg_departure (9-202 range, mean: 57.6)
- **Coach Types:** Balanced distribution (First Class: 33.8%, Standard: 33.3%, Mixed: 32.9%)
- **Stations:** 15 major UK stations, evenly distributed
- **Time Patterns:** Full year 2024, even distribution across days/hours
- **Correlations:** Strong correlation with vehicle passenger counts (0.9+)
- **Outliers:** 2.5% outliers in target variable

### Completed: Feature Engineering & Target Definition (Tasks 4.3-4.4)
- ✅ **Task 4.3:** Feature engineering completed (29 new features created)
  - Time features: hour_of_day, day_of_week, month, is_weekend, time_period, is_peak_hour
  - Location features: lat/lon coordinates, route distance, London/major city indicators
  - Occupancy features: occupancy percentages, passenger flow ratios, capacity utilization
  - Categorical encoding: coach_type, stations, headcode, rsid, time_period
- ✅ **Task 4.4:** Target variable definition completed
  - Target: comfort_tier (Quiet/Moderate/Busy) based on occupancy percentage
  - Thresholds: Quiet ≤ 53.7%, Moderate 53.7%-74.5%, Busy > 74.5%
  - Balanced classes: 33.0% Quiet, 34.0% Moderate, 33.0% Busy
  - Encoded as: Busy=0, Moderate=1, Quiet=2

### Completed: Feature Selection (Task 4.5)
- ✅ **Task 4.5:** Feature selection for model training completed
  - Selected features across 6 categories: Time (6), Location (10), Vehicle (2), Service (2), Contextual (9), Occupancy (7)
  - Total features for training: 36 features
  - Feature matrix ready: X (10,000 × 36), y (10,000,)
  - No missing values in selected features
  - Ready for XGBoost model training

### Completed: XGBoost Model Training & Evaluation (Tasks 5.1-5.6)
- ✅ **Task 5.1-5.6:** XGBoost training and evaluation completed
  - Issue resolved: Installed OpenMP runtime (libomp) via Homebrew for macOS
  - Model files generated: xgboost_comfort_classifier.joblib, feature_encoders.joblib, target_encoder.joblib, feature_list.joblib, feature_scaler.joblib, model_metadata.joblib
  - Evaluation plots: model_evaluation.png, comfort_tier_analysis.png, occupancy_distribution.png
  - Model successfully loads and makes predictions in API

### Completed: API Data Preparation (Tasks 6.1-6.2)
- ✅ **Task 6.1:** Identified API required data from MVP dataset
  - Station information: 15 stations with coordinates
  - Service routes: 9,965 service route patterns
  - Historical averages: 234 patterns for arrival state estimation
  - Coach types: 3 types (Standard, First Class, Mixed)
- ✅ **Task 6.2:** Created SQLite database for API lookups
  - Database: `api/train_comfort_api_lookups.sqlite`
  - Tables: stations, service_routes_summary_mvp, historical_averages, coach_info, metadata
  - Indexes created for efficient lookups
  - Validation successful with test queries

### Completed: API Development (Tasks 7.1-7.4)
- ✅ **Task 7.1:** API project structure created
- ✅ **Task 7.2:** Model loading at startup implemented
- ✅ **Task 7.3:** API endpoint `/predict_comfort_first_leg` implemented
- ✅ **Task 7.4:** Error handling and logging implemented

## API Features Implemented
- **FastAPI Application:** Complete REST API with OpenAPI documentation
- **Endpoints:**
  - `POST /predict_comfort_first_leg` - Main prediction endpoint
  - `GET /health` - Health check
  - `GET /stations` - List available stations
- **Features:**
  - Service identification from historical patterns
  - Historical averages lookup for arrival state estimation
  - Feature vector construction for both Standard and First Class
  - XGBoost model predictions with confidence scores
  - Comprehensive error handling

## Dataset Status
- **Final shape:** 10,000 records × 56 features
- **Target variable:** comfort_tier_encoded (3 balanced classes)
- **Features ready:** Time, location, occupancy, categorical (all encoded)
- **API Database:** SQLite with 5 tables, fully indexed and validated

### Completed: Containerization & Testing (Tasks 8.1-8.2)
- ✅ **Task 8.1:** Dockerfile created with all necessary components
  - Includes API code, model files, database, and proper dependencies
  - Multi-stage build with Python 3.11-slim base image
  - Health check endpoint configured
- ✅ **Task 8.2:** Docker image built and tested locally
  - Image builds successfully: train-comfort-api
  - Container runs and API responds correctly
  - API endpoints tested: /health, /stations, /predict_comfort_first_leg
  - Fixed Pydantic validation issue in response model

## Current Status
- **Completed:** Tasks 3.1-3.7, 4.1-4.5, 5.1-5.6, 6.1-6.2, 7.1-7.4, 8.1-8.2
- **Remaining:** Tasks 8.3-8.6 (Google Cloud deployment)
- **Next:** Google Cloud setup and deployment (requires manual cloud configuration)

## Technical Notes
- **XGBoost Issue:** Required OpenMP runtime installation on macOS (`brew install libomp`)
- **Import Issues:** Fixed relative import paths in Python modules
- **Database Path:** Corrected DuckDB connection path from '../duck' to 'duck'
- **API Database Path:** Fixed SQLite database path from root to 'api/' subdirectory
- **API Structure:** Comprehensive FastAPI implementation with proper error handling and validation
- **Docker Build:** Successfully containerized with all dependencies and data files
- **Pydantic Fix:** Updated response model to use Dict[str, Any] for mixed types in predictions
