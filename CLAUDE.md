# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning API that predicts train carriage comfort levels (Quiet, Moderate, Busy) for UK rail journeys using XGBoost and historical passenger data. The system is containerized and ready for Google Cloud Run deployment.

## Development Commands

### Local Development
```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train the model (required before running API)
python src/xgboost_model_training.py

# Run API locally
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Run API on different port (if 8080 is occupied)
uvicorn main:app --reload --host 0.0.0.0 --port 8081
```

### Testing
```bash
# Test API endpoints
python test_api.py

# Test with custom API URL
python test_api.py http://localhost:8001
```

### Docker Development
```bash
# Build and run locally
docker build -t train-comfort-api .
docker-compose up

# Or run container directly
docker run -p 8080:8080 train-comfort-api
```

### Code Quality
```bash
# Format code
black src/ api/ --line-length 88
isort src/ api/ --profile black

# Lint code
flake8 src/ api/
```

## Architecture

### Core Components

1. **Data Pipeline** (`src/`):
   - `create_sample_data.py`: Generates synthetic training data
   - `feature_engineering.py`: Creates ML features from raw data
   - `feature_selection.py`: Selects optimal features for training
   - `xgboost_model_training.py`: Trains and saves the XGBoost model
   - `api_data_preparation.py`: Creates SQLite lookup database for API

2. **API Service** (`api/`):
   - `main.py`: FastAPI application with prediction endpoints
   - `train_comfort_api_lookups.sqlite`: Pre-built lookup database
   - Endpoints: `/predict_comfort_first_leg`, `/health`, `/stations`, `/tiplocs`

3. **Model Artifacts** (`models/`):
   - `xgboost_comfort_classifier.joblib`: Trained model
   - `feature_encoders.joblib`: Categorical encoders
   - `feature_list.joblib`: Expected feature names
   - `target_encoder.joblib`: Class label encoder
   - Supporting metadata and evaluation files

### Data Flow

1. **Training Pipeline**: Raw data → Feature engineering → Model training → Saved artifacts
2. **API Pipeline**: User request → Service lookup → Feature construction → Model prediction → JSON response
3. **Prediction Process**: TIPLOC resolution → Historical data lookup → Feature vector → XGBoost prediction

### Key Design Patterns

- **TIPLOC Integration**: Uses UK rail TIPLOC codes for precise station identification
- **Historical Averages**: Leverages passenger flow patterns for feature engineering
- **Batch Prediction**: Single endpoint handles multiple datetime predictions
- **Fallback Logic**: Multiple service matching strategies for robustness
- **Feature Consistency**: Maintains exact feature alignment between training and inference

## Important Notes

- **Model Training Required**: Run `python src/xgboost_model_training.py` before starting API
- **Database Dependency**: API requires `api/train_comfort_api_lookups.sqlite` to exist
- **TIPLOC Format**: API expects UK rail TIPLOC codes, not station names
- **Datetime Format**: Use ISO format strings (e.g., "2024-01-15T08:30:00")
- **Port Configuration**: Default port 8080 (Cloud Run standard), use 8081 if occupied

## Deployment

The project is containerized and configured for Google Cloud Run:
- Use `./deploy.sh` for automated GCP deployment
- Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_REGION` environment variables
- Docker image includes all dependencies and model artifacts
- Health checks and auto-scaling configured