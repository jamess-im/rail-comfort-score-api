# Train Comfort Predictor API

A machine learning API that predicts train carriage comfort levels (Quiet, Moderate, Busy) for UK rail journeys using XGBoost and historical passenger data.

## Overview

This API provides comfort level predictions for both Standard and First Class carriages based on:
- Departure and destination stations
- Date and time of travel
- Historical passenger patterns
- Service characteristics

## Features

- **Real-time Predictions**: Get comfort predictions for specific journeys
- **Multi-class Support**: Separate predictions for Standard and First Class
- **Historical Data**: Based on comprehensive UK rail passenger data
- **RESTful API**: Easy integration with FastAPI and OpenAPI documentation
- **Cloud Ready**: Containerized for deployment on Google Cloud Run

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (for containerization)
- Google Cloud CLI (for deployment)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd uk-rail-comfort-score
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if not already done)
   ```bash
   python src/xgboost_model_training.py
   ```

5. **Run the API locally**
   ```bash
   cd api
   uvicorn main:app --reload --host 0.0.0.0 --port 8080
   ```

6. **Access the API**
   - API: http://localhost:8080
   - Documentation: http://localhost:8080/docs
   - Health Check: http://localhost:8080/health

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t train-comfort-api .
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up
   ```

### Google Cloud Run Deployment

1. **Set environment variables**
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export GOOGLE_CLOUD_REGION="europe-west1"
   ```

2. **Deploy using the script**
   ```bash
   ./deploy.sh
   ```

## API Usage

### Predict Comfort Level

**POST** `/predict_comfort_first_leg`

```json
{
  "from_station": "London Paddington",
  "to_station": "Reading",
  "departure_datetime": "2024-01-15T08:30:00"
}
```

**Response:**
```json
{
  "from_station": "London Paddington",
  "to_station": "Reading",
  "departure_datetime": "2024-01-15T08:30:00",
  "standard_class": {
    "comfort_tier": "Moderate",
    "confidence": 0.85
  },
  "first_class": {
    "comfort_tier": "Quiet",
    "confidence": 0.92
  },
  "service_info": {
    "headcode": "1A23",
    "rsid": "UK123456"
  }
}
```

### Available Stations

**GET** `/stations`

Returns a list of all supported stations with their coordinates.

### Health Check

**GET** `/health`

Returns API status and model information.

## Project Structure

```
uk-rail-comfort-score/
├── api/                          # FastAPI application
│   ├── main.py                   # Main API application
│   └── train_comfort_api_lookups.sqlite  # API lookup database
├── src/                          # Source code
│   ├── create_sample_data.py     # Data generation
│   ├── feature_selection.py      # Feature engineering
│   ├── xgboost_model_training.py # Model training
│   └── api_data_preparation.py   # API data preparation
├── models/                       # Trained models and artifacts
├── data/                         # Training data
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Test files
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Local development
├── deploy.sh                     # Cloud deployment script
└── requirements.txt              # Python dependencies
```

## Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: 36 engineered features including time, location, service, and occupancy data
- **Target**: 3-class comfort levels (Quiet, Moderate, Busy)
- **Training Data**: 10,000 historical journey records
- **Performance**: Balanced accuracy across all comfort tiers

### Feature Categories

1. **Time Features** (6): Hour, day of week, month, weekend indicator, peak hours
2. **Location Features** (10): Station coordinates, route distance, major city indicators
3. **Vehicle Features** (2): Coach type, vehicle capacity
4. **Service Features** (2): Service identifiers (headcode, RSID)
5. **Contextual Features** (9): Historical passenger counts and flows
6. **Occupancy Features** (7): Occupancy percentages and capacity utilization

## Development

### Training Pipeline

1. **Data Preparation**: Load and clean historical journey data
2. **Feature Engineering**: Create time, location, and occupancy features
3. **Model Training**: Train XGBoost classifier with hyperparameter tuning
4. **Model Evaluation**: Validate performance with cross-validation
5. **Model Persistence**: Save trained model and preprocessing artifacts

### API Pipeline

1. **Service Identification**: Match user request to historical service patterns
2. **Historical Lookup**: Retrieve average passenger flows for similar journeys
3. **Feature Construction**: Build feature vectors for both coach types
4. **Prediction**: Generate comfort predictions with confidence scores
5. **Response Formatting**: Return structured JSON response

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on the repository or contact the development team. 