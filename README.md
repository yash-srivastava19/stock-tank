## Every Second Counts
# Stock Price Prediction MLOps Pipeline

This project implements a complete MLOps pipeline for stock price prediction using LSTM neural networks. It includes data collection, preprocessing, model training, automated retraining, performance monitoring, and data drift detection.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stock-prediction-mlops.git
   cd stock-prediction-mlops
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start the MLOps pipeline for a specific stock symbol:

```
python src/main_workflow.py AAPL
```

Replace `AAPL` with the stock symbol you want to predict.

This will:
- Perform initial training if no model exists
- Schedule daily tasks (performance monitoring and data drift detection)
- Schedule weekly tasks (model retraining)

## Components

- `src/data/`: Scripts for data collection and preprocessing
- `src/models/`: Scripts for model training, evaluation, and versioning
- `src/monitoring/`: Scripts for performance monitoring and data drift detection
- `src/utils/`: Utility scripts for logging and alerting
- `src/api/`: Streamlit app for serving predictions

## Running Individual Components

You can run individual components of the pipeline using their respective CLI interfaces:

1. Data Collection:
   ```
   python src/data/data_collection.py AAPL --days 365
   ```

2. Data Preprocessing:
   ```
   python src/data/data_preprocessing.py AAPL
   ```

3. Model Training:
   ```
   python src/models/train_model.py AAPL --seq_length 10
   ```

4. Model Evaluation:
   ```
   python src/models/evaluate_model.py AAPL --seq_length 10
   ```

5. Performance Monitoring:
   ```
   python src/monitoring/performance_monitor.py AAPL --days 30
   ```

6. Data Drift Detection:
   ```
   python src/monitoring/data_drift.py AAPL --reference_days 365 --current_days 30
   ```

7. Streamlit App:
   ```
   streamlit run src/api/app.py
   ```

## Logs and Monitoring

- Logs are stored in the `logs/` directory
- Model versions are stored in `models/versions/`
- Monitoring results are stored in `monitoring/`

## Customization
You can customize various aspects of the pipeline by modifying the respective scripts. For example, you can adjust the retraining frequency, monitoring thresholds, or model architecture.


## License
This project is licensed under the MIT License.