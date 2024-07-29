# Stock-Tank: Stock Price Prediction MLOps Pipeline

This project implements a complete MLOps pipeline for stock price prediction using LSTM neural networks. It includes data collection, preprocessing, model training, automated retraining, performance monitoring, and data drift detection and deployment on Streamlit.


Check out the deployed version here : [Streamlit App](https://stock-tank.streamlit.app/)

On daily basis, the model performs monitoring routines.
On weekly basis, the models are retrained, plus the monitoring routines are performed as usual.
On monthly, the models are retrained and the monitoring routines are performed.

The retraining and monitoring actions are done using Github Actions. The workflows are available in the `.github/workflows` directory.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yash-srivastava19/stock-tank.git
   cd stock-tank
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r src/requirements.txt
   ```

## Usage

All the files required for a particular part of the pipeline are in their respective folder. Here's what the structure of the project looks like :

```
- .github/workflows
- data
- job_logs
- logs
   - main.log
   - monitoring.log
- model/versions
   - AAPL
   - GOOGL
   - MSFT
   - AMZN
   - TSLA
- monitoring
- plots
- src
   - api
   - data
   - models
   - monitoring
   - utils
   app.py
   jobs.py
   main.py
   requirements.txt
- tests
Dockerfile
README.md
```

## Components 

- `src/data/`: Scripts for data collection and preprocessing
- `src/models/`: Scripts for model training, evaluation, and versioning
- `src/monitoring/`: Scripts for performance monitoring and data drift detection
- `src/utils/`: Utility scripts for logging and alerting.


Every component of the pipeline can be tested with CLI interface. Suppose, you want to collect data for a a particular stock, just provide the ticker(AAPL, GOOGL, AMZN, TSLA, MSFT), and run the file

```bash
$ python src/data/data_collect.py AAPL
```

Replace `AAPL` with the stock symbol you want to collect data for. Similarly, you can test for other modules as well. Here's the overview :

1. Data Collection:
```bash
   $ python src/data/data_collect.py AAPL --days 365
```

2. Data Preprocessing:
```bash
   $ python src/data/data_preprocess.py AAPL
```

3. Model Training:
```bash
   $ python src/models/model_train.py AAPL --seq_length 10
```

4. Model Evaluation:
```bash
   $ python src/models/evaluate_model.py AAPL --seq_length 10
```

5. Performance Monitoring:
```bash
   $ python src/monitoring/monitor_performance.py AAPL --days 30
```

6. Data Drift Detection:
```bash
   $ python src/monitoring/data_drift.py AAPL --reference_days 365 --current_days 30
```

7. Streamlit App:
```bash
   $ streamlit run src/app.py
```

## Logs and Monitoring

- Logs are stored in the `logs/` directory
- Model versions are stored in `models/versions/`
- Monitoring results are stored in `monitoring/`

## Customization
You can customize various aspects of the pipeline by modifying the respective scripts. For example, you can adjust the retraining frequency, monitoring thresholds, or model architecture.


## License
This project is licensed under the MIT License.