# Smart Traffic Congestion Prediction

A Flask + XGBoost starter application for estimating traffic speed, congestion category, and trip time.

## Setup
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install packages: `pip install -r requirements.txt`
5. Train the demo model: `python train.py`
6. Start the app: `python app.py`
7. Visit http://127.0.0.1:5000

## Important
`train.py` generates synthetic data to demonstrate the end-to-end workflow. It is not a real traffic dataset and its metrics must not be presented as real-world performance. For a final-year study, replace it with documented traffic observations, use time-based train/test splits, and compare against simple baselines.

The initial travel-time output is distance divided by predicted speed; it is an estimate, not a route optimizer. Accident status is supplied as an input and is not independently detected by this version.
