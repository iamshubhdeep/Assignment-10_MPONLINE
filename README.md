# Heart Disease Prediction — End-to-End ML Deployment

A machine learning model that predicts whether a patient is at risk of heart
disease based on clinical parameters, served as a REST API with Flask and
deployed on Render.

## Details

- Name : Shubhdeep Singh
- Application No. : IN26011804
- College Reg. No. : 23BCE11460
- Batch No. : 1(A)
- Email Id : shubhdeep.23bce11460@vitbhopal.ac.in

## Problem Statement

A healthcare organization wants to deploy a machine learning model that
predicts whether a patient is at risk of heart disease based on clinical
parameters (age, cholesterol, blood pressure, ECG results, etc.).

## Dataset

**Heart Disease Prediction Dataset** (Kaggle):
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

> **Note:** `heart.csv` in this repo should be replaced with the file
> downloaded directly from the Kaggle link above before final training,
> to ensure results match the official dataset.

Features: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal` — Target: `target` (1 = heart disease present).

## Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                 # Flask REST API
├── model.pkl              # Trained classification model (Joblib)
├── scaler.pkl             # Fitted StandardScaler
├── feature_names.pkl      # Ordered list of input features
├── requirements.txt
├── Procfile                # Start command for Render (gunicorn)
├── README.md
├── train_model.py          # Data preprocessing + model training
├── heart.csv                # Dataset
├── templates/               # (optional)
└── static/                  # (optional)
```

## Task 1 & 2: Data Preprocessing and Model Development

`train_model.py`:
1. Loads `heart.csv` with Pandas and displays the first five records.
2. Identifies numerical features and the target variable.
3. Checks for missing values.
4. Splits the data 80/20 into train/test sets.
5. Scales features with `StandardScaler`.
6. Trains a **Random Forest Classifier**.
7. Evaluates using accuracy score (~0.76 on the held-out test set).
8. Saves the model, scaler, and feature list with Joblib.

Run it:
```bash
python train_model.py
```

## Task 3: API Development

`app.py` exposes a Flask REST API:

- `GET /` — health check / usage info
- `POST /predict` — accepts patient details as JSON, returns a prediction

### Example Request
```bash
curl -X POST https://<your-render-url>/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
      }'
```

### Example Response
```json
{
  "prediction": "Heart Disease Detected",
  "probability": 0.81
}
```

Run locally:
```bash
pip install -r requirements.txt
python app.py
```

## Task 4: GitHub and Render Deployment

### GitHub
1. Create a **public** repository named `HeartDiseaseDeployment`.
2. Push all files listed in the repository structure above.

### Render
1. Log in to [Render](https://render.com) and create a **New Web Service**.
2. Connect your GitHub repository.
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Deploy and confirm the service is live and returns predictions.

**Deployed Application URL:** `<PASTE YOUR RENDER URL HERE>`

## Task 5: Conclusion

The Random Forest model achieved an accuracy of approximately 76% on the
held-out test set, showing that clinical parameters such as chest pain
type, exercise-induced angina, and ST depression are informative predictors
of heart disease risk. Performance could likely be improved further through
hyperparameter tuning, cross-validation, and testing alternative algorithms
such as Gradient Boosting or SVM.

The main challenges during deployment involved ensuring the exact feature
order and preprocessing (scaling) used during training was replicated at
inference time inside the Flask API, and configuring a production-ready
start command (Gunicorn) for Render rather than relying on Flask's
development server. Environment and dependency versions also needed to be
pinned in `requirements.txt` to avoid mismatches between the local and
deployed environments.

This project highlights why MLOps matters in real-world machine learning:
a model is only useful once it can be reliably packaged, version-controlled,
served through an API, and monitored in production. Practices like model
serialization, containerized or managed deployment, and reproducible
environments are what turn a notebook experiment into a dependable
healthcare-facing service.
