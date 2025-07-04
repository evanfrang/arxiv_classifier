from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#PREDICT_DIR = os.path.join(ROOT_DIR, "predict")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

model = joblib.load(MODEL_DIR + "/model.joblib")
label_encoder = joblib.load(MODEL_DIR + "/label_encoder.joblib")
vectorizer = joblib.load(MODEL_DIR + "/vectorizer.joblib")

app = FastAPI()

class AbstractInput(BaseModel):
    abstract: str

@app.post("/predict")
def predict_category(data: AbstractInput):
    X = [data.abstract]
    X_vect = vectorizer.transform(X)  

    pred_encoded = model.predict(X_vect)
    pred_label = label_encoder.inverse_transform(pred_encoded)[0]

    probs = model.predict_proba(X_vect)[0]  # shape: (n_classes,)
    
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_probs = probs[top3_indices]
    top3_labels = label_encoder.inverse_transform(top3_indices)
    
    result = [
        {"category": label, "probability": round(float(prob), 4)}
        for label, prob in zip(top3_labels, top3_probs)
    ]
    
    return {"top_3_predictions": result}