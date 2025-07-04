from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("models/model.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, abstract: str = Form(...)):
    vec = vectorizer.transform([abstract])
    pred = model.predict(vec)
    #label = label_encoder.inverse_transform(pred)[0]
    probs = model.predict_proba(vec)[0]
    
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_probs = probs[top3_indices]
    top3_labels = label_encoder.inverse_transform(top3_indices)
    
    result = [
        {"category": label, "probability": round(float(prob), 4)}
        for label, prob in zip(top3_labels, top3_probs)
    ]
    return templates.TemplateResponse("form.html", {"request": request, "top 3 predictions": result, "abstract": abstract})