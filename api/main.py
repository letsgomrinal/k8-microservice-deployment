from fastapi import FastAPI, Request
import numpy as np

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    inputs = np.array(data.get("inputs"))
    return {"prediction": inputs.mean().tolist()}
