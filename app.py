from fastapi import FastAPI
from pydantic import BaseModel

# initialize the application
app = FastAPI(title = "My App")

# define input data structure
class InputData(BaseModel):
    text: str

# function to predict outcome
def mock_predict(text: str):
    cv = "pass" if "ok" in text.lower() else "decline"
    return {"cv": cv, "status": "ok baby"}

# create endpoint API
@app.post("/predict")
async def predict_cv(data: InputData):
    result = mock_predict(data.text)
    return {
        "status": "success",
        "data": result
    }

# endpoint for checking the system's status
@app.get("/health")
def health_check():
    return {"status": "healthy"}