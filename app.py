import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

# Model
from transformers import pipeline

model=pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

ml_models={}

# Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["Sentiment"]=model
    print("Model cached and ready to serve requests")
    yield
    ml_models.clear()
    print("Memory cleared successfully")

app=FastAPI(
    title="ML Model Sentiment Analysis API",
    description="REST API for evaluating string text structures",
    version="1.0.0",
    lifespan=lifespan
)

security=HTTPBasic()
def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user="admin"
    correct_pass="123"
    if credentials.username!=correct_user or credentials.password!=correct_pass:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect API credentials",
            headers={"WWW-Authenticate": "Basic"}
        )
    return credentials.username

# Pydantic data regulator
class PredictionRequest(BaseModel):
    text: str=Field(..., min_length=3, description="The raw string content to analyze")
class PredictionResponse(BaseModel):
    label: str=Field(..., description="Calculated target class")
    confidence: float=Field(..., description="Confidence of the answer")

# Database saving (Imitation)
def log_metadata(text: str, label: str):
    print(f"[LOG TRACE LOGGER] {text} evaluated as '{label}'.")

# REST API Endpoints
@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to sentiment classification engine"}

@app.post("/predict", response_model=PredictionResponse, tags=["Machine Learning"])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks, user: str = Depends(authenticate_user)):

    if not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input string payload cannot be empty"
        )
    try:
        classifier=ml_models["Sentiment"]
        result=classifier(request.text)
        label=result[0]["label"]
        score=result[0]["score"]
        background_tasks.add_task(log_metadata, request.text, label)
        return {"label": label, "confidence": score}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model exceution failed: {str(e)}"
        )
