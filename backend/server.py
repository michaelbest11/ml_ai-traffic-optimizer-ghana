from fastapi import FastAPI, APIRouter, HTTPException # type: ignore
from dotenv import load_dotenv # type: ignore
from starlette.middleware.cors import CORSMiddleware # type: ignore
from motor.motor_asyncio import AsyncIOMotorClient # type: ignore
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field # type: ignore
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import asyncio
import json
import random
import math
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore
import pickle
import warnings

warnings.filterwarnings('ignore')

# Load .env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB Setup (Fail-safe)
try:
    mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get("DB_NAME", "traffic_db")]
except Exception as e:
    client = None
    db = None
    print("⚠️ MongoDB not connected:", e)

# FastAPI Setup
app = FastAPI(title="Traffic Flow Optimization API")
api_router = APIRouter(prefix="/api")

# Basic Models
class TrafficData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intersection_id: str
    city: str
    location: Dict[str, float]
    vehicle_count: int
    average_speed: float
    congestion_level: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    weather_condition: Optional[str] = "Clear"

class RouteRequest(BaseModel):
    start_location: Dict[str, float]
    end_location: Dict[str, float]
    city: str
    vehicle_type: str = "car"
    departure_time: Optional[datetime] = None

# Dummy Intersections
ACCRA_INTERSECTIONS = [
    {"id": "ACC_001", "name": "37 Military Hospital", "lat": 5.5600, "lng": -0.1969},
    {"id": "ACC_002", "name": "Circle", "lat": 5.5566, "lng": -0.1969},
]

KUMASI_INTERSECTIONS = [
    {"id": "KUM_001", "name": "Tech Junction", "lat": 6.6745, "lng": -1.5716},
    {"id": "KUM_002", "name": "Adum", "lat": 6.6961, "lng": -1.6208},
]

# Sample ML Engine Stub
class TrafficMLEngine:
    def __init__(self):
        self.is_trained = False
        self.model_accuracy = {"traffic_mae": 5.0, "speed_mae": 3.2, "congestion_mae": 0.12}

    def train_models(self, city):
        self.is_trained = True
        return True

    def predict_traffic(self, city, intersection_id, horizon):
        return {
            "intersection_id": intersection_id,
            "city": city,
            "predicted_congestion": random.choice(["Low", "Medium", "High"]),
            "predicted_vehicle_count": random.randint(40, 100),
            "confidence_score": round(random.uniform(0.85, 0.95), 2),
            "ml_model_used": "RandomForest + GradientBoosting",
            "timestamp": datetime.utcnow().isoformat()
        }

ml_engine = TrafficMLEngine()

# Routes
@api_router.get("/")
def home():
    return {"message": "Traffic Optimization API running"}

@api_router.get("/traffic/current/{city}")
async def get_current_traffic(city: str):
    if city not in ["Accra", "Kumasi"]:
        raise HTTPException(status_code=400, detail="City must be 'Accra' or 'Kumasi'")
    
    intersections = ACCRA_INTERSECTIONS if city == "Accra" else KUMASI_INTERSECTIONS
    traffic = []

    for i in intersections:
        data = TrafficData(
            intersection_id=i["id"],
            city=city,
            location={"lat": i["lat"], "lng": i["lng"]},
            vehicle_count=random.randint(30, 100),
            average_speed=random.uniform(15, 40),
            congestion_level=random.choice(["Low", "Medium", "High", "Critical"])
        )
        if db:
            await db.traffic_data.insert_one(data.dict())
        traffic.append(data.dict())

    return {
        "city": city,
        "traffic_data": traffic
    }

@api_router.get("/ml/predict/{city}/{intersection_id}")
async def predict(city: str, intersection_id: str, horizon: int = 60):
    if not ml_engine.is_trained:
        ml_engine.train_models(city)
    result = ml_engine.predict_traffic(city, intersection_id, horizon)
    return result

# Register router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and Shutdown Events
@app.on_event("startup")
async def startup():
    logging.info("Server is starting...")
    if db:
        await db.traffic_data.create_index("intersection_id")
    try:
        ml_engine.train_models("Accra")
    except Exception as e:
        logging.error(f"ML training failed: {e}")

@app.on_event("shutdown")
async def shutdown():
    if client:
        client.close()
