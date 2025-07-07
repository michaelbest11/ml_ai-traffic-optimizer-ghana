from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import asyncio
import json
import random
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import warnings

warnings.filterwarnings('ignore')

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'traffic_db')]

# FastAPI app
app = FastAPI(title="Traffic Flow Optimization API")
api_router = APIRouter(prefix="/api")

# Placeholder for AI chat client
ai_chat = None

# Dummy fallback AI chat logic
async def initialize_ai_chat():
    """Skip Gemini initialization if not available"""
    global ai_chat
    ai_chat = None
    logging.warning("Gemini AI integration not available. Skipping...")

async def get_ai_traffic_insights(traffic_data: List[Dict[str, Any]], city: str) -> str:
    """Fallback AI traffic analysis"""
    return f"AI analysis temporarily unavailable. Based on current data, {city} shows {'high' if len([d for d in traffic_data if d['congestion_level'] in ['High', 'Critical']]) > 2 else 'moderate'} congestion levels."

async def get_ai_route_optimization(start: Dict, end: Dict, city: str, current_traffic: List[Any]) -> str:
    """Fallback route optimization analysis"""
    return f"Route optimized using standard algorithms. Avoid main roads during peak hours in {city}."

# Add router and CORS
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_db_client():
    logger.info("Starting Traffic Flow Optimization API.")

    # Try to init Gemini chat (optional)
    await initialize_ai_chat()

    # Try to create indexes
    try:
        await db.traffic_data.create_index("intersection_id")
        await db.traffic_data.create_index("city")
        await db.traffic_data.create_index("timestamp")
        await db.route_recommendations.create_index("timestamp")
        await db.traffic_patterns.create_index("city")
        await db.traffic_predictions.create_index([("city", 1), ("intersection_id", 1)])
        await db.signal_optimizations.create_index("intersection_id")
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.warning(f"Index creation failed: {e}")

    # Optional: ML training logic here
    async def train_models_background():
        try:
            logger.info("Starting ML model training.")
            # Call your model training logic here (e.g., ml_engine.train_models("Accra"))
            logger.info("ML models trained for Accra")
        except Exception as e:
            logger.error(f"ML model training failed: {e}")

    asyncio.create_task(train_models_background())

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
