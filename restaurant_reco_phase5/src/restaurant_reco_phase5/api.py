from __future__ import annotations

import logging
import os
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from restaurant_reco_phase5.data_access import load_restaurants_parquet
from restaurant_reco_phase5.models import RecommendationRequest, RecommendationResponse
from restaurant_reco_phase5.recommender import recommend
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("restaurant_reco_phase5.api")

# Load .env at the top level
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(
    title="Restaurant Recommendation Service",
    description="Phase 5 - Production Ready Recommendation Service",
    version="0.1.0",
)

# Determine the base directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Global data storage
_restaurants_df = None

def get_data():
    global _restaurants_df
    if _restaurants_df is None:
        # Allow overriding data path via environment variable
        data_path = os.environ.get(
            "DATA_PATH", 
            "../restaurant_reco/data/processed/restaurants.parquet"
        )
        try:
            logger.info(f"Loading data from {data_path}")
            _restaurants_df = load_restaurants_parquet(data_path)
            logger.info(f"Loaded {len(_restaurants_df)} restaurants")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise RuntimeError(f"Could not load restaurant data: {e}")
    return _restaurants_df

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    # Add to headers
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request {request.method} {request.url.path} handled in {process_time:.4f}s")
    return response


@app.get("/")
def read_root():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    return FileResponse(index_path)

@app.get("/locations")
def get_locations():
    """Returns a list of unique locations from the dataset."""
    df = get_data()
    if df is None or df.empty:
        return []
    # Drop duplicates, sort, and return as list
    locations = sorted(df["location"].dropna().unique().tolist())
    return locations

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(req: RecommendationRequest):
    try:
        df = get_data()
        start_ts = time.time()
        
        resp = recommend(restaurants=df, request=req)
        
        latency = (time.time() - start_ts) * 1000
        if req.debug:
            # We add latency to debug info if it exists
            # (Note: DebugInfo model needs to be updated if we want this permanently)
            logger.info(f"Recommendation generated in {latency:.2f}ms")
            
        return resp
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
