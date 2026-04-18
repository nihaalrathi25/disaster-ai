"""
api/main.py — Crisis Intelligence System Backend (v4 — fully rebuilt)

New in v4:
  - /geocode endpoint: resolves location strings → lat/lon via Nominatim
  - /refresh endpoint: triggers background scrape+enrich, non-blocking
  - /tweets endpoint: returns enriched tweets with coordinates
  - /status endpoint: shows pipeline run status
  - Background task queue so pipeline never blocks API
  - Geocoding cache in memory (persists across requests)
  - Full error handling on every route
"""

import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import requests as http_requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.inference import get_model_manager
from utils.location_extractor import extract_location, get_priority_label, load_spacy_model
from utils.preprocessor import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Global state
# ─────────────────────────────────────────
_nlp            = None
_model_manager  = None
_geocode_cache: Dict[str, Optional[Dict]] = {}   # location_str → {lat, lon} or None
_pipeline_status = {
    "running": False,
    "last_run": None,
    "last_error": None,
    "total_tweets": 0,
    "high_urgency": 0,
}
_executor = ThreadPoolExecutor(max_workers=2)

DATA_DIR  = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

NOMINATIM = "https://nominatim.openstreetmap.org/search"

# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────
app = FastAPI(
    title="Crisis Intelligence API",
    description="AI disaster tweet analyzer — classify, score, geolocate",
    version="4.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global _nlp, _model_manager
    logger.info("Loading models...")
    _nlp           = load_spacy_model()
    _model_manager = get_model_manager()
    logger.info("API ready — model: %s",
                "bert" if not _model_manager.is_using_fallback else "rule-based")
    # Auto-run pipeline on startup if no data exists
    if not (DATA_DIR / "enriched_tweets.csv").exists():
        logger.info("No data found — running initial pipeline...")
        asyncio.get_event_loop().run_in_executor(_executor, _run_pipeline_sync, False)


# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────
class TweetInput(BaseModel):
    text: str             = Field(..., min_length=1, max_length=1000)
    user_bio: str         = Field("", max_length=500)
    profile_location: str = Field("", max_length=200)

class TweetOutput(BaseModel):
    category:            str
    urgency_score:       float
    location:            str
    location_source:     str
    location_confidence: str
    priority:            str
    all_probs:           Dict[str, float]
    model_mode:          str
    lat:                 Optional[float] = None
    lon:                 Optional[float] = None

class BatchInput(BaseModel):
    tweets: List[TweetInput] = Field(..., max_items=100)

class GeocodeRequest(BaseModel):
    locations: List[str] = Field(..., max_items=200)


# ─────────────────────────────────────────
# Geocoding (Nominatim — free, no key)
# ─────────────────────────────────────────
def _geocode_single(location: str) -> Optional[Dict]:
    """Geocode one location string → {lat, lon} or None."""
    if not location or location.lower() in ("unknown", "", "none"):
        return None

    # Already a coordinate pair?
    m = re.match(r"^(-?\d+\.?\d*),\s*(-?\d+\.?\d*)$", location.strip())
    if m:
        return {"lat": float(m.group(1)), "lon": float(m.group(2))}

    # Cache hit
    if location in _geocode_cache:
        return _geocode_cache[location]

    try:
        resp = http_requests.get(
            NOMINATIM,
            params={"q": location, "format": "json", "limit": 1},
            headers={"User-Agent": "CrisisIntelligenceSystem/4.0 disaster-ai"},
            timeout=6,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                result = {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
                _geocode_cache[location] = result
                return result
    except Exception as e:
        logger.warning(f"Geocode failed for '{location}': {e}")

    _geocode_cache[location] = None
    return None


def _geocode_batch(locations: List[str]) -> Dict[str, Optional[Dict]]:
    """Geocode a list of locations respecting Nominatim rate limit (1 req/sec)."""
    unique = list(dict.fromkeys(l for l in locations if l and l != "unknown"))
    result = {}
    for i, loc in enumerate(unique):
        if loc in _geocode_cache:
            result[loc] = _geocode_cache[loc]
        else:
            result[loc] = _geocode_single(loc)
            if i < len(unique) - 1:
                time.sleep(1.1)   # Nominatim hard rate limit
    return result


# ─────────────────────────────────────────
# Core tweet processing
# ─────────────────────────────────────────
def _process_one(tweet: TweetInput, geocode: bool = True) -> TweetOutput:
    cleaned   = clean_text(tweet.text)
    prediction = _model_manager.predict_single(cleaned)
    hashtags  = " ".join(re.findall(r"#(\w+)", tweet.text))

    loc = extract_location(
        tweet_text=tweet.text,
        hashtags_str=hashtags,
        profile_location=tweet.profile_location or "",
        user_bio=tweet.user_bio or "",
        nlp=_nlp,
    )
    priority = get_priority_label(prediction["urgency_score"], loc["location_source"])

    lat, lon = None, None
    if geocode and loc["location"] != "unknown":
        coords = _geocode_single(loc["location"])
        if coords:
            lat, lon = coords["lat"], coords["lon"]

    return TweetOutput(
        category=prediction["category"],
        urgency_score=prediction["urgency_score"],
        location=loc["location"],
        location_source=loc["location_source"],
        location_confidence=loc["location_confidence"],
        priority=priority,
        all_probs=prediction["all_probs"],
        model_mode="bert" if not _model_manager.is_using_fallback else "rule-based",
        lat=lat,
        lon=lon,
    )


# ─────────────────────────────────────────
# Pipeline (runs in background thread)
# ─────────────────────────────────────────
def _run_pipeline_sync(force_scrape: bool = False):
    """Full scrape→preprocess→classify→locate→geocode→save pipeline."""
    global _pipeline_status
    import pandas as pd

    _pipeline_status["running"]    = True
    _pipeline_status["last_error"] = None

    try:
        from utils.scraper import scrape_tweets, _generate_synthetic_tweets
        from utils.preprocessor import preprocess_dataframe

        # Step 1 — collect tweets
        logger.info("Pipeline: collecting tweets...")
        try:
            raw = scrape_tweets(force_refresh=True)
        except Exception as e:
            logger.warning(f"Scraper failed ({e}), using synthetic data")
            raw = _generate_synthetic_tweets()

        df = pd.DataFrame(raw)
        if df.empty:
            df = pd.DataFrame(_generate_synthetic_tweets())

        # Step 2 — preprocess
        df = preprocess_dataframe(df)

        # Step 3 — classify
        logger.info(f"Pipeline: classifying {len(df)} tweets...")
        preds = _model_manager.predict_batch(df["clean_text"].tolist())
        df["category"]     = [p["category"]     for p in preds]
        df["urgency_score"]= [p["urgency_score"] for p in preds]

        # Step 4 — location extraction
        logger.info("Pipeline: extracting locations...")
        locs, srcs, confs, pris = [], [], [], []
        for _, row in df.iterrows():
            loc = extract_location(
                tweet_text=row.get("content", ""),
                hashtags_str=row.get("hashtags_str", ""),
                profile_location=row.get("user_location", ""),
                user_bio=row.get("user_bio", ""),
                nlp=_nlp,
            )
            locs.append(loc["location"])
            srcs.append(loc["location_source"])
            confs.append(loc["location_confidence"])
            pris.append(get_priority_label(row["urgency_score"], loc["location_source"]))

        df["location"]            = locs
        df["location_source"]     = srcs
        df["location_confidence"] = confs
        df["priority"]            = pris

        # Step 5 — geocode all known locations
        logger.info("Pipeline: geocoding locations...")
        known_locs = [l for l in df["location"].unique() if l and l != "unknown"]
        geo_map    = _geocode_batch(known_locs)

        df["lat"] = df["location"].map(lambda l: geo_map.get(l, {}).get("lat") if geo_map.get(l) else None)
        df["lon"] = df["location"].map(lambda l: geo_map.get(l, {}).get("lon") if geo_map.get(l) else None)

        # Step 6 — save
        out_cols = ["date","content","category","urgency_score","location",
                    "location_source","location_confidence","priority",
                    "lat","lon","username","user_location","query_used"]
        out_cols = [c for c in out_cols if c in df.columns]
        df = df[out_cols].sort_values("urgency_score", ascending=False)
        df.to_csv(DATA_DIR / "enriched_tweets.csv", index=False)

        _pipeline_status["last_run"]     = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        _pipeline_status["total_tweets"] = len(df)
        _pipeline_status["high_urgency"] = int((df["urgency_score"] > 70).sum())
        logger.info(f"Pipeline complete: {len(df)} tweets saved")

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        _pipeline_status["last_error"] = str(e)
    finally:
        _pipeline_status["running"] = False


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_mode":   "bert" if not _model_manager.is_using_fallback else "rule-based",
        "spacy_loaded": _nlp is not None,
        "version":      "4.0.0",
    }


@app.get("/status")
async def pipeline_status():
    """Returns current pipeline run status."""
    return _pipeline_status


@app.post("/refresh")
async def refresh(background_tasks: BackgroundTasks, force_scrape: bool = False):
    """
    Trigger a full pipeline refresh in the background.
    Returns immediately — poll /status to track progress.
    """
    if _pipeline_status["running"]:
        return {"message": "Pipeline already running", "status": _pipeline_status}

    background_tasks.add_task(_run_pipeline_sync, force_scrape)
    return {"message": "Pipeline started in background", "poll": "/status"}


@app.post("/predict", response_model=TweetOutput)
async def predict(tweet: TweetInput):
    """Classify a single tweet with location extraction and geocoding."""
    try:
        return _process_one(tweet, geocode=True)
    except Exception as e:
        logger.error(f"predict error: {e}")
        raise HTTPException(500, str(e))


@app.post("/batch")
async def batch_predict(batch: BatchInput):
    """Classify up to 100 tweets."""
    try:
        results = [_process_one(t, geocode=False) for t in batch.tweets]
        return {
            "total":                len(results),
            "results":              [r.dict() for r in results],
            "high_urgency_count":   sum(1 for r in results if r.urgency_score > 70),
            "unknown_location_count": sum(1 for r in results if r.location == "unknown"),
        }
    except Exception as e:
        logger.error(f"batch error: {e}")
        raise HTTPException(500, str(e))


@app.post("/geocode")
async def geocode_locations(req: GeocodeRequest):
    """
    Batch geocode a list of location strings → lat/lon.
    Results cached in memory, respects Nominatim 1 req/sec.
    """
    results = {}
    unique = list(dict.fromkeys(l for l in req.locations if l and l != "unknown"))

    for i, loc in enumerate(unique):
        coords = _geocode_single(loc)
        results[loc] = coords
        if coords and i < len(unique) - 1:
            time.sleep(1.1)

    return {"results": results, "total": len(unique), "found": sum(1 for v in results.values() if v)}


@app.get("/tweets")
async def get_tweets(
    limit:       int   = Query(100, ge=1, le=500),
    min_urgency: float = Query(0,   ge=0, le=100),
    category:    str   = Query("",  description="Filter by category"),
    location:    str   = Query("",  description="Filter by location text"),
    has_coords:  bool  = Query(False, description="Only return tweets with lat/lon"),
):
    """Return enriched tweets with coordinates. Ready for map rendering."""
    import pandas as pd

    path = DATA_DIR / "enriched_tweets.csv"
    if not path.exists():
        raise HTTPException(404, "No data yet. POST /refresh to generate.")

    df = pd.read_csv(path).fillna("")

    if min_urgency > 0:
        df = df[df["urgency_score"] >= min_urgency]
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if location:
        df = df[df["location"].str.contains(location, case=False, na=False)]
    if has_coords:
        df = df[df["lat"].notna() & df["lon"].notna()]

    df = df.sort_values("urgency_score", ascending=False).head(limit)

    return {
        "total":        len(df),
        "last_updated": _pipeline_status.get("last_run"),
        "data":         df.to_dict(orient="records"),
    }


@app.get("/stats")
async def stats():
    """Summary statistics for the current dataset."""
    import pandas as pd

    path = DATA_DIR / "enriched_tweets.csv"
    if not path.exists():
        raise HTTPException(404, "No data yet. POST /refresh to generate.")

    df = pd.read_csv(path).fillna("")
    geocoded = df[df["lat"].notna() & df["lon"].notna()]

    return {
        "total_tweets":      len(df),
        "high_urgency":      int((df["urgency_score"] > 70).sum()),
        "medium_urgency":    int(((df["urgency_score"] >= 40) & (df["urgency_score"] <= 70)).sum()),
        "low_urgency":       int((df["urgency_score"] < 40).sum()),
        "unknown_locations": int((df["location"] == "unknown").sum()),
        "geocoded_tweets":   len(geocoded),
        "categories":        df["category"].value_counts().to_dict(),
        "location_sources":  df["location_source"].value_counts().to_dict(),
        "priorities":        df["priority"].value_counts().to_dict(),
        "avg_urgency":       round(float(df["urgency_score"].mean()), 2),
        "pipeline":          _pipeline_status,
    }
