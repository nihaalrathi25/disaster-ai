"""
location_extractor.py — Hybrid multi-source location extraction pipeline
Implements hierarchical extraction: tweet text → hashtags → profile → bio → unknown
"""

import re
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Confidence level mapping
CONFIDENCE_MAP = {
    "tweet": "high",
    "hashtag": "medium",
    "profile_location": "medium",
    "bio": "low",
    "none": "none",
}

# Known location-like hashtags pattern
LOCATION_HASHTAG_PATTERN = re.compile(
    r"(?:^|#)([A-Z][a-z]+(?:[A-Z][a-z]+)?(?:flood|quake|fire|storm|hurricane|cyclone|typhoon)?)\b"
)

# Regex for city/state combos in text
CITY_STATE_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z\s]{2,20}),\s*([A-Z]{2}|[A-Z][a-z]+)\b"
)

# Coordinates pattern (lat,lon in tweet)
COORDS_PATTERN = re.compile(r"(-?\d{1,3}\.\d+),\s*(-?\d{1,3}\.\d+)")


def _extract_spacy_gpe(text: str, nlp) -> Optional[str]:
    """Extract GPE (geopolitical entity) using spaCy NER."""
    if not text or not nlp:
        return None
    try:
        doc = nlp(text[:512])  # limit for speed
        gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        if gpes:
            return gpes[0]
    except Exception as e:
        logger.debug(f"spaCy extraction failed: {e}")
    return None


def _extract_from_hashtags(hashtags_str: str) -> Optional[str]:
    """Extract location hint from hashtag text."""
    if not hashtags_str:
        return None
    # Look for hashtags that contain known place names
    # Simple heuristic: hashtags with flood/quake suffixes likely have a location prefix
    for tag in hashtags_str.split():
        tag_clean = tag.lstrip("#")
        # Match patterns like HoustonFlood, LAfireDisaster, NYCstorm
        m = re.match(r"^([A-Z][a-z]+(?:[A-Z][a-z]+)?)(Flood|Quake|Fire|Storm|Hurricane|Cyclone|Typhoon|Disaster|Earthquake)$", tag_clean)
        if m:
            return m.group(1)
        # Match pure location-like tags (CamelCase, 3–15 chars)
        if re.match(r"^[A-Z][a-z]{2,14}$", tag_clean):
            return tag_clean
    return None


def _normalize_location(loc: str) -> Optional[str]:
    """Clean and normalize a location string."""
    if not loc or not isinstance(loc, str):
        return None
    loc = loc.strip()
    # Remove common noise
    noise = ["worldwide", "global", "earth", "everywhere", "internet", "web", "online", "🌍", "🌎", "🌏"]
    if loc.lower() in noise or len(loc) < 2:
        return None
    # Remove emoji
    loc = re.sub(r"[^\x00-\x7F]+", "", loc).strip()
    if not loc or len(loc) < 2:
        return None
    return loc[:80]  # max length


def extract_location(
    tweet_text: str,
    hashtags_str: str = "",
    profile_location: str = "",
    user_bio: str = "",
    nlp=None,
) -> Dict[str, str]:
    """
    Hierarchical location extraction pipeline.

    Returns:
        dict with keys: location, location_source, location_confidence
    """
    # Step 1: spaCy NER on tweet text
    loc = _extract_spacy_gpe(tweet_text, nlp)
    if loc:
        norm = _normalize_location(loc)
        if norm:
            return {
                "location": norm,
                "location_source": "tweet",
                "location_confidence": CONFIDENCE_MAP["tweet"],
            }

    # Step 1b: City, State regex in tweet text
    m = CITY_STATE_PATTERN.search(tweet_text)
    if m:
        norm = _normalize_location(f"{m.group(1)}, {m.group(2)}")
        if norm:
            return {
                "location": norm,
                "location_source": "tweet",
                "location_confidence": CONFIDENCE_MAP["tweet"],
            }

    # Step 2: Hashtag location
    loc = _extract_from_hashtags(hashtags_str)
    if loc:
        norm = _normalize_location(loc)
        if norm:
            return {
                "location": norm,
                "location_source": "hashtag",
                "location_confidence": CONFIDENCE_MAP["hashtag"],
            }

    # Step 3: Profile location field
    loc = _normalize_location(profile_location)
    if loc:
        return {
            "location": loc,
            "location_source": "profile_location",
            "location_confidence": CONFIDENCE_MAP["profile_location"],
        }

    # Step 4: User bio
    loc = _extract_spacy_gpe(user_bio, nlp)
    if not loc:
        # fallback: look for City, State in bio
        m = CITY_STATE_PATTERN.search(user_bio or "")
        if m:
            loc = f"{m.group(1)}, {m.group(2)}"
    if loc:
        norm = _normalize_location(loc)
        if norm:
            return {
                "location": norm,
                "location_source": "bio",
                "location_confidence": CONFIDENCE_MAP["bio"],
            }

    # Step 5: Unknown
    return {
        "location": "unknown",
        "location_source": "none",
        "location_confidence": CONFIDENCE_MAP["none"],
    }


def get_priority_label(urgency_score: float, location_source: str) -> str:
    """
    Assign priority label based on urgency and location source.
    """
    is_high_urgency = urgency_score > 70

    if is_high_urgency and location_source == "tweet":
        return "HIGHEST"
    elif is_high_urgency and location_source in ("hashtag", "profile_location"):
        return "HIGH"
    elif is_high_urgency and location_source in ("bio",):
        return "MEDIUM"
    elif is_high_urgency and location_source == "none":
        return "CRITICAL_REVIEW"  # urgent but no location
    elif urgency_score > 40:
        return "MEDIUM"
    else:
        return "LOW"


def load_spacy_model():
    """Load spaCy NLP model (en_core_web_sm)."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded: en_core_web_sm")
            return nlp
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            return None
    except ImportError:
        logger.warning("spaCy not installed.")
        return None


if __name__ == "__main__":
    nlp = load_spacy_model()
    tests = [
        {
            "tweet_text": "URGENT: Massive flooding in downtown Houston. People trapped! Send help!",
            "hashtags_str": "HoustonFlood rescue",
            "profile_location": "Houston, TX",
            "user_bio": "Texas resident",
        },
        {
            "tweet_text": "Please rescue us! We are stuck. No water no food.",
            "hashtags_str": "",
            "profile_location": "",
            "user_bio": "local resident | father",
        },
        {
            "tweet_text": "Earthquake hit near Los Angeles, buildings collapsing! #LAEarthquake",
            "hashtags_str": "LAEarthquake",
            "profile_location": "",
            "user_bio": "Journalist | West Coast",
        },
    ]
    for t in tests:
        result = extract_location(nlp=nlp, **t)
        print(f"TEXT: {t['tweet_text'][:60]}...")
        print(f"  → {result}\n")
