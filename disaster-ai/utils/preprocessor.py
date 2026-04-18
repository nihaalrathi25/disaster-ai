"""
preprocessor.py — Tweet text preprocessing for the disaster AI pipeline
"""

import re
import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Common disaster keywords for keyword-based urgency boost
URGENT_KEYWORDS = [
    "urgent", "help", "rescue", "trapped", "emergency", "sos",
    "dying", "stranded", "buried", "flood", "earthquake", "fire",
    "critical", "save", "need help", "please help", "now", "immediately",
    "disaster", "evacuate", "collapse", "injured", "missing",
]

MEDIUM_KEYWORDS = [
    "damage", "destroyed", "alert", "warning", "update", "report",
    "shelter", "relief", "aid", "victims", "displaced", "affected",
    "power outage", "blocked", "cut off", "closed",
]


def clean_text(text: str) -> str:
    """Clean tweet text for BERT input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Keep hashtag text (remove # symbol only)
    text = text.replace("#", "")
    # Remove non-ASCII characters except common punctuation
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Remove excessive punctuation
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    return text.strip()


def compute_keyword_score(text: str) -> float:
    """
    Compute a 0–100 keyword urgency score based on presence of disaster keywords.
    """
    text_lower = text.lower()
    score = 0.0
    for kw in URGENT_KEYWORDS:
        if kw in text_lower:
            score += 8.0
    for kw in MEDIUM_KEYWORDS:
        if kw in text_lower:
            score += 3.0
    # Cap at 100
    return min(score, 100.0)


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from tweet text."""
    return re.findall(r"#(\w+)", text, re.IGNORECASE)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to the full dataframe.
    Adds: clean_text, keyword_score, hashtags columns.
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ["content", "user_bio", "user_location"]:
        if col not in df.columns:
            df[col] = ""

    df["clean_text"] = df["content"].apply(clean_text)
    df["keyword_score"] = df["content"].apply(compute_keyword_score)
    df["hashtags"] = df["content"].apply(extract_hashtags)
    df["hashtags_str"] = df["hashtags"].apply(lambda x: " ".join(x))

    logger.info(f"Preprocessed {len(df)} tweets.")
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    sample = pd.DataFrame([
        {
            "content": "URGENT! Flood in #Houston TX — people trapped! @FEMA please help NOW https://t.co/xyz",
            "user_bio": "Texas local",
            "user_location": "Houston, TX",
        }
    ])
    result = preprocess_dataframe(sample)
    print(result[["clean_text", "keyword_score", "hashtags"]].to_string())
