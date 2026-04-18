"""
inference.py — Inference pipeline for disaster tweet classification + urgency scoring
Implements singleton model loading for FastAPI and batch pipeline for CSV processing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models" / "disaster_bert"

LABEL_NAMES = ["Information", "Help Request", "Damage Report"]
ID2LABEL = {0: "Information", 1: "Help Request", 2: "Damage Report"}

# Keywords for score blending
URGENT_KEYWORDS = [
    "urgent", "help", "rescue", "trapped", "emergency", "sos",
    "dying", "stranded", "buried", "flood", "earthquake", "fire",
    "critical", "save", "need help", "please help", "immediately",
    "disaster", "evacuate", "collapse", "injured", "missing",
]


def _keyword_urgency_score(text: str) -> float:
    """0–100 keyword-based urgency score."""
    text_lower = text.lower()
    score = 0.0
    for kw in URGENT_KEYWORDS:
        if kw in text_lower:
            score += 8.0
    return min(score, 100.0)


class ModelManager:
    """
    Singleton model manager — load once, reuse for all requests.
    Falls back to a rule-based classifier if no trained model is found.
    """
    _instance: Optional["ModelManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._tokenizer = None
        self._device = torch.device("cpu")
        self._use_fallback = False
        self._load_model()
        self._initialized = True

    def _load_model(self):
        model_path = MODEL_DIR
        if not model_path.exists():
            logger.warning(
                f"No trained model found at {model_path}. "
                "Using rule-based fallback classifier. "
                "Run utils/train.py to train the BERT model."
            )
            self._use_fallback = True
            return

        logger.info(f"Loading model from {model_path}")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self._model.to(self._device)
            self._model.eval()
            logger.info("BERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Using fallback.")
            self._use_fallback = True

    def predict_single(self, text: str) -> Dict:
        """
        Run inference on a single text.
        Returns: category, urgency_score, all_probs
        """
        if self._use_fallback:
            return self._fallback_predict(text)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        label_id = int(np.argmax(probs))
        category = ID2LABEL[label_id]

        # Urgency = probability of "Help Request" class (index 1) * 100
        model_score = float(probs[1]) * 100

        # Blend with keyword score
        kw_score = _keyword_urgency_score(text)
        final_score = 0.8 * model_score + 0.2 * kw_score

        return {
            "category": category,
            "urgency_score": round(float(final_score), 2),
            "all_probs": {
                "Information": round(float(probs[0]), 4),
                "Help Request": round(float(probs[1]), 4),
                "Damage Report": round(float(probs[2]), 4),
            },
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Batch inference."""
        if self._use_fallback:
            return [self._fallback_predict(t) for t in texts]

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            for j, prob in enumerate(probs):
                label_id = int(np.argmax(prob))
                category = ID2LABEL[label_id]
                model_score = float(prob[1]) * 100
                kw_score = _keyword_urgency_score(batch[j])
                final_score = 0.8 * model_score + 0.2 * kw_score

                results.append({
                    "category": category,
                    "urgency_score": round(float(final_score), 2),
                    "all_probs": {
                        "Information": round(float(prob[0]), 4),
                        "Help Request": round(float(prob[1]), 4),
                        "Damage Report": round(float(prob[2]), 4),
                    },
                })
        return results

    def _fallback_predict(self, text: str) -> Dict:
        """
        Rule-based fallback when no trained model is available.
        Uses keyword scoring to estimate category and urgency.
        """
        text_lower = text.lower()

        urgent_words = ["urgent", "help", "rescue", "trapped", "sos", "emergency", "stranded", "dying", "please help", "send help"]
        damage_words = ["destroyed", "collapsed", "damage", "ruins", "burning", "washed away", "demolished", "devastated"]
        info_words = ["update", "report", "shelter", "teams deployed", "fema", "volunteers", "hotline", "closed", "reopening"]

        urgent_score = sum(1 for w in urgent_words if w in text_lower)
        damage_score = sum(1 for w in damage_words if w in text_lower)
        info_score = sum(1 for w in info_words if w in text_lower)

        if urgent_score >= damage_score and urgent_score >= info_score and urgent_score > 0:
            category = "Help Request"
            # Higher urgency for help requests
            model_prob = min(0.3 + urgent_score * 0.12, 0.95)
        elif damage_score > info_score and damage_score > 0:
            category = "Damage Report"
            model_prob = 0.1
        else:
            category = "Information"
            model_prob = 0.05

        model_score = model_prob * 100
        kw_score = _keyword_urgency_score(text)
        final_score = 0.8 * model_score + 0.2 * kw_score

        # Fake probability distribution
        if category == "Help Request":
            probs = [max(0.0, 1.0 - model_prob - 0.1), model_prob, 0.1]
        elif category == "Damage Report":
            probs = [0.15, 0.1, 0.75]
        else:
            probs = [0.75, 0.1, 0.15]

        return {
            "category": category,
            "urgency_score": round(float(final_score), 2),
            "all_probs": {
                "Information": round(probs[0], 4),
                "Help Request": round(probs[1], 4),
                "Damage Report": round(probs[2], 4),
            },
        }

    @property
    def is_using_fallback(self) -> bool:
        return self._use_fallback


# Global singleton
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def run_full_pipeline(df):
    """
    Run full enrichment pipeline on a DataFrame.
    Adds: category, urgency_score, location, location_source, location_confidence, priority
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.preprocessor import preprocess_dataframe
    from utils.location_extractor import extract_location, get_priority_label, load_spacy_model

    logger.info("Preprocessing tweets...")
    df = preprocess_dataframe(df)

    logger.info("Loading NLP models...")
    nlp = load_spacy_model()
    manager = get_model_manager()

    logger.info("Running BERT inference...")
    texts = df["clean_text"].tolist()
    predictions = manager.predict_batch(texts)

    categories = [p["category"] for p in predictions]
    urgency_scores = [p["urgency_score"] for p in predictions]

    df["category"] = categories
    df["urgency_score"] = urgency_scores

    logger.info("Extracting locations...")
    locations = []
    sources = []
    confidences = []
    priorities = []

    for _, row in df.iterrows():
        loc_result = extract_location(
            tweet_text=row.get("content", ""),
            hashtags_str=row.get("hashtags_str", ""),
            profile_location=row.get("user_location", ""),
            user_bio=row.get("user_bio", ""),
            nlp=nlp,
        )
        locations.append(loc_result["location"])
        sources.append(loc_result["location_source"])
        confidences.append(loc_result["location_confidence"])
        priorities.append(get_priority_label(row["urgency_score"], loc_result["location_source"]))

    df["location"] = locations
    df["location_source"] = sources
    df["location_confidence"] = confidences
    df["priority"] = priorities

    return df


if __name__ == "__main__":
    import pandas as pd
    test_texts = [
        "URGENT! Trapped on rooftop in Houston flood. Need rescue NOW! #HoustonFlood",
        "Red Cross shelter opened at Lincoln High School. Food available.",
        "Earthquake destroyed 40 buildings in downtown. Major damage reported.",
    ]
    manager = get_model_manager()
    for text in test_texts:
        result = manager.predict_single(text)
        print(f"TEXT: {text[:60]}...")
        print(f"  Category: {result['category']}")
        print(f"  Urgency:  {result['urgency_score']:.1f}")
        print(f"  Probs:    {result['all_probs']}")
        print()
