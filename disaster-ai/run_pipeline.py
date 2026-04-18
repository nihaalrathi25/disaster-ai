"""
run_pipeline.py — Full end-to-end pipeline runner
Usage: python run_pipeline.py [--scrape] [--train] [--max-tweets 50]
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def run_pipeline(scrape: bool = True, train: bool = False, max_tweets: int = 50):
    import pandas as pd

    DATA_DIR = Path(__file__).parent / "data"
    DATA_DIR.mkdir(exist_ok=True)

    # ── Step 1: Data Collection ──────────────
    if scrape:
        logger.info("═" * 50)
        logger.info("STEP 1: Scraping tweets...")
        logger.info("═" * 50)
        from utils.scraper import scrape_tweets
        tweets = scrape_tweets(max_per_query=max_tweets // 4)
        df = pd.DataFrame(tweets)
        logger.info(f"Collected {len(df)} tweets")
    else:
        raw_path = DATA_DIR / "raw_tweets.csv"
        if raw_path.exists():
            logger.info(f"Loading existing data from {raw_path}")
            df = pd.read_csv(raw_path)
        else:
            logger.info("No raw data found, generating synthetic tweets...")
            from utils.scraper import _generate_synthetic_tweets
            df = pd.DataFrame(_generate_synthetic_tweets())
            df.to_csv(raw_path, index=False)

    logger.info(f"Dataset shape: {df.shape}")

    # ── Step 2: Train model (optional) ───────
    if train:
        logger.info("═" * 50)
        logger.info("STEP 2: Training BERT model...")
        logger.info("═" * 50)
        from utils.train import train as train_model
        train_model()
    else:
        model_path = Path(__file__).parent / "models" / "disaster_bert"
        if not model_path.exists():
            logger.warning("No trained model found. Will use rule-based fallback.")
            logger.info("To train: python run_pipeline.py --train")

    # ── Step 3: Preprocessing ────────────────
    logger.info("═" * 50)
    logger.info("STEP 3: Preprocessing tweets...")
    logger.info("═" * 50)
    from utils.preprocessor import preprocess_dataframe
    df = preprocess_dataframe(df)

    # ── Step 4: Inference ────────────────────
    logger.info("═" * 50)
    logger.info("STEP 4: Running AI classification...")
    logger.info("═" * 50)
    from utils.inference import get_model_manager
    manager = get_model_manager()
    logger.info(f"Model mode: {'BERT' if not manager.is_using_fallback else 'Rule-based fallback'}")

    texts = df["clean_text"].tolist()
    predictions = manager.predict_batch(texts)
    df["category"] = [p["category"] for p in predictions]
    df["urgency_score"] = [p["urgency_score"] for p in predictions]

    # ── Step 5: Location Extraction ──────────
    logger.info("═" * 50)
    logger.info("STEP 5: Extracting locations...")
    logger.info("═" * 50)
    from utils.location_extractor import extract_location, get_priority_label, load_spacy_model

    nlp = load_spacy_model()
    locations, sources, confidences, priorities = [], [], [], []

    for _, row in df.iterrows():
        loc = extract_location(
            tweet_text=row.get("content", ""),
            hashtags_str=row.get("hashtags_str", ""),
            profile_location=row.get("user_location", ""),
            user_bio=row.get("user_bio", ""),
            nlp=nlp,
        )
        locations.append(loc["location"])
        sources.append(loc["location_source"])
        confidences.append(loc["location_confidence"])
        priorities.append(get_priority_label(row["urgency_score"], loc["location_source"]))

    df["location"] = locations
    df["location_source"] = sources
    df["location_confidence"] = confidences
    df["priority"] = priorities

    # ── Step 6: Output ───────────────────────
    logger.info("═" * 50)
    logger.info("STEP 6: Saving enriched dataset...")
    logger.info("═" * 50)

    output_cols = [
        "date", "content", "category", "urgency_score",
        "location", "location_source", "location_confidence", "priority",
        "username", "user_location", "query_used",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    out_df = df[output_cols].copy()
    out_df = out_df.sort_values("urgency_score", ascending=False)

    output_path = DATA_DIR / "enriched_tweets.csv"
    out_df.to_csv(output_path, index=False)

    # ── Summary ──────────────────────────────
    logger.info("═" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("═" * 50)
    logger.info(f"Total tweets processed : {len(out_df)}")
    logger.info(f"High urgency (>70)     : {(out_df['urgency_score'] > 70).sum()}")
    logger.info(f"Unknown locations      : {(out_df['location'] == 'unknown').sum()}")
    logger.info(f"Output saved to        : {output_path}")
    logger.info("")
    logger.info("Category distribution:")
    for cat, count in out_df["category"].value_counts().items():
        logger.info(f"  {cat}: {count}")
    logger.info("")
    logger.info("Priority distribution:")
    for pri, count in out_df["priority"].value_counts().items():
        logger.info(f"  {pri}: {count}")
    logger.info("")
    logger.info("Top 5 URGENT tweets:")
    for _, row in out_df[out_df["urgency_score"] > 70].head(5).iterrows():
        logger.info(f"  [{row['urgency_score']:.1f}] {str(row['content'])[:80]}...")
        logger.info(f"          Location: {row['location']} ({row['location_source']}) | {row['priority']}")

    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Disaster Tweet Analyzer Pipeline")
    parser.add_argument("--scrape", action="store_true", help="Run tweet scraper")
    parser.add_argument("--train", action="store_true", help="Train BERT model")
    parser.add_argument("--max-tweets", type=int, default=50, help="Max tweets per query")
    args = parser.parse_args()

    run_pipeline(
        scrape=args.scrape,
        train=args.train,
        max_tweets=args.max_tweets,
    )
