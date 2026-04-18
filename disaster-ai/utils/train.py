"""
train.py — BERT fine-tuning for disaster tweet classification
Labels: 0=Information, 1=Help Request (URGENT), 2=Damage Report

Uses Hugging Face Trainer API + MLflow tracking
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 3
LABEL_NAMES = ["Information", "Help Request", "Damage Report"]
LABEL2ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: name for idx, name in enumerate(LABEL_NAMES)}

SEED = 42
set_seed(SEED)


def build_training_dataset() -> pd.DataFrame:
    """
    Build a labeled training dataset.
    In production: load from CrisisLex, TREC-IS, or HumAID datasets.
    Here we provide a comprehensive synthetic dataset for demonstration.
    """
    data = []

    # Label 0 - Information
    info_tweets = [
        "Red Cross shelter opened at Lincoln High School. Free meals and bedding available.",
        "FEMA update: disaster relief funds approved for affected counties.",
        "National Guard deployed to flood zones for support operations.",
        "Earthquake recorded 6.1 magnitude near Pacific coast, no immediate damage reports.",
        "Weather service: tropical storm expected to weaken before landfall.",
        "Donation centers accepting supplies for hurricane victims in Tampa.",
        "Emergency hotline 1-800-FEMA activated for disaster assistance.",
        "Death toll from earthquake rises to 23 according to government officials.",
        "Power restoration underway, 60% of customers back online.",
        "Roads reopening after flood waters recede in downtown Memphis.",
        "Disaster relief teams assessing damage in affected neighborhoods.",
        "Governor declares state of emergency in 5 counties.",
        "Schools to remain closed this week due to ongoing flood damage.",
        "Insurance companies setting up temporary claim offices in disaster zones.",
        "Volunteers needed at local disaster relief centers this weekend.",
        "Utility crews working round the clock to restore electricity.",
        "Water distribution points set up across 12 locations in the city.",
        "Aerial survey confirms extent of wildfire damage — 5,000 acres burned.",
        "Emergency broadcast: curfew in effect for flood-affected zones.",
        "International aid arriving at airport for earthquake victims.",
        "Hospital reports treating 150 patients for flood-related injuries.",
        "Military helicopters assist with supply delivery to isolated communities.",
        "Damage assessment teams deployed across all affected districts.",
        "Shelter capacity expanded to accommodate 2,000 displaced residents.",
        "Communication towers restored — cell service returning to normal.",
    ]

    # Label 1 - Help Request (URGENT)
    help_tweets = [
        "URGENT: We are trapped on the roof! Flood water rising fast! Send rescue NOW!",
        "Help! My family is stuck on the 3rd floor. Water everywhere. No food. SOS!",
        "Please send rescue boats to Oak Street. 10 people stranded including elderly.",
        "I'm buried under rubble after earthquake. Can hear others. Please find us!",
        "EMERGENCY: Children trapped in school after landslide. Send help immediately!",
        "SOS! Stranded in car on highway with 2-year-old. Flood blocking all exits.",
        "Need medical help NOW. Person unconscious after building collapse. 5th and Main.",
        "We are running out of insulin. Diabetic patient needs urgent medical aid!",
        "Please rescue us! On roof of 123 Oak Ave. No food or water for 2 days!",
        "Rescue needed at community center. Roof collapsed. People trapped inside.",
        "Critical: 80-year-old woman needs medical evacuation. Too sick to walk.",
        "Fire spreading to residential area! Families trapped! Call fire department!",
        "Help us please. Water rising to chest level. Young children with us. Urgent!",
        "Earthquake trapped my parents in basement. They are alive but injured. Send help!",
        "Person having heart attack in flood zone. Roads blocked. Need air rescue!",
        "MAYDAY! Rescue boat capsized. 6 people in water. Send immediate help!",
        "We have been waiting 36 hours for rescue. No one has come. Please help!",
        "Pregnant woman in labor. Hospital road blocked by flood. Need emergency help!",
        "My house is on fire and the road is flooded. Cannot escape. SEND HELP.",
        "Three families trapped on apartment roof since yesterday. Need water and rescue.",
        "Disabled person unable to evacuate. Stuck on first floor. Water coming in fast.",
        "Gas leak after earthquake. People evacuating but some unable to move. Help!",
        "Bridge collapsed with car on it. Driver alive. Need rescue immediately!",
        "Multiple people injured in building collapse. We need paramedics and rescuers NOW.",
        "URGENT HELP NEEDED: 15 kids trapped in daycare after flood. Send boats!",
    ]

    # Label 2 - Damage Report
    damage_tweets = [
        "Entire neighborhood underwater. Hundreds of homes destroyed by flood.",
        "Earthquake damage: 40 buildings collapsed in downtown area. Major infrastructure loss.",
        "Wildfire destroyed 300 homes in Paradise area. Thousands displaced.",
        "Bridge collapsed after floods. Major road artery completely blocked.",
        "Power grid down for entire city after hurricane. 500,000 without electricity.",
        "Flood damage to water treatment plant. City water supply contaminated.",
        "Earthquake destroyed hospital. Medical services severely impacted.",
        "Tornado wiped out entire mobile home community. 200+ structures gone.",
        "Landslide blocked main highway. 50km of road inaccessible for weeks.",
        "Flood waters reached second floor of residential buildings across 3 blocks.",
        "Oil refinery damaged by hurricane. Spill risk in coastal waters.",
        "School building structurally compromised after earthquake. Evacuated.",
        "Flooding in subway tunnels. City transit completely halted.",
        "Wildfire burned through 2,000 acres of forest and farmland overnight.",
        "Storm surge destroyed beachfront properties across 15km of coastline.",
        "Earthquake fault line activated. Ground subsidence visible across farmland.",
        "Dam partially damaged. Engineers assessing risk of failure. Area on alert.",
        "Major shopping district underwater. Estimated billions in economic damage.",
        "Communications tower collapsed. Thousands without phone service.",
        "Airport runway flooded. All flights cancelled until further notice.",
        "Train derailment caused by flood-weakened track. No casualties reported.",
        "Factory roof collapsed under weight of snow after blizzard. No workers inside.",
        "Gas pipeline ruptured during earthquake. District evacuated as precaution.",
        "Flood destroyed crop fields across entire agricultural valley.",
        "Historic cathedral collapsed in earthquake. Cultural heritage lost.",
    ]

    for text in info_tweets:
        data.append({"text": text, "label": 0})
    for text in help_tweets:
        data.append({"text": text, "label": 1})
    for text in damage_tweets:
        data.append({"text": text, "label": 2})

    return pd.DataFrame(data)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=128,
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def train(
    model_name: str = MODEL_NAME,
    output_dir: Path = MODEL_DIR,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    use_mlflow: bool = True,
) -> None:
    """Train BERT for disaster tweet classification."""

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Building training dataset...")
    df = build_training_dataset()
    logger.info(f"Dataset size: {len(df)} | Label distribution:\n{df['label'].value_counts()}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    logger.info("Tokenizing datasets...")
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=SEED,
        report_to="none",  # We handle MLflow manually
        logging_steps=10,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # MLflow tracking
    run_id = None
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("disaster-tweet-classifier")
            mlflow.start_run()
            mlflow.log_params({
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_size": len(train_df),
                "val_size": len(val_df),
            })
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run started: {run_id}")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            use_mlflow = False

    logger.info("Starting training...")
    trainer.train()

    # Final evaluation
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    if use_mlflow:
        try:
            mlflow.log_metrics({
                "eval_accuracy": eval_results.get("eval_accuracy", 0),
                "eval_f1": eval_results.get("eval_f1", 0),
                "eval_precision": eval_results.get("eval_precision", 0),
                "eval_recall": eval_results.get("eval_recall", 0),
            })
        except Exception:
            pass

    # Save model and tokenizer
    save_path = output_dir / "disaster_bert"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    # Save label config
    label_config = {
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "label_names": LABEL_NAMES,
    }
    with open(save_path / "label_config.json", "w") as f:
        json.dump(label_config, f, indent=2)

    logger.info(f"Model saved to: {save_path}")

    if use_mlflow:
        try:
            mlflow.log_artifacts(str(save_path), artifact_path="model")
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train disaster tweet BERT classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no_mlflow", action="store_true")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_mlflow=not args.no_mlflow,
    )
