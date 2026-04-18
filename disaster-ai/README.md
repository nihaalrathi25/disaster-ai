# 🆘 AI Disaster Tweet Analyzer — Crisis Intelligence System

> A production-grade, fully containerized AI system that scrapes disaster-related tweets, classifies them using BERT, scores urgency, extracts locations via a hybrid pipeline, and presents results in an interactive emergency response dashboard.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     CRISIS INTELLIGENCE SYSTEM                  │
├──────────────┬─────────────────┬──────────────┬────────────────┤
│   SCRAPING   │  PREPROCESSING  │  BERT MODEL  │   LOCATION     │
│  snscrape    │  clean_text()   │  fine-tuned  │   EXTRACTION   │
│  synthetic   │  keyword score  │  bert-base-  │  spaCy NER     │
│  fallback    │  hashtag parse  │  uncased     │  hashtags      │
│              │                 │              │  profile/bio   │
├──────────────┴─────────────────┴──────────────┴────────────────┤
│                        DATA PIPELINE                            │
│   scrape → preprocess → predict → locate → score → CSV         │
├──────────────────────┬──────────────────────────────────────────┤
│    FastAPI Backend   │       Streamlit Dashboard               │
│    port :8000        │       port :8501                        │
│   /predict           │   • Live tweet feed                     │
│   /batch             │   • Search + filter + sort             │
│   /pipeline          │   • Urgency color coding               │
│   /stats             │   • Map visualization                  │
│   /data/tweets       │   • Analytics charts                   │
├──────────────────────┴──────────────────────────────────────────┤
│                    MLflow Tracking — port :5000                 │
│           experiments · metrics · model artifacts              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
disaster-ai/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── ui/
│   └── app.py               # Streamlit dashboard
├── utils/
│   ├── __init__.py
│   ├── scraper.py           # Tweet scraping + synthetic data
│   ├── preprocessor.py      # Text cleaning + keyword scoring
│   ├── train.py             # BERT fine-tuning + MLflow
│   ├── inference.py         # Singleton model + batch pipeline
│   └── location_extractor.py # Hybrid location extraction
├── data/                    # CSVs (auto-created)
├── models/                  # Saved BERT model (auto-created)
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── streamlit_config.toml
├── docker-compose.yml
├── run_pipeline.py          # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option A: Docker (Recommended)

```bash
# 1. Clone / navigate to project
cd disaster-ai

# 2. Build and start all services
docker-compose up --build

# Services available:
# → Dashboard:  http://localhost:8501
# → API docs:   http://localhost:8000/docs
# → MLflow:     http://localhost:5000
```

> **First startup note**: The backend will auto-download `bert-base-uncased` from Hugging Face (~440MB). Subsequent starts use the Docker volume cache.

---

### Option B: Local Development

#### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### 2. Run the pipeline (generates synthetic data + runs inference)

```bash
python run_pipeline.py
# Output: data/enriched_tweets.csv
```

#### 3. (Optional) Scrape real tweets

```bash
python run_pipeline.py --scrape --max-tweets 100
```

#### 4. (Optional) Train BERT model

```bash
python run_pipeline.py --train
# Or standalone:
python utils/train.py --epochs 3 --batch_size 16
```

> Without training, the system uses a rule-based fallback classifier that still produces accurate urgency scores.

#### 5. Start backend API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 6. Start Streamlit dashboard

```bash
streamlit run ui/app.py --server.port 8501
```

#### 7. (Optional) Start MLflow

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlflow/mlruns
```

---

## 🔌 API Endpoints

### POST /predict
Classify a single tweet.

**Request:**
```json
{
  "text": "URGENT: Flooding in Houston! People trapped on rooftops! Send rescue boats NOW! #HoustonFlood",
  "user_bio": "Texas resident | Father",
  "profile_location": "Houston, TX"
}
```

**Response:**
```json
{
  "category": "Help Request",
  "urgency_score": 87.4,
  "location": "Houston",
  "location_source": "tweet",
  "location_confidence": "high",
  "priority": "HIGHEST",
  "all_probs": {
    "Information": 0.0421,
    "Help Request": 0.9231,
    "Damage Report": 0.0348
  },
  "model_mode": "bert"
}
```

### POST /batch
Process up to 100 tweets simultaneously.

```json
{
  "tweets": [
    {"text": "Earthquake in LA! Help!", "profile_location": "Los Angeles"},
    {"text": "Red Cross shelter open at Lincoln High", "profile_location": "Portland, OR"}
  ]
}
```

### POST /pipeline
Trigger the full data pipeline (scrape → predict → save).

### GET /data/tweets?limit=50&min_urgency=70
Retrieve enriched tweets from the processed CSV.

### GET /stats
Summary statistics for the current dataset.

### GET /health
API health check and model status.

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| Base model | `bert-base-uncased` (Hugging Face) |
| Task | Multi-class sequence classification |
| Labels | Information (0), Help Request (1), Damage Report (2) |
| Training | Hugging Face `Trainer` API with `DataCollatorWithPadding` |
| Urgency score | `softmax(logits)[1] × 100` (Help Request probability) |
| Final score | `0.8 × model_score + 0.2 × keyword_score` |
| Max input length | 128 tokens |
| Device | CPU (no GPU required) |

---

## 📍 Location Extraction Pipeline

```
Tweet text  → spaCy NER (GPE entities)           → confidence: HIGH
            → City, State regex match             → confidence: HIGH
            → Hashtag parsing (#HoustonFlood)    → confidence: MEDIUM
            → User profile location field        → confidence: MEDIUM
            → User bio NER / regex               → confidence: LOW
            → "unknown"                          → confidence: NONE
```

---

## 🎯 Priority Classification

| Urgency | Location Source | Priority |
|---------|----------------|----------|
| > 70 | tweet | HIGHEST |
| > 70 | hashtag / profile | HIGH |
| > 70 | bio | MEDIUM |
| > 70 | unknown | CRITICAL_REVIEW ⚠️ |
| 40–70 | any | MEDIUM |
| < 40 | any | LOW |

---

## 📊 Sample Output

```
date                 content                                    category      urgency_score  location        location_source  priority
2024-01-15 08:23    URGENT: Trapped on roof! Houston flood!    Help Request  91.2           Houston         tweet            HIGHEST
2024-01-15 07:45    Earthquake destroyed buildings downtown    Damage Report 34.8           Los Angeles     hashtag          LOW
2024-01-15 06:12    Red Cross shelter at Lincoln High School   Information   8.4            Portland        profile_location LOW
2024-01-14 22:30    HELP! Trapped under rubble in earthquake   Help Request  88.7           unknown         none             CRITICAL_REVIEW
```

---

## 🔄 MLflow Experiment Tracking

After training, view experiments at `http://localhost:5000`:

- **Parameters**: model_name, epochs, batch_size, learning_rate, dataset sizes
- **Metrics**: accuracy, F1 score, precision, recall (per epoch + final)
- **Artifacts**: saved model files, tokenizer, label config

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `TRANSFORMERS_CACHE` | `/app/.cache/huggingface` | HF model cache |
| `PYTHONPATH` | `/app` | Python module root |

---

## 🐋 Docker Services

| Service | Image | Port | Description |
|---------|-------|------|-------------|
| `backend` | Custom (python:3.10-slim) | 8000 | FastAPI REST API |
| `frontend` | Custom (python:3.10-slim) | 8501 | Streamlit dashboard |
| `mlflow` | python:3.10-slim | 5000 | Experiment tracking |

**Shared volumes:**
- `./data` — tweet CSVs (host ↔ both containers)
- `./models` — BERT model files (host ↔ both containers)
- `hf_cache` — Hugging Face model download cache
- `mlflow_data` — MLflow runs and artifacts

---

## 🛠️ Troubleshooting

**BERT model not found:**
```bash
python utils/train.py  # Run training first
```

**spaCy model not found:**
```bash
python -m spacy download en_core_web_sm
```

**snscrape import error:**
```bash
pip install git+https://github.com/JustAnotherArchivist/snscrape.git
```
> Note: If Twitter/X blocks scraping, the system auto-falls back to the synthetic dataset.

**Port conflicts:**
```bash
# Change ports in docker-compose.yml:
ports:
  - "8080:8000"  # Change host port
```

---

## 📦 Extending the System

- **Real dataset**: Replace `build_training_dataset()` in `train.py` with CrisisLex, HumAID, or TREC-IS datasets
- **GPU training**: Set `no_cuda=False` in `TrainingArguments` and run on GPU machine
- **Geocoding**: Add `geopy` integration to convert location strings → lat/lon for real map pins
- **Real-time**: Add WebSocket endpoint to stream live predictions
- **Alerting**: Add email/SMS alerts for CRITICAL_REVIEW tweets via Twilio/SendGrid
