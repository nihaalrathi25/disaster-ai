"""
ui/app.py — Crisis Intelligence Dashboard (v4 — complete rewrite)

Design principles:
  - No CSS variables in inline styles (Streamlit strips them)
  - No time.sleep() in render path (causes infinite loading)
  - Sidebar toggle via Streamlit native button, not JS hacks
  - Map uses backend /geocode endpoint (coordinates already in CSV)
  - All API calls have timeouts + fallbacks
  - State managed cleanly via st.session_state
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE = "http://backend:8000"
DATA_DIR  = Path(__file__).parent.parent / "data"
CSV_PATH  = DATA_DIR / "enriched_tweets.csv"
API_TIMEOUT = 8   # seconds

st.set_page_config(
    page_title="Crisis Intelligence",
    page_icon="🆘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ────────────────────────────────────────────────────────────
_defaults = {
    "show_sidebar":   True,
    "pipeline_msg":   "",
    "last_refresh":   0.0,
    "refresh_error":  "",
    "auto_refresh":   False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@700;900&family=Barlow:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0c10 !important;
    color: #e8eaed !important;
}
[data-testid="stSidebar"] {
    background: #111419 !important;
    border-right: 1px solid #1e2530 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stTextInput label {
    color: #8d9db0 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.main .block-container { padding: 1rem 1.5rem; max-width: 100%; }
#MainMenu, footer, header { visibility: hidden; }

/* buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #ff2d2d !important;
    color: #ff2d2d !important;
    font-size: 0.72rem !important;
    border-radius: 3px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #ff2d2d !important;
    color: #fff !important;
}

/* tabs */
.stTabs [data-baseweb="tab"] {
    color: #8d9db0 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
}
.stTabs [aria-selected="true"] {
    color: #00b4ff !important;
    border-bottom: 2px solid #00b4ff !important;
}
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #1e2530 !important;
}

/* header */
.hdr {
    background: linear-gradient(135deg, #0a0c10, #0f1520, #0a0c10);
    border: 1px solid #ff2d2d;
    border-left: 4px solid #ff2d2d;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}
.hdr-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem; font-weight: 900;
    color: #e8eaed; text-transform: uppercase;
    letter-spacing: 0.05em; margin: 0; line-height: 1;
}
.hdr-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem; color: #00b4ff;
    letter-spacing: 0.15em; margin-top: 0.3rem;
}

/* metric cards */
.mgrid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.7rem; margin-bottom: 1rem;
}
.mcard {
    background: #111419;
    border: 1px solid #1e2530;
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
    position: relative; overflow: hidden;
}
.mcard::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
}
.mc-blue::after  { background: #00b4ff; }
.mc-red::after   { background: #ff2d2d; }
.mc-orange::after{ background: #ff7a00; }
.mc-yellow::after{ background: #ffd600; }
.mlbl { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    color: #8d9db0; text-transform: uppercase; letter-spacing: 0.1em; }
.mval { font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem; font-weight: 700; line-height: 1; margin-top: 0.15rem; }
.mc-blue  .mval { color: #00b4ff; }
.mc-red   .mval { color: #ff2d2d; }
.mc-orange .mval{ color: #ff7a00; }
.mc-yellow .mval{ color: #ffd600; }

/* tweet cards */
.tcard {
    background: #111419; border: 1px solid #1e2530;
    border-left: 3px solid; border-radius: 0 4px 4px 0;
    padding: 0.8rem 1rem; margin-bottom: 0.5rem;
}
.tc-high { border-left-color: #ff2d2d; }
.tc-med  { border-left-color: #ff7a00; }
.tc-low  { border-left-color: #00e676; }
.tc-crit { background: rgba(255,45,45,0.06); border-left-color: #ff2d2d; }
.tmeta   { display: flex; gap: 0.6rem; align-items: center;
    flex-wrap: wrap; margin-bottom: 0.4rem; }
.ttext   { font-size: 0.88rem; line-height: 1.5; color: #e8eaed; margin-bottom: 0.35rem; }
.tfoot   { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }

/* badges */
.badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.06em;
    padding: 0.15rem 0.5rem; border-radius: 2px;
    text-transform: uppercase; font-weight: 600; display: inline-block;
}
.bd-r { background: rgba(255,45,45,.15);  color: #ff2d2d; border: 1px solid rgba(255,45,45,.3); }
.bd-o { background: rgba(255,122,0,.15);  color: #ff7a00; border: 1px solid rgba(255,122,0,.3); }
.bd-g { background: rgba(0,230,118,.12);  color: #00e676; border: 1px solid rgba(0,230,118,.25); }
.bd-b { background: rgba(0,180,255,.12);  color: #00b4ff; border: 1px solid rgba(0,180,255,.25); }
.bd-y { background: rgba(255,214,0,.12);  color: #ffd600; border: 1px solid rgba(255,214,0,.25); }
.bd-x { background: rgba(141,157,176,.1); color: #8d9db0; border: 1px solid rgba(141,157,176,.2); }

.sc-h { font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; font-weight: bold; color: #ff2d2d; }
.sc-m { font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; font-weight: bold; color: #ff7a00; }
.sc-l { font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; font-weight: bold; color: #00e676; }
.crit-f { font-family: 'Share Tech Mono', monospace; font-size: 0.58rem;
    background: rgba(255,214,0,.12); color: #ffd600;
    border: 1px solid rgba(255,214,0,.3); padding: 0.12rem 0.4rem; text-transform: uppercase; }
.loctext { font-family: 'Share Tech Mono', monospace; font-size: 0.65rem; color: #8d9db0; }
.umeta   { font-size: 0.6rem; color: #8d9db0; margin-left: auto; }

/* status bar */
.sbar {
    display: flex; justify-content: space-between; align-items: center;
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    color: #8d9db0; padding: 0.4rem 0; margin-bottom: 0.6rem;
    border-bottom: 1px solid #1e2530;
}
.pill {
    display: inline-block; padding: 0.2rem 0.6rem;
    border-radius: 20px; font-size: 0.6rem;
    font-family: 'Share Tech Mono', monospace;
}
.pill-online  { background: rgba(0,230,118,.15); color: #00e676; border: 1px solid rgba(0,230,118,.3); }
.pill-offline { background: rgba(255,45,45,.15);  color: #ff2d2d; border: 1px solid rgba(255,45,45,.3); }
.pill-running { background: rgba(0,180,255,.15);  color: #00b4ff; border: 1px solid rgba(0,180,255,.3); }

.info-box {
    background: rgba(0,180,255,.07); border: 1px solid rgba(0,180,255,.2);
    border-radius: 4px; padding: 0.6rem 0.9rem;
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    color: #00b4ff; margin-bottom: 0.7rem;
}
.err-box {
    background: rgba(255,45,45,.07); border: 1px solid rgba(255,45,45,.2);
    border-radius: 4px; padding: 0.6rem 0.9rem;
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    color: #ff6666; margin-bottom: 0.7rem;
}
.no-data {
    text-align: center; padding: 2rem;
    font-family: 'Share Tech Mono', monospace; font-size: 0.8rem;
    color: #8d9db0; border: 1px dashed #1e2530; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
def api_get(path: str, params: dict = None, timeout: int = API_TIMEOUT):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except requests.exceptions.ConnectionError:
        return None, "Backend offline"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)


def api_post(path: str, json_body: dict = None, timeout: int = API_TIMEOUT):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json_body, timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except requests.exceptions.ConnectionError:
        return None, "Backend offline"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)


def get_api_health():
    data, err = api_get("/health", timeout=3)
    if data:
        return data
    return {"status": "offline", "model_mode": "unknown"}


def get_pipeline_status():
    data, _ = api_get("/status", timeout=3)
    return data or {"running": False, "last_run": None, "last_error": None}


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_tweets_from_api(limit: int = 200) -> pd.DataFrame:
    """Fetch tweets from backend API."""
    data, err = api_get("/tweets", params={"limit": limit})
    if data and data.get("data"):
        return pd.DataFrame(data["data"]).fillna("")
    return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_tweets_from_csv() -> pd.DataFrame:
    """Fallback: load directly from CSV."""
    if CSV_PATH.exists():
        try:
            return pd.read_csv(CSV_PATH).fillna("")
        except Exception:
            pass
    return pd.DataFrame()


def load_data() -> pd.DataFrame:
    """Try API first, fall back to CSV, then run local pipeline."""
    df = load_tweets_from_api()
    if not df.empty:
        return df

    df = load_tweets_from_csv()
    if not df.empty:
        return df

    # Last resort: run local pipeline
    try:
        from utils.scraper import _generate_synthetic_tweets
        from utils.preprocessor import preprocess_dataframe
        from utils.inference import get_model_manager
        from utils.location_extractor import extract_location, get_priority_label, load_spacy_model

        raw = _generate_synthetic_tweets()
        df  = pd.DataFrame(raw)
        df  = preprocess_dataframe(df)
        mgr = get_model_manager()
        preds = mgr.predict_batch(df["clean_text"].tolist())
        df["category"]      = [p["category"]      for p in preds]
        df["urgency_score"] = [p["urgency_score"]  for p in preds]
        nlp = load_spacy_model()
        locs, srcs, confs, pris = [], [], [], []
        for _, row in df.iterrows():
            loc = extract_location(
                tweet_text=row.get("content",""),
                hashtags_str=row.get("hashtags_str",""),
                profile_location=row.get("user_location",""),
                user_bio=row.get("user_bio",""),
                nlp=nlp,
            )
            locs.append(loc["location"])
            srcs.append(loc["location_source"])
            confs.append(loc["location_confidence"])
            pris.append(get_priority_label(row["urgency_score"], loc["location_source"]))
        df["location"]            = locs
        df["location_source"]     = srcs
        df["location_confidence"] = confs
        df["priority"]            = pris
        df["lat"] = None
        df["lon"] = None
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        return df.fillna("")
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()


# ── Card renderers ────────────────────────────────────────────────────────────
def _score_html(s: float) -> str:
    if s > 70:  return f'<span class="sc-h">&#9889; {s:.0f}</span>'
    if s >= 40: return f'<span class="sc-m">&#9650; {s:.0f}</span>'
    return f'<span class="sc-l">&#9679; {s:.0f}</span>'

def _cat_html(c: str) -> str:
    m = {"Help Request":"bd-r","Damage Report":"bd-o","Information":"bd-b"}
    return f'<span class="badge {m.get(c,"bd-x")}">{c}</span>'

def _pri_html(p: str) -> str:
    m = {"HIGHEST":"bd-r","HIGH":"bd-o","CRITICAL_REVIEW":"bd-y","MEDIUM":"bd-b","LOW":"bd-x"}
    label = p.replace("_"," ")
    return f'<span class="badge {m.get(p,"bd-x")}">{label}</span>'

def _conf_html(c: str) -> str:
    m = {"high":"bd-g","medium":"bd-y","low":"bd-o","none":"bd-x"}
    return f'<span class="badge {m.get(c,"bd-x")}">{c}</span>'

def _card_cls(score: float, loc: str) -> str:
    if score > 70:
        return "tcard tc-crit" if loc == "unknown" else "tcard tc-high"
    if score >= 40: return "tcard tc-med"
    return "tcard tc-low"

def render_card(row) -> str:
    score = float(row.get("urgency_score", 0))
    loc   = str(row.get("location", "unknown"))
    src   = str(row.get("location_source","none"))
    conf  = str(row.get("location_confidence","none"))
    text  = str(row.get("content",""))[:420]
    date  = str(row.get("date",""))[:16].replace("T"," ")
    uname = str(row.get("username",""))
    cat   = str(row.get("category",""))
    pri   = str(row.get("priority",""))

    crit = '<span class="crit-f">&#9888; UNKNOWN LOCATION &mdash; CRITICAL REVIEW</span>' \
           if score > 70 and loc == "unknown" else ""
    icon = "&#128205;" if src not in ("none","") else "&#10067;"
    has_coord = row.get("lat") not in (None,"","nan") and row.get("lon") not in (None,"","nan")
    coord_badge = f'<span class="badge bd-g">&#10003; geocoded</span>' if has_coord else ""

    return f"""<div class="{_card_cls(score,loc)}">
  <div class="tmeta">{_score_html(score)}{_cat_html(cat)}{_pri_html(pri)}{crit}
    <span class="umeta">@{uname} &middot; {date}</span></div>
  <div class="ttext">{text}</div>
  <div class="tfoot"><span class="loctext">{icon} {loc}</span>
    <span class="badge bd-x">{src}</span>{_conf_html(conf)}{coord_badge}</div>
</div>"""


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;
        color:#ff2d2d;letter-spacing:0.15em;text-transform:uppercase;
        padding-bottom:0.6rem;margin-bottom:1rem;border-bottom:1px solid #1e2530;">
        &#9670; Control Panel</div>""", unsafe_allow_html=True)

    search_q     = st.text_input("Search", placeholder="flood, rescue, Houston...")
    urgency_rng  = st.slider("Urgency score", 0, 100, (0, 100), step=5)
    cat_filter   = st.multiselect("Category",
                    ["Help Request","Damage Report","Information"],
                    default=["Help Request","Damage Report","Information"])
    loc_filter   = st.text_input("Location contains", placeholder="Mumbai, Texas...")
    src_filter   = st.multiselect("Location source",
                    ["tweet","hashtag","profile_location","bio","none"],
                    default=["tweet","hashtag","profile_location","bio","none"])
    sort_col     = st.selectbox("Sort by", ["urgency_score","date","priority"])
    sort_asc     = st.checkbox("Ascending", value=False)

    st.markdown("---")
    st.markdown("<div style='font-family:monospace;font-size:0.62rem;color:#8d9db0;margin-bottom:0.4rem;'>DATA REFRESH</div>", unsafe_allow_html=True)

    # ── Refresh button — triggers background pipeline via API ─────────────
    if st.button("⟳  Fetch Live Data", use_container_width=True):
        pst = get_pipeline_status()
        if pst.get("running"):
            st.session_state["pipeline_msg"] = "⏳ Pipeline already running..."
        else:
            data, err = api_post("/refresh", {"force_scrape": True}, timeout=5)
            if data:
                st.session_state["pipeline_msg"] = "✅ Pipeline started! Data will appear in ~30s"
                st.session_state["last_refresh"]  = time.time()
                st.cache_data.clear()
            else:
                # Fallback: run locally if backend is offline
                st.session_state["pipeline_msg"] = "⚠ Backend offline — running local pipeline..."
                try:
                    from utils.scraper import scrape_tweets, _generate_synthetic_tweets
                    from utils.preprocessor import preprocess_dataframe
                    from utils.inference import get_model_manager
                    from utils.location_extractor import extract_location, get_priority_label, load_spacy_model

                    with st.spinner("Running local pipeline..."):
                        try:
                            raw = scrape_tweets(force_refresh=True)
                        except Exception:
                            raw = _generate_synthetic_tweets()
                        df2 = pd.DataFrame(raw)
                        df2 = preprocess_dataframe(df2)
                        mgr2 = get_model_manager()
                        preds2 = mgr2.predict_batch(df2["clean_text"].tolist())
                        df2["category"]      = [p["category"]      for p in preds2]
                        df2["urgency_score"] = [p["urgency_score"]  for p in preds2]
                        nlp2 = load_spacy_model()
                        locs2,srcs2,confs2,pris2 = [],[],[],[]
                        for _,row2 in df2.iterrows():
                            l2 = extract_location(
                                tweet_text=row2.get("content",""),
                                hashtags_str=row2.get("hashtags_str",""),
                                profile_location=row2.get("user_location",""),
                                user_bio=row2.get("user_bio",""), nlp=nlp2)
                            locs2.append(l2["location"]); srcs2.append(l2["location_source"])
                            confs2.append(l2["location_confidence"])
                            pris2.append(get_priority_label(row2["urgency_score"],l2["location_source"]))
                        df2["location"]=locs2; df2["location_source"]=srcs2
                        df2["location_confidence"]=confs2; df2["priority"]=pris2
                        df2["lat"]=None; df2["lon"]=None
                        DATA_DIR.mkdir(exist_ok=True)
                        df2.to_csv(CSV_PATH, index=False)
                    st.session_state["pipeline_msg"] = f"✅ Local pipeline done — {len(df2)} tweets loaded"
                    st.cache_data.clear()
                except Exception as ex:
                    st.session_state["pipeline_msg"] = f"❌ Pipeline failed: {ex}"

    st.session_state["auto_refresh"] = st.checkbox(
        "Auto-refresh (30s)", value=st.session_state["auto_refresh"])

    if st.session_state["pipeline_msg"]:
        color = "#00e676" if "✅" in st.session_state["pipeline_msg"] else \
                "#ff7a00" if "⚠" in st.session_state["pipeline_msg"] else "#ff2d2d"
        st.markdown(f'<div style="font-family:monospace;font-size:0.68rem;'
                    f'color:{color};margin-top:0.4rem;padding:0.5rem;'
                    f'background:#111419;border-radius:4px;">'
                    f'{st.session_state["pipeline_msg"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline status
    pst = get_pipeline_status()
    pst_color  = "#00b4ff" if pst.get("running") else \
                 "#ff2d2d" if pst.get("last_error") else "#00e676"
    pst_label  = "RUNNING" if pst.get("running") else \
                 "ERROR"   if pst.get("last_error") else "IDLE"
    pst_detail = pst.get("last_error","")[:60] if pst.get("last_error") else \
                 pst.get("last_run","Never") or "Never"

    # API health
    health    = get_api_health()
    api_color = "#00e676" if health.get("status") == "ok" else "#ff2d2d"

    st.markdown(f"""
    <div style="background:#0a0c10;border:1px solid #1e2530;border-radius:4px;padding:0.7rem;">
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                  color:#8d9db0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">
        System Status</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;margin-bottom:0.3rem;">
        API <span style="color:{api_color};">&#9679; {health.get('status','offline').upper()}</span>
        &nbsp;&middot;&nbsp;
        <span style="color:#8d9db0;font-size:0.6rem;">{health.get('model_mode','?')}</span>
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;">
        Pipeline <span style="color:{pst_color};">&#9679; {pst_label}</span>
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;
                  color:#8d9db0;margin-top:0.2rem;">{pst_detail}</div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="hdr">
  <div class="hdr-title">Crisis <span style="color:#ff2d2d;">Intelligence</span> System</div>
  <div class="hdr-sub">
    <span style="width:7px;height:7px;background:#00e676;border-radius:50%;
                 display:inline-block;margin-right:6px;"></span>
    AI Disaster Tweet Analyzer &nbsp;&middot;&nbsp; Real-time Emergency Response Intelligence
  </div>
</div>""", unsafe_allow_html=True)

# Auto-refresh logic (no time.sleep — uses last_refresh timestamp)
if st.session_state["auto_refresh"]:
    elapsed = time.time() - st.session_state["last_refresh"]
    if elapsed > 30:
        st.cache_data.clear()
        st.session_state["last_refresh"] = time.time()
        st.rerun()

# Load data
with st.spinner("Loading data..."):
    df = load_data()

if df.empty:
    st.markdown("""<div class="info-box">
        &#9432; No data loaded yet. Click <strong>Fetch Live Data</strong> in the sidebar
        to start the pipeline. It will scrape tweets, classify them with AI,
        extract locations, and geocode them automatically.
    </div>""", unsafe_allow_html=True)

    # Show a button here too for visibility
    if st.button("▶  Run Pipeline Now", type="primary"):
        api_post("/refresh", {}, timeout=5)
        st.session_state["pipeline_msg"] = "Pipeline started — refresh in 30 seconds"
        st.rerun()
    st.stop()

# Metrics
total     = len(df)
high_urg  = int((pd.to_numeric(df.get("urgency_score", pd.Series(dtype=float)), errors="coerce") > 70).sum())
unk_loc   = int((df.get("location", pd.Series(dtype=str)) == "unknown").sum())
crit_rev  = int(((pd.to_numeric(df.get("urgency_score", pd.Series(dtype=float)), errors="coerce") > 70) &
                  (df.get("location", pd.Series(dtype=str)) == "unknown")).sum())

st.markdown(f"""
<div class="mgrid">
  <div class="mcard mc-blue"><div class="mlbl">Total Tweets</div><div class="mval">{total}</div></div>
  <div class="mcard mc-red"><div class="mlbl">High Urgency</div><div class="mval">{high_urg}</div></div>
  <div class="mcard mc-orange"><div class="mlbl">Unknown Location</div><div class="mval">{unk_loc}</div></div>
  <div class="mcard mc-yellow"><div class="mlbl">Critical Review</div><div class="mval">{crit_rev}</div></div>
</div>""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📡  LIVE FEED", "📊  ANALYTICS", "🗺  MAP", "🔬  API TESTER"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE FEED
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    filt = df.copy()
    if "urgency_score" in filt.columns:
        filt["urgency_score"] = pd.to_numeric(filt["urgency_score"], errors="coerce").fillna(0)
        filt = filt[(filt["urgency_score"] >= urgency_rng[0]) & (filt["urgency_score"] <= urgency_rng[1])]
    if cat_filter and "category" in filt.columns:
        filt = filt[filt["category"].isin(cat_filter)]
    if src_filter and "location_source" in filt.columns:
        filt = filt[filt["location_source"].isin(src_filter)]
    if search_q and "content" in filt.columns:
        filt = filt[filt["content"].str.contains(search_q, case=False, na=False)]
    if loc_filter and "location" in filt.columns:
        filt = filt[filt["location"].str.contains(loc_filter, case=False, na=False)]
    if sort_col in filt.columns:
        filt = filt.sort_values(sort_col, ascending=sort_asc)

    # Status bar
    age_str = "unknown"
    if CSV_PATH.exists():
        age_s = time.time() - CSV_PATH.stat().st_mtime
        age_str = f"{int(age_s)}s ago" if age_s < 60 else \
                  f"{int(age_s/60)}m ago" if age_s < 3600 else f"{int(age_s/3600)}h ago"

    geocoded_count = int(df["lat"].notna().sum()) if "lat" in df.columns else 0

    st.markdown(f"""<div class="sbar">
      <span>Showing <strong style="color:#e8eaed">{len(filt)}</strong> / {total} tweets</span>
      <span style="display:flex;gap:0.8rem;align-items:center;">
        <span>Updated: <span style="color:#00e676;">{age_str}</span></span>
        <span>Geocoded: <span style="color:#00b4ff;">{geocoded_count}/{total}</span></span>
      </span>
    </div>""", unsafe_allow_html=True)

    if filt.empty:
        st.markdown('<div class="no-data">No tweets match the current filters.</div>',
                    unsafe_allow_html=True)
    else:
        # Render all cards in ONE markdown call (fast)
        html = "\n".join(render_card(row) for _, row in filt.head(100).iterrows())
        st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        BG   = "#0a0c10"
        GRID = "#1e2530"
        TC   = "#8d9db0"
        FONT = "Share Tech Mono, monospace"

        c1, c2 = st.columns(2)
        with c1:
            if "category" in df.columns:
                cc = df["category"].value_counts().reset_index()
                cc.columns = ["category","count"]
                fig = px.bar(cc, x="category", y="count", color="category",
                    color_discrete_map={"Help Request":"#ff2d2d",
                                        "Damage Report":"#ff7a00","Information":"#00b4ff"},
                    title="TWEET CATEGORIES")
                fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, showlegend=False,
                    font=dict(family=FONT,color=TC,size=11),
                    title_font=dict(size=11,color="#e8eaed"),
                    xaxis=dict(gridcolor=GRID,linecolor=GRID),
                    yaxis=dict(gridcolor=GRID,linecolor=GRID),
                    margin=dict(l=5,r=5,t=35,b=5))
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "urgency_score" in df.columns:
                scores = pd.to_numeric(df["urgency_score"], errors="coerce").dropna()
                fig = go.Figure(go.Histogram(x=scores, nbinsx=20,
                    marker=dict(color=scores,
                        colorscale=[[0,"#00e676"],[0.4,"#ff7a00"],[1,"#ff2d2d"]])))
                fig.add_vline(x=70,line_dash="dash",line_color="#ff2d2d",
                              annotation_text="HIGH",annotation_font_color="#ff2d2d")
                fig.add_vline(x=40,line_dash="dash",line_color="#ff7a00",
                              annotation_text="MED",annotation_font_color="#ff7a00")
                fig.update_layout(plot_bgcolor=BG,paper_bgcolor=BG,showlegend=False,
                    title=dict(text="URGENCY DISTRIBUTION",
                               font=dict(size=11,color="#e8eaed")),
                    font=dict(family=FONT,color=TC,size=11),
                    xaxis=dict(gridcolor=GRID,linecolor=GRID),
                    yaxis=dict(gridcolor=GRID,linecolor=GRID),
                    margin=dict(l=5,r=5,t=35,b=5))
                st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if "location_source" in df.columns:
                sc = df["location_source"].value_counts().reset_index()
                sc.columns = ["source","count"]
                fig = px.pie(sc, values="count", names="source", hole=0.55,
                    title="LOCATION SOURCES",
                    color_discrete_sequence=["#00b4ff","#00e676","#ff7a00","#ff2d2d","#8d9db0"])
                fig.update_layout(plot_bgcolor=BG,paper_bgcolor=BG,
                    title_font=dict(size=11,color="#e8eaed"),
                    font=dict(family=FONT,color=TC,size=11),
                    legend=dict(font=dict(color=TC)),
                    margin=dict(l=5,r=5,t=35,b=5))
                st.plotly_chart(fig, use_container_width=True)

        with c4:
            if "priority" in df.columns:
                pc = df["priority"].value_counts().reset_index()
                pc.columns = ["priority","count"]
                fig = px.bar(pc, x="count", y="priority", orientation="h",
                    color="priority",
                    color_discrete_map={"HIGHEST":"#ff2d2d","HIGH":"#ff7a00",
                        "CRITICAL_REVIEW":"#ffd600","MEDIUM":"#00b4ff","LOW":"#8d9db0"},
                    title="PRIORITY DISTRIBUTION")
                fig.update_layout(plot_bgcolor=BG,paper_bgcolor=BG,showlegend=False,
                    title_font=dict(size=11,color="#e8eaed"),
                    font=dict(family=FONT,color=TC,size=11),
                    xaxis=dict(gridcolor=GRID),yaxis=dict(gridcolor=GRID),
                    margin=dict(l=5,r=5,t=35,b=5))
                st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.error("Plotly not installed. Add plotly to requirements.txt")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MAP  (uses coordinates already in CSV from backend pipeline)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
        color:#8d9db0;margin-bottom:0.6rem;">
        &#128205; Map shows tweets with resolved coordinates.
        Coordinates are geocoded by the backend pipeline via OpenStreetMap (Nominatim).
        Click <strong style="color:#00b4ff;">Fetch Live Data</strong> to refresh.
    </div>""", unsafe_allow_html=True)

    if "lat" not in df.columns or "lon" not in df.columns:
        st.markdown("""<div class="info-box">
            No coordinate columns found in data. Run pipeline via
            <strong>Fetch Live Data</strong> to geocode locations.
        </div>""", unsafe_allow_html=True)
    else:
        # Build map dataframe from pre-geocoded data
        map_df = df.copy()
        map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
        map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
        map_df = map_df[map_df["lat"].notna() & map_df["lon"].notna()].copy()
        map_df["urgency_score"] = pd.to_numeric(map_df["urgency_score"], errors="coerce").fillna(0)
        map_df["size"]          = map_df["urgency_score"].clip(lower=5)
        map_df["text_preview"]  = map_df["content"].str[:100]

        total_known    = len(df[df["location"] != "unknown"]) if "location" in df.columns else 0
        total_geocoded = len(map_df)
        total_unknown  = int((df.get("location","") == "unknown").sum())

        # Stats row
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total tweets",    total)
        col_s2.metric("Known location",  total_known)
        col_s3.metric("Geocoded (map)",  total_geocoded)
        col_s4.metric("Unknown location",total_unknown)

        if map_df.empty:
            st.markdown("""<div class="info-box">
                No geocoded tweets yet. The backend geocodes locations during the pipeline run.
                Click <strong>Fetch Live Data</strong> in the sidebar to trigger geocoding.
            </div>""", unsafe_allow_html=True)

            # Show table of known-but-not-geocoded locations
            if "location" in df.columns:
                known = df[(df["location"] != "unknown") & (df["location"] != "")]
                if not known.empty:
                    st.markdown("**Locations extracted (not yet geocoded):**")
                    st.dataframe(
                        known[["content","location","location_source","urgency_score"]].head(20),
                        use_container_width=True
                    )
        else:
            try:
                import plotly.express as px
                fig = px.scatter_mapbox(
                    map_df,
                    lat="lat", lon="lon",
                    color="urgency_score",
                    size="size",
                    color_continuous_scale=[[0,"#00e676"],[0.4,"#ff7a00"],[1,"#ff2d2d"]],
                    range_color=[0, 100],
                    size_max=35,
                    hover_name="location",
                    hover_data={
                        "category":      True,
                        "urgency_score": ":.1f",
                        "priority":      True,
                        "text_preview":  True,
                        "lat":           False,
                        "lon":           False,
                        "size":          False,
                    },
                    mapbox_style="carto-darkmatter",
                    zoom=1.5,
                    center={"lat": 20, "lon": 0},
                    title=f"DISASTER TWEET MAP — {total_geocoded} geocoded locations",
                )
                fig.update_layout(
                    paper_bgcolor="#0a0c10",
                    font=dict(family="Share Tech Mono,monospace", color="#8d9db0"),
                    title_font=dict(size=11, color="#e8eaed"),
                    coloraxis_colorbar=dict(
                        title="Urgency",
                        tickfont=dict(color="#8d9db0"),
                        titlefont=dict(color="#8d9db0"),
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=520,
                )
                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                st.error("Plotly not available.")
            except Exception as e:
                st.error(f"Map error: {e}")
                # Fallback table
                st.dataframe(
                    map_df[["location","lat","lon","urgency_score","category","priority"]],
                    use_container_width=True
                )

        # Not-geocoded list
        if "location" in df.columns:
            failed_locs = df[(df["location"] != "unknown") &
                             (df["location"] != "") &
                             (df["lat"].isna() if "lat" in df.columns else True)]["location"].unique()
            if len(failed_locs) > 0:
                with st.expander(f"Locations not yet geocoded ({len(failed_locs)})"):
                    st.write(", ".join(str(l) for l in failed_locs[:30]))
                    st.caption("These will be resolved on the next pipeline run.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — API TESTER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
        color:#8d9db0;margin-bottom:1rem;">
        Live test the /predict endpoint with any tweet text.
        Falls back to local inference if backend is offline.
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])

    with col_a:
        test_text = st.text_area("Tweet text",
            value="URGENT: Flooding in Houston TX. Families trapped on rooftops. Need rescue boats NOW! #HoustonFlood",
            height=90)
        test_bio  = st.text_input("User bio (optional)", value="Texas resident | Emergency volunteer")
        test_ploc = st.text_input("Profile location (optional)", value="Houston, TX")

        if st.button("▶  Analyze Tweet"):
            with st.spinner("Analyzing..."):
                result, err = api_post("/predict", {
                    "text": test_text,
                    "user_bio": test_bio,
                    "profile_location": test_ploc,
                }, timeout=12)

                if not result:
                    # Local fallback
                    try:
                        from utils.inference import get_model_manager as _gmm
                        from utils.location_extractor import extract_location as _el, \
                            get_priority_label as _gpl, load_spacy_model as _lsm
                        from utils.preprocessor import clean_text as _ct
                        _mgr  = _gmm()
                        _nlp2 = _lsm()
                        _pred = _mgr.predict_single(_ct(test_text))
                        _htags= " ".join(re.findall(r"#(\w+)", test_text))
                        _loc  = _el(test_text, _htags, test_ploc, test_bio, _nlp2)
                        result = {
                            "category":            _pred["category"],
                            "urgency_score":       _pred["urgency_score"],
                            "location":            _loc["location"],
                            "location_source":     _loc["location_source"],
                            "location_confidence": _loc["location_confidence"],
                            "priority":            _gpl(_pred["urgency_score"], _loc["location_source"]),
                            "all_probs":           _pred["all_probs"],
                            "model_mode":          "local fallback",
                            "lat": None, "lon": None,
                        }
                    except Exception as ex:
                        st.error(f"Both API and local inference failed: {ex}")
                        result = None

                if result:
                    sc    = float(result.get("urgency_score", 0))
                    scol  = "#ff2d2d" if sc>70 else "#ff7a00" if sc>=40 else "#00e676"
                    coord_info = ""
                    if result.get("lat") and result.get("lon"):
                        coord_info = f'<div><div style="font-size:0.58rem;color:#8d9db0;">COORDINATES</div><div style="font-family:monospace;color:#00e676;">{result["lat"]:.4f}, {result["lon"]:.4f}</div></div>'

                    probs_html = "".join(
                        f'<div><div style="font-size:0.58rem;color:#8d9db0;">{k}</div>'
                        f'<div style="font-family:monospace;color:#e8eaed;">{v:.1%}</div></div>'
                        for k,v in result.get("all_probs",{}).items()
                    )
                    st.markdown(f"""
<div style="background:#111419;border:1px solid #1e2530;border-left:3px solid {scol};
            padding:1rem;margin-top:0.7rem;border-radius:0 4px 4px 0;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.58rem;color:#8d9db0;
              text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">
    Analysis Result &mdash; <span style="color:#8d9db0;">{result.get('model_mode','')}</span></div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.7rem;margin-bottom:0.7rem;">
    <div><div style="font-size:0.58rem;color:#8d9db0;">CATEGORY</div>
         <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                     font-weight:700;color:{scol};">{result.get('category','')}</div></div>
    <div><div style="font-size:0.58rem;color:#8d9db0;">URGENCY</div>
         <div style="font-family:monospace;font-size:1.8rem;font-weight:bold;color:{scol};">{sc:.1f}</div></div>
    <div><div style="font-size:0.58rem;color:#8d9db0;">PRIORITY</div>
         <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                     font-weight:700;color:#ffd600;">{result.get('priority','').replace('_',' ')}</div></div>
    <div><div style="font-size:0.58rem;color:#8d9db0;">LOCATION</div>
         <div style="font-family:monospace;color:#00b4ff;">{result.get('location','')}</div></div>
    <div><div style="font-size:0.58rem;color:#8d9db0;">SOURCE</div>
         <div style="font-family:monospace;color:#e8eaed;">{result.get('location_source','')}</div></div>
    <div><div style="font-size:0.58rem;color:#8d9db0;">CONFIDENCE</div>
         <div style="font-family:monospace;color:#e8eaed;">{result.get('location_confidence','')}</div></div>
    {coord_info}
  </div>
  <div style="border-top:1px solid #1e2530;padding-top:0.5rem;">
    <div style="font-size:0.58rem;color:#8d9db0;margin-bottom:0.3rem;">PROBABILITY BREAKDOWN</div>
    <div style="display:flex;gap:1.2rem;">{probs_html}</div>
  </div>
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
<div style="background:#111419;border:1px solid #1e2530;padding:1rem;border-radius:4px;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#8d9db0;
              text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">
    Priority Reference</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.66rem;line-height:1.9;color:#e8eaed;">
    <span style="color:#ff2d2d;">HIGHEST</span><br>
    Score &gt;70 + tweet NER location<br><br>
    <span style="color:#ff7a00;">HIGH</span><br>
    Score &gt;70 + hashtag/profile<br><br>
    <span style="color:#ffd600;">CRITICAL REVIEW</span><br>
    Score &gt;70 + unknown location<br><br>
    <span style="color:#00b4ff;">MEDIUM</span><br>
    Score 40&ndash;70<br><br>
    <span style="color:#8d9db0;">LOW</span><br>
    Score &lt;40
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("""
<div style="background:#111419;border:1px solid #1e2530;padding:1rem;
            border-radius:4px;margin-top:0.7rem;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#8d9db0;
              text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">
    API Endpoints</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;
              color:#00b4ff;line-height:2;">
    POST /predict<br>POST /batch<br>POST /refresh<br>
    GET  /tweets<br>GET  /stats<br>GET  /status<br>GET  /health
  </div>
</div>""", unsafe_allow_html=True)
