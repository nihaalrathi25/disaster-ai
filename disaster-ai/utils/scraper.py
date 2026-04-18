"""
scraper.py — Real disaster tweet/report fetcher using multiple free sources
Sources tried in order:
  1. Nitter RSS (free Twitter mirror, no auth)
  2. GDACS RSS (UN Global Disaster Alert, free)
  3. ReliefWeb API (UN OCHA, free, no auth)
  4. snscrape (if installed)
  5. Synthetic fallback (always works)
"""

import csv
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "date", "tweet_id", "content", "username",
    "user_bio", "user_location", "like_count", "retweet_count", "query_used",
]

NITTER_INSTANCES = [
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.1d4.us",
    "https://nitter.kavin.rocks",
    "https://nitter.unixfox.eu",
]

SEARCH_QUERIES = [
    "flood help rescue",
    "earthquake urgent rescue",
    "disaster emergency help",
    "hurricane rescue needed",
    "wildfire evacuate",
    "cyclone flood relief",
]

GDACS_RSS = "https://www.gdacs.org/xml/rss.xml"
RELIEFWEB_API = "https://api.reliefweb.int/v1/reports"


def _fetch_nitter_rss(query: str, instance: str, max_results: int = 20) -> List[Dict]:
    results = []
    try:
        encoded = requests.utils.quote(query)
        url = f"{instance}/search/rss?q={encoded}&f=tweets"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        for item in items[:max_results]:
            title = item.findtext("title", "")
            desc = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            creator = item.findtext("{http://purl.org/dc/elements/1.1/}creator", "unknown")
            text = re.sub(r"<[^>]+>", " ", desc or title)
            text = re.sub(r"\s+", " ", text).strip()
            if not text or len(text) < 10:
                continue
            try:
                dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
                date_str = dt.isoformat()
            except Exception:
                date_str = datetime.utcnow().isoformat()
            tweet_id = hashlib.md5(f"{link}{text[:30]}".encode()).hexdigest()[:16]
            results.append({
                "date": date_str,
                "tweet_id": tweet_id,
                "content": text[:500],
                "username": creator.lstrip("@"),
                "user_bio": "",
                "user_location": "",
                "like_count": 0,
                "retweet_count": 0,
                "query_used": query,
            })
        logger.info(f"Nitter ({instance}): {len(results)} results for '{query}'")
    except Exception as e:
        logger.debug(f"Nitter {instance} failed: {e}")
    return results


def fetch_from_nitter(queries: List[str], max_per_query: int = 20) -> List[Dict]:
    all_results = []
    for query in queries:
        for instance in NITTER_INSTANCES:
            results = _fetch_nitter_rss(query, instance, max_per_query)
            if results:
                all_results.extend(results)
                break
            time.sleep(0.3)
    return all_results


def fetch_from_gdacs() -> List[Dict]:
    results = []
    try:
        resp = requests.get(GDACS_RSS, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        for item in items[:30]:
            title = item.findtext("title", "")
            desc = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            lat = item.findtext("{http://www.w3.org/2003/01/geo/wgs84_pos#}lat", "")
            lon = item.findtext("{http://www.w3.org/2003/01/geo/wgs84_pos#}long", "")
            text = re.sub(r"<[^>]+>", " ", f"{title}. {desc}")
            text = re.sub(r"\s+", " ", text).strip()[:500]
            try:
                dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
                date_str = dt.isoformat()
            except Exception:
                date_str = datetime.utcnow().isoformat()
            tweet_id = hashlib.md5(link.encode()).hexdigest()[:16]
            results.append({
                "date": date_str,
                "tweet_id": tweet_id,
                "content": text,
                "username": "GDACS_Alert",
                "user_bio": "Global Disaster Alert and Coordination System (UN)",
                "user_location": f"{lat},{lon}" if lat and lon else "",
                "like_count": 0,
                "retweet_count": 0,
                "query_used": "gdacs_feed",
            })
        logger.info(f"GDACS: {len(results)} disaster events")
    except Exception as e:
        logger.warning(f"GDACS failed: {e}")
    return results


def fetch_from_reliefweb(limit: int = 20) -> List[Dict]:
    results = []
    try:
        payload = {
            "limit": limit,
            "sort": ["date:desc"],
            "fields": {"include": ["title", "body", "date", "country", "source"]},
            "filter": {"field": "type", "value": ["Situation Report", "News and Press Release"]}
        }
        resp = requests.post(RELIEFWEB_API, json=payload, timeout=10,
                             headers={"User-Agent": "DisasterAI/1.0"})
        if resp.status_code != 200:
            return []
        data = resp.json()
        for item in data.get("data", []):
            fields = item.get("fields", {})
            title = fields.get("title", "")
            body = fields.get("body", "")[:300]
            date_info = fields.get("date", {})
            country = fields.get("country", [{}])
            source = fields.get("source", [{}])
            text = re.sub(r"\s+", " ", f"{title}. {body}").strip()[:500]
            date_str = date_info.get("created", datetime.utcnow().isoformat())
            location = country[0].get("name", "") if country else ""
            username = source[0].get("name", "ReliefWeb").replace(" ", "_") if source else "ReliefWeb"
            tweet_id = hashlib.md5(f"{title}{date_str}".encode()).hexdigest()[:16]
            results.append({
                "date": date_str,
                "tweet_id": tweet_id,
                "content": text,
                "username": username[:30],
                "user_bio": "UN ReliefWeb disaster report",
                "user_location": location,
                "like_count": 0,
                "retweet_count": 0,
                "query_used": "reliefweb_api",
            })
        logger.info(f"ReliefWeb: {len(results)} reports")
    except Exception as e:
        logger.warning(f"ReliefWeb failed: {e}")
    return results


def fetch_from_snscrape(queries: List[str], max_per_query: int = 30) -> List[Dict]:
    results = []
    try:
        import snscrape.modules.twitter as sntwitter
        since = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
        for query in queries:
            count = 0
            try:
                for tweet in sntwitter.TwitterSearchScraper(f"{query} since:{since} lang:en").get_items():
                    if count >= max_per_query:
                        break
                    results.append({
                        "date": tweet.date.isoformat(),
                        "tweet_id": str(tweet.id),
                        "content": getattr(tweet, "rawContent", tweet.content) or "",
                        "username": tweet.user.username if tweet.user else "",
                        "user_bio": (tweet.user.description or "") if tweet.user else "",
                        "user_location": (tweet.user.location or "") if tweet.user else "",
                        "like_count": tweet.likeCount or 0,
                        "retweet_count": tweet.retweetCount or 0,
                        "query_used": query,
                    })
                    count += 1
            except Exception as e:
                logger.debug(f"snscrape query '{query}' failed: {e}")
        logger.info(f"snscrape: {len(results)} tweets")
    except ImportError:
        pass
    return results


def scrape_tweets(
    queries: List[str] = SEARCH_QUERIES,
    max_per_query: int = 20,
    output_file: Optional[Path] = None,
    use_cache_minutes: int = 5,
    force_refresh: bool = False,
) -> List[Dict]:
    """
    Fetch real disaster tweets from multiple free sources.
    Caches results for `use_cache_minutes` to avoid hammering APIs.
    Set force_refresh=True to bypass cache.
    """
    output_file = output_file or DATA_DIR / "raw_tweets.csv"

    # Cache check
    if not force_refresh and output_file.exists():
        age_minutes = (time.time() - output_file.stat().st_mtime) / 60
        if age_minutes < use_cache_minutes:
            logger.info(f"Cache fresh ({age_minutes:.1f}m old), skipping fetch")
            import pandas as pd
            return pd.read_csv(output_file).fillna("").to_dict(orient="records")

    all_results = []

    logger.info("=== Fetching from Nitter RSS ===")
    all_results.extend(fetch_from_nitter(queries[:4], max_per_query))

    logger.info("=== Fetching from GDACS ===")
    all_results.extend(fetch_from_gdacs())

    logger.info("=== Fetching from ReliefWeb ===")
    all_results.extend(fetch_from_reliefweb(20))

    logger.info("=== Trying snscrape ===")
    all_results.extend(fetch_from_snscrape(queries[:3], max_per_query))

    if len(all_results) < 5:
        logger.warning("Real sources insufficient — using synthetic dataset")
        all_results = _generate_synthetic_tweets()

    # Deduplicate
    seen, deduped = set(), []
    for r in all_results:
        tid = str(r.get("tweet_id", ""))
        if tid and tid not in seen:
            seen.add(tid)
            deduped.append(r)

    # Keyword filter for disaster relevance
    DISASTER_KEYWORDS = [
        "flood", "earthquake", "hurricane", "cyclone", "tornado", "wildfire",
        "disaster", "rescue", "emergency", "evacuate", "tsunami", "landslide",
        "storm", "relief", "trapped", "damage", "collapse", "help", "urgent",
        "missing", "injured", "casualt", "victim", "displaced", "crisis",
    ]
    filtered = [r for r in deduped if any(
        kw in (r.get("content", "") + r.get("user_bio", "")).lower()
        for kw in DISASTER_KEYWORDS
    )]
    if len(filtered) < 5:
        filtered = deduped

    _save_to_csv(filtered, output_file)
    logger.info(f"Saved {len(filtered)} disaster records")
    return filtered


def _save_to_csv(tweets: List[Dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for t in tweets:
            writer.writerow({k: str(t.get(k, ""))[:1000] for k in FIELDNAMES})


def _generate_synthetic_tweets() -> List[Dict]:
    import random
    random.seed(int(time.time()) // 300)
    samples = [
        {"content": "URGENT: Massive flooding in downtown Houston. People trapped on rooftops! Send rescue boats NOW #HoustonFlood #rescue", "user_bio": "Emergency responder | Houston TX", "user_location": "Houston, TX", "query_used": "flood help"},
        {"content": "6.8 magnitude earthquake hit near Los Angeles. Buildings collapsing in Santa Monica. Need immediate rescue #earthquake #LA", "user_bio": "Journalist | West Coast", "user_location": "Los Angeles, CA", "query_used": "earthquake urgent"},
        {"content": "Multiple families trapped under rubble in Nepal earthquake zone. Rescue teams needed urgently #NepalEarthquake", "user_bio": "Aid worker | South Asia", "user_location": "Kathmandu", "query_used": "rescue needed"},
        {"content": "Wildfire spreading rapidly near Paradise CA. Hundreds of homes destroyed. Residents fleeing #CaliforniaFire #evacuate", "user_bio": "Fire safety volunteer", "user_location": "Northern California", "query_used": "wildfire evacuate"},
        {"content": "HELP! trapped under collapsed building in Hatay after earthquake. phone dying. please find us", "user_bio": "Student | Hatay University", "user_location": "", "query_used": "rescue needed"},
        {"content": "Flash flood warning active for Memphis area. Do not drive through flooded roads #memphisflood", "user_bio": "National Weather Service", "user_location": "Memphis, TN", "query_used": "flood help"},
        {"content": "HURRICANE WARNING: Category 4 storm approaching Tampa Bay. All residents MUST evacuate Zone A NOW #Tampa", "user_bio": "Hillsborough County Emergency Management", "user_location": "Tampa, FL", "query_used": "hurricane emergency"},
        {"content": "Earthquake damage: downtown Izmir - 40+ buildings collapsed, hospitals overwhelmed. International aid needed", "user_bio": "Crisis journalist | Mediterranean", "user_location": "Izmir, Turkey", "query_used": "earthquake urgent"},
        {"content": "Rescue operations across 6 districts in Assam flood zone. Need rubber boats urgently", "user_bio": "Assam State Disaster Management", "user_location": "Guwahati, Assam", "query_used": "flood help"},
        {"content": "Cyclone making landfall near Gujarat coast. Storm surge 4-5m expected. Immediate evacuation needed", "user_bio": "IMD meteorologist", "user_location": "Ahmedabad, India", "query_used": "disaster relief needed"},
        {"content": "People need rescue in Jakarta flood zones NOW. Water at chest level. Emergency! #Jakarta", "user_bio": "Community leader | East Jakarta", "user_location": "Jakarta", "query_used": "rescue needed"},
        {"content": "Relief supplies dispatched to tornado hit areas in Oklahoma. Shelter across 5 counties", "user_bio": "FEMA spokesperson", "user_location": "Oklahoma City, OK", "query_used": "disaster relief needed"},
        {"content": "Bangladesh floods: death toll 47, thousands displaced. UN requesting emergency relief funding", "user_bio": "UN disaster relief officer", "user_location": "Dhaka, Bangladesh", "query_used": "flood help"},
        {"content": "Gas lines rupturing across San Francisco after 6.0 quake. Stay away from downtown #SFEarthquake", "user_bio": "SF Dept of Emergency Management", "user_location": "San Francisco, CA", "query_used": "earthquake urgent"},
        {"content": "Bridge collapsed in Miami due to hurricane. People stranded. Aerial rescue needed #HurricaneMiami", "user_bio": "Traffic reporter | South Florida", "user_location": "Miami, FL", "query_used": "hurricane emergency"},
        {"content": "Red Cross shelter at Lincoln High School for flood victims. Free meals and bedding available.", "user_bio": "Red Cross volunteer", "user_location": "Portland, OR", "query_used": "disaster relief needed"},
        {"content": "Landslide blocked highway in Philippines after heavy rains. Vehicles buried. Emergency crews on scene", "user_bio": "Philippine disaster bureau", "user_location": "Manila, Philippines", "query_used": "disaster relief needed"},
        {"content": "EARTHQUAKE! 7.2 magnitude near Anchorage Alaska. Tsunami warning for coastal areas. Evacuate!", "user_bio": "USGS seismologist", "user_location": "Anchorage, AK", "query_used": "earthquake urgent"},
        {"content": "3 hospitals offline, 12 schools damaged, water plant flooded in Port-au-Prince. Crisis maximum", "user_bio": "Haiti infrastructure NGO", "user_location": "Port-au-Prince, Haiti", "query_used": "earthquake urgent"},
        {"content": "Tornado near Dallas. Search and rescue underway. Dozens missing #DallasTornado", "user_bio": "Dallas emergency services", "user_location": "Dallas, TX", "query_used": "disaster relief needed"},
        {"content": "SOS! Fishing boats capsized in storm off Mumbai coast. 18 fishermen missing. Coast guard deployed", "user_bio": "Indian Coast Guard", "user_location": "Mumbai, India", "query_used": "rescue needed"},
        {"content": "I'm stuck on roof of my house in Baton Rouge. Flood water rising. Please send help #floodhelp", "user_bio": "Father of 3 | Baton Rouge Louisiana", "user_location": "Baton Rouge, LA", "query_used": "flood help"},
        {"content": "Aftershock 5.4 hit same area as yesterday earthquake. Survivors still trapped in Kahramanmaras", "user_bio": "Search and Rescue coordinator", "user_location": "Kahramanmaras, Turkey", "query_used": "earthquake urgent"},
        {"content": "Volunteers needed for flood relief in Kerala. Boat operations, medical aid required. DM to sign up", "user_bio": "Kerala Disaster Relief NGO", "user_location": "Kerala, India", "query_used": "disaster relief needed"},
        {"content": "UPDATE: Rescue teams reached isolated village in Uttarakhand after landslide. 200 people evacuated safely", "user_bio": "NDRF India", "user_location": "Uttarakhand, India", "query_used": "rescue needed"},
    ]
    base_date = datetime.utcnow()
    tweets = []
    indices = list(range(len(samples)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        s = samples[idx]
        tweets.append({
            "date": (base_date - timedelta(minutes=i * random.randint(3, 15))).isoformat(),
            "tweet_id": f"synth_{int(time.time())}_{i}",
            "content": s["content"],
            "username": f"user_{random.randint(100, 999)}",
            "user_bio": s["user_bio"],
            "user_location": s["user_location"],
            "like_count": random.randint(5, 5000),
            "retweet_count": random.randint(2, 2000),
            "query_used": s["query_used"],
        })
    return tweets


if __name__ == "__main__":
    tweets = scrape_tweets(force_refresh=True)
    print(f"Total: {len(tweets)}")
    for t in tweets[:3]:
        print(f"  [{t['query_used']}] {t['content'][:80]}...")
