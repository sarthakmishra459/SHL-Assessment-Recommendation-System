import json
import requests
from bs4 import BeautifulSoup
import time
import random

def extract_duration(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            print(f"Failed to fetch {url} with status {res.status_code}")
            return None

        soup = BeautifulSoup(res.text, "html.parser")
        all_rows = soup.find_all("div", class_="product-catalogue-training-calendar__row")

        for row in all_rows:
            h4 = row.find("h4")
            if h4 and "assessment length" in h4.text.strip().lower():
                # Try to find <p> tag with the duration
                for p in row.find_all("p"):
                    if "Approximate Completion Time" in p.get_text():
                        try:
                            return int("".join(filter(str.isdigit, p.get_text())))
                        except ValueError:
                            return None
        return None

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def enrich_duration_only(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        print(f"Scraping duration for: {entry['name']}")
        duration = extract_duration(entry["url"])
        if duration is not None:
            print(f"Found duration: {duration} minutes")
            entry["duration"] = duration
        time.sleep(random.uniform(1, 2))  # Polite delay

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated {file_path} with durations.")

# Enrich both JSONs with duration only
enrich_duration_only("shl_assessments_individual.json")
enrich_duration_only("shl_assessments_pre.json")
