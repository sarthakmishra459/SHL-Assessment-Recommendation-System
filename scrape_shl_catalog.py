import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://www.shl.com"
CATALOG_URL = BASE_URL + "/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def parse_page(soup):
    assessments = []

    rows = soup.select("tr[data-entity-id]")
    for row in rows:
        name_tag = row.select_one("td.custom__table-heading__title a")
        name = name_tag.text.strip()
        relative_url = name_tag["href"]
        full_url = BASE_URL + relative_url

        remote_support = "Yes" if row.select("td:nth-of-type(2) .-yes") else "No"
        adaptive_support = "Yes" if row.select("td:nth-of-type(3) .-yes") else "No"

        test_type_tags = row.select("td:nth-of-type(4) .product-catalogue__key")
        test_types = [tag.text.strip() for tag in test_type_tags]

        assessments.append({
            "name": name,
            "url": full_url,
            "remote_support": remote_support,
            "adaptive_support": adaptive_support,
            "test_types": test_types
        })

    return assessments

def scrape_all_pages():
    all_assessments = []

    # Assuming max 144 items and 12 per page (can change if needed)
    for start in range(0, 384, 12):  
        url = f"{CATALOG_URL}?start={start}&type=1"
        print(f"🔍 Scraping: {url}")
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.content, "html.parser")
        page_data = parse_page(soup)
        if not page_data:
            print("⚠️ No data found on this page, stopping.")
            break
        all_assessments.extend(page_data)
        time.sleep(2)  # Polite delay

    # Save to JSON
    with open("shl_assessments_individual.json", "w", encoding="utf-8") as f:
        json.dump(all_assessments, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Scraped {len(all_assessments)} assessments across all pages.")

if __name__ == "__main__":
    scrape_all_pages()
