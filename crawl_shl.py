from playwright.sync_api import sync_playwright
import pandas as pd

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

assessments = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    print("Opening SHL catalog...")
    page.goto(CATALOG_URL, timeout=60000)
    page.wait_for_timeout(5000)  # wait for JS to load

    links = page.locator("a").all()

    for link in links:
        try:
            href = link.get_attribute("href")
            name = link.inner_text().strip()
        except:
            continue

        if not href or not name:
            continue

        if href.startswith("/solutions/products/") and "job" not in href.lower():
            assessments.append({
                "name": name,
                "url": BASE_URL + href
            })

    browser.close()

df = pd.DataFrame(assessments).drop_duplicates()
df.to_csv("data.csv", index=False)

print("Total assessments saved:", len(df))
