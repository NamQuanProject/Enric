import re
from bs4 import BeautifulSoup
import os
import json
from pathlib import Path


# Configuration
OUTPUT_PATH = "result-hoang.json"
DATABASE_NEW_JSON = "./database_new.json"
CRAWLED_FOLDER = Path("crawled")
CONTEXT_EXTRACTION_IMAGE_ARTICLE = "./final_json_result/context_extraction_image_article.json"

##################################################

with open(DATABASE_NEW_JSON, 'r', encoding="utf-8") as f:
    database_new = json.load(f)
images_info = dict()
for _, v in database_new.items():
    for data in v["images"]:
        images_info[data["id"]] = data

captions = {} # {url: alt}
filenames = os.listdir(CRAWLED_FOLDER)
for filename in filenames:
    with open(CRAWLED_FOLDER / filename, 'r', encoding="utf-8") as f:
        data = json.load(f)
        for obj in data["images"]:
            captions[obj["url"]] = obj
with open(CONTEXT_EXTRACTION_IMAGE_ARTICLE, 'r', encoding="utf-8") as f:
    queries = json.load(f)

##################################################

def normalize_text(html_str):
    """
    - Remove the entire HTML card, retain the text.
    - Delete all http(s)://...
    """
    # 1) strip tags
    soup = BeautifulSoup(html_str or "", "lxml")
    text = soup.get_text(separator=" ", strip=True)
    # 2) remove URLs
    text = re.sub(r"http[s]?://\S+", "", text)
    return text.strip()


result = []
for query in queries.values():
    article_id = query["article_id"]
    image_id = query["image_id"]
    query_id = query["query_id"]

    # 'crawl_caption': str,
    # image_id -> url -> crawl_caption
    image_info = images_info[image_id]
    url = image_info["url"]
    crawl_alt =     normalize_text(captions[url]["alt"]     if captions.get(url) is not None else "")
    crawl_caption = normalize_text(captions[url]["caption"] if captions.get(url) is not None else "")

    # 'article': str,
    # 'article_position': int,
    article = database_new[article_id]["content"]
    article_position = image_info["position"] if image_info["position"] is not None else 0

    res = dict()
    res["query_id"] = query_id
    res["crawl_alt"] = crawl_alt
    res["crawl_caption"] = crawl_caption
    res["article"] = article
    res["article_position"] = article_position
    res["image_url"] = url
    res["article_url"] = database_new[article_id]["url"]
    res["image_id"] = image_id
    res["article_id"] = article_id
    result.append(res)

with open(OUTPUT_PATH, 'w', encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
