import asyncio
import aiohttp
import async_timeout
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlsplit, urlunsplit
from bs4 import BeautifulSoup
from tqdm import tqdm


# Configuration
CONCURRENT_REQUESTS = 32
TIMEOUT = 60
ORIGIN_DATABASE_JSON = "./data/database/database.json"
CRAWLED_FOLDER = Path("crawled")
NEW_IMG_FOLDER = Path("imgs")
RETRIES = 8  # retry attempts
BACKOFF = [1, 3, 5, 7, 9, 11, 13, 15]  # seconds backoff for retries

##################################################

def prepare_dirs():
    CRAWLED_FOLDER.mkdir(exist_ok=True)
    NEW_IMG_FOLDER.mkdir(exist_ok=True)

def normalize_url(url):
    return url if not url.startswith("//") else "https:" + url

def get_original_image_url(url):
    url = normalize_url(url)
    if "cnn" not in url:
        return url
    parts = urlsplit(url)
    # remove all the following parameters
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


async def fetch(session, url, key):
    url = normalize_url(url)
    for attempt in range(RETRIES):
        try:
            async with async_timeout.timeout(TIMEOUT):
                async with session.get(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                ) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        return html
                    elif resp.status == 429:
                        wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
                        await asyncio.sleep(wait)
                        continue
                    else:
                        return None
        except Exception as e:
            await asyncio.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
    return None


async def fetch_binary(session, url, key):
    url = get_original_image_url(url)
    for attempt in range(RETRIES):
        try:
            async with async_timeout.timeout(TIMEOUT):
                async with session.get(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        return data
                    elif resp.status == 429:
                        wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
                        await asyncio.sleep(wait)
                        continue
                    else:
                        return None
        except Exception as e:
            await asyncio.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
    return None


# Parse and download, including inline placeholders
async def parse_and_download(html, key, url, session):
    try:
        soup = BeautifulSoup(html, "lxml")
        data = {
            "key": key,
            "url": url,
            "crawl_date": datetime.utcnow().isoformat() + "Z",
        }
        # title, publish_date, section, category, author, desc, keywords ...
        title = soup.find("title")
        data["title"] = title.get_text(strip=True) if title else None
        date = None
        for sel in [
            {"itemprop": "datePublished"},
            {"name": "pubdate"},
            {"property": "article:published_time"},
        ]:
            t = soup.find("meta", sel)
            if t and t.get("content"):
                date = t["content"]
                break
        data["publish_date"] = date
        # section
        ms = soup.find("meta", {"property": "article:published_time"})
        data["section"] = ms["content"] if ms else None
        # category
        crumbs = soup.select("ul.breadcrumbs li a")
        data["category"] = crumbs[-1].get_text(strip=True) if crumbs else None
        auth = soup.find("meta", {"name": "author"})
        data["author"] = auth["content"] if auth else None
        desc = soup.find("meta", {"name": "description"})
        data["meta_description"] = desc["content"] if desc else None
        kw = soup.find("meta", {"name": "keywords"})
        data["keywords"] = [k.strip() for k in kw["content"].split(",")] if kw else None
        # build content
        body = soup.find("article") or soup.find("div", {"class": "article-body"})
        parts = []
        images = []
        if body:
            # get text from <p>
            for el in body.find_all("p"):
                text = el.get_text(strip=True)
                if text:
                    parts.append(text)

            # get images in <div class="image__container">
            for img_div in body.find_all("div", class_="image__container"):
                img_tag = img_div.find("img")
                if not img_tag or not img_tag.get("src"):
                    continue
                img_url = get_original_image_url(img_tag["src"])

                ext = Path(urlparse(img_url).path).suffix or ".jpg"
                img_key = f"{key}_{len(images)}{ext}"
                img_bytes = await fetch_binary(session, img_url, key)
                if not img_bytes:
                    continue
                (NEW_IMG_FOLDER / img_key).write_bytes(img_bytes)
                
                # find caption in <div class="image__metadata">
                cap_div = img_div.find_next_sibling("div", class_="image__metadata")
                caption = cap_div.get_text(strip=True) if cap_div else None

                # insert placeholder and record image metadata
                pos = len(parts)
                parts.append(f"<{img_key}>")
                images.append({
                    "id": img_key,
                    "url": img_url,
                    "alt": img_tag.get("alt", ""),
                    "caption": caption,
                    "position": pos,
                })

        data["content"] = "\n".join(parts)
        data["images"] = images
        # stats
        words = data["content"].split()
        data["word_count"] = len(words)
        data["reading_time_minutes"] = round(len(words) / 200, 2)
        data["named_entities"] = []
        data["event_date"] = date
        data["event_location"] = None
        return data
    except Exception as e:
        return None


async def worker(session, queue, pbar):
    while True:
        key, url = await queue.get()
        out = CRAWLED_FOLDER / f"{key}.json"
        if out.exists():
            pbar.update(1)
            queue.task_done()
            continue
        html = await fetch(session, url, key)
        if html:
            data = await parse_and_download(html, key, url, session)
            if data:
                out.write_text(json.dumps(data, ensure_ascii=False))
        queue.task_done()
        pbar.update(1)


async def main():
    prepare_dirs()
    mapping = json.load(open(ORIGIN_DATABASE_JSON))
    total = len(mapping)
    q = asyncio.Queue()
    for k, e in mapping.items():
        q.put_nowait((k, e["url"]))
    async with aiohttp.ClientSession() as s:
        with tqdm(total=total, desc="Crawling") as p:
            tasks = [
                asyncio.create_task(worker(s, q, p)) for _ in range(CONCURRENT_REQUESTS)
            ]
            await q.join()
            [t.cancel() for t in tasks]


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    print(f"Done in {time.time()-start:.2f}s")
