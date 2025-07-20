import re
import os
import json
from fuzzysearch import find_near_matches
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


# Configuration
ORIGIN_IMG_FOLDER = "./data/database/database_origin/database_img"
NEW_IMG_FOLDER = "imgs"
ORIGIN_DATABASE_JSON = "./data/database/database.json"
CRAWLED_FOLDER = Path("crawled")

MATCHING_FOLDER = Path("matching-01-no-threshold")
assert MATCHING_FOLDER.exists(), f"Directory {MATCHING_FOLDER} does not exist."
WINDOW_LENGTH = 6400
NUM_WORKERS = 64
OUTPUT_PATH = "database_new.json"

##################################################

origin_img = Path(ORIGIN_IMG_FOLDER)
my_img = Path(NEW_IMG_FOLDER)

with open(ORIGIN_DATABASE_JSON, "r", encoding="utf-8") as f:
    origin_db = json.load(f)
my_db = dict()

filenames = list(os.listdir(CRAWLED_FOLDER))
for idx, filename in enumerate(filenames):
    key = filename.split(".")[0]
    try:
        with open(CRAWLED_FOLDER / filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            my_db[key] = data
    except Exception as e:
        print(f"Error reading {filename} at index {idx}: {e}")

##################################################

def is_marker(s):
    return bool(re.fullmatch(r"<[^>]+>", s.strip()))


def fast_approx_match(substring, text, max_err_ratio=0.1):
    max_errors = int(len(substring) * max_err_ratio)
    matches = find_near_matches(substring, text, max_l_dist=max_errors)
    return matches[0].start if matches else -1


def process_key(key):
    """return (key, new_object)"""
    origin_object = origin_db[key]
    if key in my_db:
        my_object = my_db[key]
    else:
        my_object = {
            "category": "",
            "author": "",
            "meta_description": "",
            "keywords": [],
            "content": "",
            "images": [],
            "word_count": 0,
            "reading_time_minutes": 0,
        }
    matching_file = MATCHING_FOLDER / f"{key}.json"
    if matching_file.exists():
        with open(matching_file, "r", encoding="utf-8") as f:
            matching_data = json.load(f)
    else:
        matching_data = {}

    new_object = {
        "url": origin_object["url"],
        "date": origin_object["date"],
        "title": origin_object["title"],
        "category": my_object["category"],
        "author": my_object["author"],
        "meta_description": my_object["meta_description"],
        "keywords": my_object["keywords"],
        "word_count": my_object["word_count"],
        "reading_time_minutes": my_object["reading_time_minutes"],
    }

    my_content_parts = my_object["content"].split("\n")
    my_images = {img["id"]: img for img in my_object["images"]}
    origin_text = origin_object["content"]

    origin_mapping_my = []
    for origin_id in origin_object["images"]:
        if matching_data.get(origin_id) is None:
            origin_mapping_my.append({
                "id": origin_id,
                "url": "",
                "position": None,  # Can not find the corresponding image
                "score": 0.0,
            })
            continue

        my_image_id = matching_data[origin_id]["filename"]
        my_image_score = matching_data[origin_id]["score"]
        # get id in my_db
        # get corresponding object in my_db
        my_image_info = my_images[my_image_id]
        # my_image_info["position"] is the #part in my_content_parts
        pos = my_image_info["position"]
        # Find prev_idx: the index of the text is in front of the marker
        prev_idx = pos - 1
        while prev_idx >= 0 and is_marker(my_content_parts[prev_idx]): prev_idx -= 1
        # Find next_idx: the index of the text behind the marker
        next_idx = pos + 1
        while next_idx < len(my_content_parts) and is_marker(my_content_parts[next_idx]): next_idx += 1

        prev_text = my_content_parts[prev_idx] if prev_idx >= 0 else ""
        next_text = (
            my_content_parts[next_idx]
            if next_idx < len(my_content_parts)
            else ""
        )

        prev_norm = prev_text[-WINDOW_LENGTH:]
        next_norm = next_text[:WINDOW_LENGTH]
        # Find the position in origin_text
        origin_pos = None

        if prev_norm:
            idx = origin_text.find(prev_norm)
            if idx != -1: origin_pos = idx + len(prev_norm)
            else:
                idx = fast_approx_match(prev_norm, origin_text)
                if idx != -1: origin_pos = idx + len(prev_norm)

        if origin_pos is None and next_norm:
            idx2 = origin_text.find(next_norm)
            if idx2 != -1: origin_pos = idx2
            else:
                idx2 = fast_approx_match(next_norm, origin_text)
                if idx2 != -1: origin_pos = idx2

        if origin_pos is None and prev_text and next_text:
            combo = prev_text + " " + next_text
            idx3 = origin_text.find(combo)
            if idx3 != -1: origin_pos = idx3 + len(prev_text) + 1
            else:
                idx3 = fast_approx_match(combo, origin_text)
                if idx3 != -1: origin_pos = idx3 + len(prev_text) + 1

        origin_mapping_my.append({
            "id": origin_id,
            "url": my_image_info["url"],
            "position": origin_pos,
            "score": my_image_score,
        })

    new_object["images"] = origin_mapping_my
    new_object["content"] = origin_text
    return key, new_object


if __name__ == "__main__":
    database_new = {}
    keys = list(origin_db.keys())

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_key, k): k for k in keys}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing keys"):
            key, new_obj = f.result()
            database_new[key] = new_obj

    output_file = Path(OUTPUT_PATH)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(database_new, f, ensure_ascii=False, indent=2)
    print(f"New database saved to {output_file}")
