from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch


# Configuration
THRESHOLD_PIXEL_DIFF = 0.01  # Threshold for pixel difference in image comparison
OUTPUT_MATCHING_FOLDER = "matching-01-no-threshold"
ORIGIN_EMBEDDING = "../embeddings/database_image_internVL_g"
NEW_EMBEDDING = "../embeddings/maching_new_database_internvlg"
ORIGIN_IMG_FOLDER = "./data/database/database_origin/database_img"
NEW_IMG_FOLDER = "imgs"
ORIGIN_DATABASE_JSON = "./data/database/database.json"
CRAWLED_FOLDER = Path("crawled")

##################################################

matching_dir = Path(OUTPUT_MATCHING_FOLDER)
os.makedirs(matching_dir, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

origin_embedding = Path(ORIGIN_EMBEDDING)
my_embedding = Path(NEW_EMBEDDING)
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

# Get intersection of keys
origin_keys = set(origin_db.keys())
my_keys = set(my_db.keys())
key_intersection = origin_keys & my_keys
print(f"Number of key intersections: {len(key_intersection)}")

##################################################

def load_images_from_list(img_dir, img_id_list, ext=".jpg"):
    img_dict = {}
    for img_id in img_id_list:
        img_path = img_dir / (img_id + ext)
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            img_dict[img_id] = arr
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return img_dict


def compare_img_arrays(arr1, arr2, threshold=THRESHOLD_PIXEL_DIFF):
    if arr1.shape != arr2.shape:
        return False
    abs_diff = np.abs(arr1.astype(np.int32) - arr2.astype(np.int32))
    total_diff = np.sum(abs_diff)
    total_values = arr1.size * 255
    diff_ratio = total_diff / total_values
    return diff_ratio <= threshold


def load_embeddings_from_list(embedding_dir, id_list):
    embedding_dict = {}
    for id in id_list:
        embedding_path = embedding_dir / (id + ".pt")
        if not embedding_path.exists(): continue
        try:
            embedding = torch.load(embedding_path, map_location=device).unsqueeze(0)
            embedding_dict[id] = embedding
        except Exception as e:
            print(f"Error loading {embedding_path}: {e}")
    return embedding_dict


def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    return torch.nn.functional.cosine_similarity(embedding1, embedding2)


empty_tensor = torch.zeros(1, 768).to(device)
def process_key(key):
    origin_images = origin_db[key]["images"]  # List of origin image ids (strings)
    my_images = [img_obj["id"] for img_obj in my_db[key]["images"]]  # List of my image ids
    my_ids = [filename.split(".")[0] for filename in my_images]

    origin_arrs = load_images_from_list(origin_img, origin_images, ext=".jpg")
    my_arrs = load_images_from_list(my_img, my_images, ext="")
    origin_embeddings = load_embeddings_from_list(origin_embedding, origin_images)
    my_embeddings = load_embeddings_from_list(my_embedding, my_ids)

    json_matching_file = matching_dir / (key + ".json")
    assert not os.path.exists(json_matching_file), f"File {json_matching_file} already exists. Please remove it before running the script."
    mapping = {}  # origin_id: new_id

    for origin_id in origin_images:
        if origin_id not in mapping:
            mapping[origin_id] = None
        if mapping[origin_id] is not None: continue
        arr1 = origin_arrs.get(origin_id)
        if arr1 is None: continue

        found_match = False
        for my_filename, arr2 in my_arrs.items():
            my_id = my_filename.split(".")[0]
            if compare_img_arrays(arr1, arr2):
                mapping[origin_id] = {
                    "filename": my_filename,
                    "score": 1.0,
                }
                found_match = True
                break
        if found_match: continue

        mx_cosine_similarity = 0
        for my_filename in my_arrs.keys():
            my_id = my_filename.split(".")[0]
            cur_cosine_similarity = cosine_similarity(
                origin_embeddings.get(origin_id, empty_tensor),
                my_embeddings.get(my_id, empty_tensor),
            ).item()
            if mx_cosine_similarity < cur_cosine_similarity:
                mx_cosine_similarity = cur_cosine_similarity
                mapping[origin_id] = {
                    "filename": my_filename,
                    "score": mx_cosine_similarity,
                }

    with open(json_matching_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


with ThreadPoolExecutor(max_workers=64) as executor:
    results = list(tqdm(executor.map(process_key, key_intersection), total=len(key_intersection), desc="Processing"))
