import json
import os
from internvl import CustonInternVLCaptionModel
from tqdm import tqdm
import csv

def preprocess_caption_query(args):
    model = CustonInternVLCaptionModel(model_name=args.model, device="cuda:5")
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            caption_json = json.load(f)
    else:
        caption_json = {}

    db_base_path = 'data/database/database_origin/database_img'
    csv_file = 'final_csv_result/temp_final_rerank.csv'
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = list(csv.DictReader(file))
        with tqdm(total=len(csv_reader), desc="📝 Generating Captions", ncols=100) as pbar:
            for row in csv_reader:
                query_id = row['query_id']
                image_id = row.get('image_id_1')

                if query_id in caption_json and len(caption_json[query_id]) == 5:
                    pbar.set_postfix_str(f"✔ Skipped {query_id}")
                    pbar.update(1)
                    continue
                
                image_path = os.path.join(db_base_path, image_id + '.jpg')
                caption = model.generate_caption(image_path)

                caption_json[query_id] = caption


                pbar.set_postfix_str(f"📌 {query_id}")
                pbar.update(1)

                with open(args.output_file, 'w') as f:
                    json.dump(caption_json, f, indent=4)
        print(f"✅ Captions saved to {args.output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess captions for InternVL model")
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL2_5-8B', help='Model to use for caption generation')
    parser.add_argument('--output_file', type=str, default='./private_test_final_elements_json/final_rerank_private_test_detail_top1_caption.json', help='Output file to save the captions')
    parser.add_argument('--start_index', type=int, default=0, help='Start index to resume processing from')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--batch', type=bool, default=False, help='Batch size for processing images')
    args = parser.parse_args()

    preprocess_caption_query(args)

if __name__ == "__main__":
    main()

