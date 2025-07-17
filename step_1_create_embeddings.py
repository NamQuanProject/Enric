import argparse
import os
from tqdm import tqdm
import torch
from internvl import CustonInternVLRetrievalModel

def main():
    parser = argparse.ArgumentParser(description="Generate image embeddings using CustonInternVLRetrievalModel.")
    parser.add_argument("--part", type=int, default=None, help="Part number to process (1-based index).")
    parser.add_argument("--total_parts", type=int, default=4, help="Total number of parts to split workload.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (e.g., 'cuda:0', 'cpu').")
    parser.add_argument("--input_folder", type=str, default="./data/database/database_origin/database_img/", help="Folder containing input images.")
    parser.add_argument("--output_folder", type=str, default="./embeddings/database/", help="Folder to save output embeddings.")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with torch.no_grad():
        embedding_model = CustonInternVLRetrievalModel(device=device)

        image_paths = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.sort()

        if args.part is not None:
            assert 1 <= args.part <= args.total_parts, "part must be between 1 and total_parts"
            total = len(image_paths)
            split_size = total // args.total_parts
            start = (args.part - 1) * split_size
            end = total if args.part == args.total_parts else start + split_size
            image_subset = image_paths[start:end]
            print(f"🔹 Running Part {args.part}/{args.total_parts}: Processing {len(image_subset)} images")
        else:
            image_subset = image_paths
            print(f"🔹 Running Full Mode: Processing all {len(image_subset)} images")

        for image in tqdm(image_subset, desc="Generating Embeddings"):
            name = os.path.splitext(image)[0]
            image_path = os.path.join(args.input_folder, image)
            if not os.path.isfile(image_path):
                continue
            embedding = embedding_model.encode_image([image_path], is_path=True).to(device)
            embedding = embedding.squeeze(0)
            output_path = os.path.join(args.output_folder, f"{name}.pt")
            torch.save(embedding.cpu(), output_path)
        
if __name__ == "__main__":
    main()
