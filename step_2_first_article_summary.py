import os
import torch
from tqdm import tqdm
import argparse
import json
import csv
from llama3 import Llama  # Make sure Llama class is correctly implemented

def main(args):
    retrieval_results_folder = args.result_folder
    database_file = args.database_file
    output_file = "./private_test_final_elements_json/reranking_query_first_article_question_answer.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    llms_bot = Llama(device= 'cuda:3')

    # Read CSV
    with open(retrieval_results_folder, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    # Load JSON database
    with open(database_file, 'r') as file:
        database_data = json.load(file)

    if os.path.exists(output_file):
        with open(output_file, 'r') as infile:
            result = json.load(infile)
    else:
        result = {}

    for row in tqdm(data):
        query = row["query_id"]
        first_article = row["article_id_1"]
        if query in result:
            continue  # Already processed
        if first_article not in database_data:
            print(f"Warning: Article {first_article} not found in database.")
            continue

        article_content = database_data[first_article]["content"]

        summarized_content = llms_bot.summarize_news(article_content)
        result[query] = {
            "article_id": first_article,
            "summary": summarized_content
        }
        
        with open(output_file, 'w') as outfile:
            json.dump(result, outfile, indent=2)

    print(f"Summarized results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', type=str, default="./final_csv_result/private_final_retrieval_merging_final_results.csv", help="CSV file with retrieval results.")
    parser.add_argument('--database_file', type=str, default="./data/database/database.json", help="JSON file containing article contents.")
    args = parser.parse_args()
    main(args)