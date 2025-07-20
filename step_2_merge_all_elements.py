import json
import csv


def merge_function(generative_caption_path, query_first_article_path, entity_name_path, question_answer_path, new_database_path, output_path):
    with open(generative_caption_path, 'r') as f:
        generated_caption = json.load(f)
    with open(query_first_article_path, 'r') as f:
        data_3 = json.load(f)
    with open(entity_name_path, 'r') as f:
        entity_name_data = json.load(f)
    with open(question_answer_path, 'r') as f:
        question_answer_data = json.load(f)

    with open(new_database_path, 'r') as f:
        new_database = json.load(f)

    # Extract query_id and crawl_caption into a new dictionary
    newdatabase_data = {}
    for value in new_database:
        temp = {
            'position': value['article_position'],
            'content': value['article'],
            'crawl_caption': value['crawl_alt']
        }
        newdatabase_data[value['query_id']] = temp

    final_merge_result = []
    for query_id, value in generated_caption.items():
        generated_caption = value
        
        temp_summary = {
            'fact_summary': data_3[query_id]['summary']
        }
        batch = {
            'query_id': query_id,
            'question_answer': question_answer_data[query_id],
            'name_entity_keyword': entity_name_data[query_id],
            'generated_caption': generated_caption,
            'article_summary': temp_summary,
            'crawl_caption': newdatabase_data[query_id]["crawl_caption"],
            'article': newdatabase_data[query_id]["content"],
        }
        final_merge_result.append(batch)    

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_merge_result, f, ensure_ascii=False, indent=2)
        print(f"Final merged result saved to: {output_path}")


