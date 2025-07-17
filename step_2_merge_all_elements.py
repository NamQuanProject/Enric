import json
import csv


with open("private_test_final_elements_json/final_rerank_private_test_detail_top1_caption.json", 'r') as f:
    generated_caption = json.load(f)
with open("private_test_final_elements_json/reranking_query_first_article_fact_extract.json", 'r') as f:
    data_3 = json.load(f)
with open("private_test_final_elements_json/name_entity.json", 'r') as f:
    entity_name_data = json.load(f)
with open("private_test_final_elements_json/question_answer.json", 'r') as f:
    question_answer_data = json.load(f)


with open("private_test_final_elements_json/result-hoang.json", 'r') as f:
    new_database = json.load(f)

    # Extract query_id and crawl_caption into a new dictionary
    newdatabase_data = {}
    for value in new_database:
        temp = {
            'position': value['article_position'],
            'content' : value['article'],
            'crawl_caption': value['crawl_alt']
        }
        newdatabase_data[value['query_id']] = temp

final_merge_result = []
for query_id, value in generated_caption.items():
    generated_caption = value
    
    temp_summary = {
        'fact_summary':data_3[query_id]['summary']
    }
    batch = {
        'query_id' : query_id,
        'question_answer' : question_answer_data[query_id],
        'name_entity_keyword': entity_name_data[query_id],
        'generated_caption' : generated_caption,
        'article_summary' : temp_summary,
        'crawl_caption' :  newdatabase_data[query_id]["crawl_caption"],
        'article': newdatabase_data[query_id]["content"],
    }
    final_merge_result.append(batch)
    # save the final_merge_result
output_path = "./private_test_final_elements_json/final_merge_result.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_merge_result, f, ensure_ascii=False, indent=2)
print(f"Final merged result saved to: {output_path}")