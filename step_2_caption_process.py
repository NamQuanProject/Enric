import argparse
from pathlib import Path
from jinja2 import Template
import json
from tqdm import tqdm
from llmassemblers import LLMAssembler
import time

PROMPT_TEMPLATE_DIR = Path('./assemble_caption_prompt_template')
RESULT_DIR = Path('./assemble_result')
FEWSHOT_EXAMPLE_PATH = Path('./assemble_caption_prompt_template/test.json')
CAPTION_INPUT_PATH = Path('./private_test_final_elements_json/final_merge_result.json')
TEST_FEWSHOT_EXAMPLE_PATH = PROMPT_TEMPLATE_DIR / 'test.json'
TEST_CAPTION_INPUT_PATH = PROMPT_TEMPLATE_DIR / 'test.json'



assembler = LLMAssembler(device = 'cuda:5', model_type = 'llama3')
template_strats: dict[str, Template] = {}

RESULT_DIR.mkdir(exist_ok=True)

with open(CAPTION_INPUT_PATH, 'r') as f:
    inputs = json.load(f)

with open(FEWSHOT_EXAMPLE_PATH, 'r') as f:
    examples = json.load(f)

for path in PROMPT_TEMPLATE_DIR.rglob("*.j2"):
    with open(path, 'r', encoding='utf-8') as f:
        template_str = f.read()
        template = Template(template_str)
        template_strats[str(path.stem)] = template

def assemble(input: dict, strat: str) -> str:
    template = template_strats[strat]
    prompt = template.render(**input)
    enhanced_caption = assembler.assemble(prompt)
    return enhanced_caption
def model_qa(input: dict, strat: str) -> str:
    template = template_strats[strat]
    prompt = template.render(**input)
    enhanced_caption = assembler.question_answer(prompt)
    return enhanced_caption
def model_name_entity(input: dict, strat: str) -> str:
    template = template_strats[strat]
    prompt = template.render(**input)
    enhanced_caption = assembler.name_entity_extraction(prompt)
    return enhanced_caption

def compose_input(batch: dict, examples: list[dict]) -> dict:
    # TODO: Fill in the Nones
    
    example_contexts = [None] * len(examples)
    batch_context = None
    return {
        "article" : batch["article"],
        "article_position": batch["article_position"],
        "generated_caption": batch['generated_caption'],
        "crawl_caption": batch['crawl_caption'],
        "question_answer": batch['question_answer'],
        'name_entity_keyword':batch['name_entity_keyword'],
        'related_phrases': batch['related_phrases'],
        "context": batch_context,
        "summary": {
            "raw": batch["article_summary"]["raw_summary"],
            "restruct": batch["article_summary"]["restruct_summary"],
            "fact": batch["article_summary"]["fact_summary"],
        },
        'enhanced_captions_55': batch['enhanced_captions_55'],
        "examples": [{
            "generated_caption": example['generated_caption'],
            "crawl_caption": example['crawl_caption'],
            "context": example_context,
            "summary": example['article_summary'],
            "expected_output": example['expected_output'],
        } for example, example_context in zip(examples, example_contexts)],
    }

def main(args):
    CAPTION_INPUT_PATH = args.caption_input_path
    strat = args.strategy
    template_test = args.template_test
    qa = args.qa
    model_name = args.model_type
    name_entity = args.name_entity
    if template_test:
        with open(TEST_CAPTION_INPUT_PATH, 'r') as f:
            test_inputs = json.load(f)
        with open(TEST_FEWSHOT_EXAMPLE_PATH, 'r') as f:
            test_examples = json.load(f)
        template = template_strats[strat]
        prompt = template.render(**compose_input(test_inputs[0], test_examples))
        with open(PROMPT_TEMPLATE_DIR / f"{strat}.md", 'w') as f:
            f.write(prompt)
        return
    
    result = []
    
    for batch in tqdm(inputs, desc="Processing batches"):
        input = compose_input(batch, examples)
        if qa:
            enhanced_caption = model_qa(input, strat)
        elif name_entity:
            enhanced_caption = model_name_entity(input, strat)
        else:
            enhanced_caption = assemble(input, strat)
        result.append({
            'query_id': batch['query_id'],
            'enhanced_caption': enhanced_caption,
        })
        
        with open(RESULT_DIR / f"{strat}_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, choices=template_strats.keys(), required=True,
                        help="Reranking strategy to use")
    parser.add_argument('--template_test', action='store_true',
                        help="Test and print loaded template strategies")
    parser.add_argument('--qa', action='store_true',
                        help="Test and print loaded template strategies")
    parser.add_argument('--caption_input_path', type=Path, default=CAPTION_INPUT_PATH,
                        help="Path to the caption input JSON file")
    parser.add_argument('--name_entity', action='store_true',
                        help="Test and print loaded template strategies")
    parser.add_argument('--model_type',type=str, default='llama',
                        help="Test and print loaded template strategies")
    args = parser.parse_args()
    main(args)