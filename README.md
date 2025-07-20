# First Place in the EVENTA 2025 Track 1 (held at ACM MM 2025)

# ENRIC: EveNt-AwaRe Captioning with Image Retrieval via UnCertainty-Guided Re-ranking and Semantic Ensemble Reasoning

[[Challenge]](https://ltnghia.github.io/eventa/) [Paper]

Team: cerebro (Members: Nam-Quan Nguyen, Minh-Hoang Le, Vinh-Toan Vong)

## Final Leaderboard

| **Team**            | **Rank** | **Overall** | **AP**  | **R@1** | **R@10** | **CLIP** | **CIDEr** |
|---------------------|----------|-------------|---------|---------|----------|----------|-----------|
| **cerebro (Ours)**  | **1**    | **0.5501**  | **0.991** | **0.989** | **0.995** | 0.826    | **0.210** |
| SodaBread           | 2        | 0.5457      | 0.982   | 0.977   | 0.988    | **0.870** | 0.204     |
| Re: Zero Slavery    | 3        | 0.4515      | 0.955   | 0.945   | 0.973    | 0.732    | 0.156     |
| ITxTK9              | 4        | 0.4200      | 0.966   | 0.955   | 0.983    | 0.828    | 0.133     |
| noname\_            | 5        | 0.2824      | 0.708   | 0.663   | 0.801    | 0.783    | 0.081     |

# Descriptions

This repository contains the solution developed by the **cerebro** team from the University of Science - VNUHCM to address the challenge posed in EVENTA - Track 1: Event-Enriched Image Retrieval and Captioning.

|<img width="1656" height="788" alt="image" src="https://github.com/user-attachments/assets/76848373-cc65-4577-a05d-2b85530c652c" />|
|:--:|
|Overview of the Retrieval and Re-ranking Module|

|<img width="1684" height="676" alt="image" src="https://github.com/user-attachments/assets/5903cd1f-c33d-48f3-a22d-33cf3eb03984" />|
|:--:|
|Overview of the Captioning Module|

To reproduce our results, please follow the instructions provided below.

# Instructions
### 🔹 Phase 1: Retrieval & Reranking

1. **Create Embeddings**

```bash
python step_1_create_embeddings.py --input_folder data/database --output_folder embeddings/database
python step_1_create_embeddings.py --input_folder data/track1_private/query --output_folder embeddings/query
```

2. **Initial Retrieval**

```bash
python step_1_retrieval.py --database_folder embeddings/database --query_folder embeddings/query
```

3. **Reranking**

```bash
python step_1_reranking.py
```

---

### 🔹 Phase 2: Captioning & Semantic Reasoning

4. **Create the crawling database**

* Crawl articles and images from URLs in the origin database:

  ```bash
  python step_2_0_crawling.py
  ```

* Create Embeddings for crawled images:

  ```bash
  python step_1_create_embeddings.py --input_folder imgs --output_folder embeddings/maching_new_database_internvlg
  ```

* For each article, create a json file mapping the origin image and the crawled image

  ```bash
  python step_2_0_matching_image.py
  ```

* Create database_new.json

  ```bash
  python step_2_0_new_database.py
  ```

* Create output corresponding to final_json_result/context_extraction_image_article.json

  ```bash
  python step_2_0_create_result.py
  ```

5. **Generate Query Captions**

```bash
python step_2_create_caption_query.py
```

6. **Retrieve First Article Summary**

```bash
python step_2_first_article_summary.py
```

7. **Caption Enhancement via Strategies**

* Using Question Answering:

  ```bash
  python step_2_caption_process.py --qa --strategy qa
  ```

* Using Named Entity Extraction:

  ```bash
  python step_2_caption_process.py --name_entity --strategy name_entity
  ```

* Using Chain-of-Thought Reasoning:

  ```bash
  python step_2_caption_process.py --strategy cot_5_things_fact_more_event
  ```

---

## 🔧 In Refactor Progress...

This repo is actively being cleaned up for public release. Stay tuned for code modularization, configs, and pretrained checkpoints.

---

# Contact

- Nam-Quan Nguyen (nnquan23@apcs.fitus.edu.vn)
- Minh-Hoang Le (lmhoang22@apcs.fitus.edu.vn)
- Vinh-Toan Vong (vvtoan22@apcs.fitus.edu.vn)



