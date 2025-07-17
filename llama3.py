from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os 

load_dotenv()
HUGGINGFACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
class Llama:
    def __init__(self, model_id = "meta-llama/Meta-Llama-3-8B-Instruct" , hf_token = HUGGINGFACE_TOKEN, device = 'cuda:5'):
        self.model_id = model_id
        self.hf_token = hf_token

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            token=hf_token
        )

        # Set up generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
    def summarize_news(self, article_content: str, max_new_tokens: int = 1024) -> str:
        user_message = f"""
        You are a specialist in restructuring news articles with accuracy, clarity, and context awareness.

        Please read the following news article and reformulate it into a **well-structured version** that:
        - **Preserves all original content and key facts** without omission
        - **Organizes the article into clear, logical paragraphs** for better readability
        - Ensures each paragraph conveys a distinct idea or part of the story (e.g., background, main event, reactions)
        - Maintains the **original tone, journalistic integrity, and neutrality**
        - Uses **professional, objective language**
        - Avoids altering the facts, adding opinions, or introducing emotional language
        - Makes the overall flow more coherent while keeping the article's meaning intact

        Here is the news article:
        \"\"\"
        {article_content}
        \"\"\"

        Provide the **restructured version of the article** with clear, complete paragraphs below:
        """
    
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional assistant trained to improve the structure and readability of news articles. "
                    "Your goal is to reformat the content into well-organized paragraphs while preserving all the information, "
                    "facts, and tone of the original piece. Ensure the result is clear, coherent, and suitable for publication."
                )
            },
            {"role": "user", "content": user_message},
        ]
        # Convert messages to a prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Define termination tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Run the generation
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        full_output = outputs[0]["generated_text"]
        summary = full_output[len(prompt):].strip()

        return summary

    

    
    def build_enriched_caption_messages(self, image_description: str, news_summary: str):
        user_message_content = f"""
        You are given:

        1. **News Summary**:
        \"\"\"{news_summary.strip()}\"\"\"

        2. **Image Description**:
        \"\"\"{image_description.strip()}\"\"\"

        Think step by step to write a detailed caption. Start by reasoning through:

        1. What happened in the news event? Who is involved? When and where did it take place?
        2. Who or what is shown in the image?
        3. What are the subjects doing?
        4. What is the setting or background of the image?
        5. What emotions or symbolic elements are present?
        6. How does the image visually connect to the news story?

        After this reasoning, write a **clear, descriptive caption in one paragraph** that links the image to the event. Use simple but precise language and include concrete details. The caption must stand alone and help readers understand the context and significance of the image.
        Only output the final caption.
        """
        return [
            {
                "role": "system",
                "content": (
                    "You are a professional journalist writing extended captions for major news outlets. "
                    "Each caption should not only describe what is visible in the photo but clearly link it to the related news story. "
                    "Your captions must answer: What does the image show? What is happening? Why does it matter? "
                    "Use a journalistic tone, follow AP style, and ensure that the caption can stand on its own without needing to read the article."
                )
            },
            {
                "role": "user",
                "content": user_message_content
            }
        ]

    def build_enriched_caption_messages_cider_boost_1(self, image_description: str, news_summary: str):
        user_message_content = f"""
        You are given the following:

        1. **News Summary**:
        \"\"\"{news_summary.strip()}\"\"\"

        2. **Image Description**:
        \"\"\"{image_description.strip()}\"\"\"

        Your task is to write a single, clear, and highly descriptive caption that could be used by multiple people to describe this image in the context of the news. Think carefully and step by step:

        1. What happened in the news event? Who is involved? Where and when did it take place?
        2. What exactly is shown in the image — people, objects, setting?
        3. What are the people or subjects doing?
        4. What is in the background? What visual context is provided?
        5. What emotions, expressions, or symbolism are visible?
        6. How does the visual content connect to the key facts or emotions of the news story?

        Then, write a **single-paragraph caption** that:
        - Describes what is clearly visible in the image with high specificity.
        - Uses natural, human-like phrasing that could match how multiple people would describe the image.
        - Avoids hallucination — do not mention anything not clearly described in the image or article.
        - Uses simple but precise vocabulary — prioritize concrete nouns, clear verbs, and useful adjectives.

        Only output the final caption.
        """

        return [
            {
                "role": "system",
                "content": (
                    "You are an expert image captioning assistant trained to write high-quality captions that reflect human consensus. "
                    "Your captions must closely describe what is visible in the image and how it connects to the related news event, using natural phrasing. "
                    "Focus on high-overlap, common-sense word choices that would be shared across multiple human captions. "
                    "Write in a clear, concise journalistic tone that maximizes semantic overlap with reference captions. "
                    "Avoid generic phrases, filler words, or speculative statements."
                )
            },
            {
                "role": "user",
                "content": user_message_content
            }
        ]
    
    def enrich_caption(self, image_description, news_summary, max_new_tokens: int = 1024):
        messages = self.build_enriched_caption_messages_cider_boost_1(image_description, news_summary) 
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Define termination tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Run the generation
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

        full_output = outputs[0]["generated_text"]
        summary = full_output[len(prompt):].strip()

        return summary
    
    def build_enrich_caption_message_2(self , image_description, news_structure_again, keywords):
        user_message_content = f"""
        **INPUTS:**

        **News Summary:**
        {news_structure_again.strip()}

        **Image Description:**
        {image_description.strip()}

        **CAPTION REQUIREMENTS:**
        - Length: 50-100 words exactly
        - Style: AP Style, active voice, comprehensive detail
        - Structure: Lead with WHO did WHAT, then WHERE/WHEN, then detailed visual description, then relevant context/impact
        - Include: Full names, titles, locations, dates, background context, significance of the event
        - Multi-layered: News event + visual details + broader context + human impact when relevant
        - Avoid: Redundant phrases ("is seen," "appears to be"), speculation, overly emotional language

        **CIDER SCORE OPTIMIZATION:**
        - Use specific object names + descriptive adjectives (e.g., "red brick building," "crowded street")
        - Include spatial relationships ("in front of," "behind," "to the left of")
        - Mention colors, materials, clothing, and physical characteristics when relevant
        - Use action verbs that create clear mental images ("gesturing," "pointing," "holding")
        - Include environmental details that human annotators typically notice (weather, lighting, crowd size)
        - Balance specificity with consensus - use terms most people would agree on

        **REASONING STEPS:**
        1. **Primary News Elements**: Who are the main subjects with full titles? What specific action occurred? When and where precisely?
        2. **Visual Scene Analysis**: What is happening in the foreground? What's visible in the background? What emotions or reactions are captured?
        3. **Contextual Significance**: Why is this moment newsworthy? What's the broader impact or stakes involved?
        4. **Human Interest**: What human elements make this story compelling? Who is affected?
        5. **Verification & Completeness**: Are all details factually supported? Does the caption stand alone as a complete news summary?

        **EXAMPLES OF STRONG COMPREHENSIVE CAPTIONS:**

        *For protest image (78 words):*
        "Climate activists from Extinction Rebellion block traffic during rush hour on Highway 101 in downtown San Francisco on Tuesday morning, holding banners demanding immediate federal action on fossil fuel emissions. The coordinated demonstration, part of a week-long series of civil disobedience actions across 12 major cities, resulted in 47 arrests as police moved to clear the roadway. The protests coincide with congressional hearings on new environmental legislation that could reshape America's energy policy for the next decade."

        *For political event (84 words):*
        "President Biden signs the bipartisan Infrastructure Investment and Jobs Act in the East Room of the White House on Monday afternoon, surrounded by Democratic and Republican congressional leaders who spent months negotiating the historic $1.2 trillion package. The legislation promises to rebuild America's crumbling roads, bridges, and broadband networks while creating an estimated 2 million jobs over the next decade. Standing behind Biden are House Speaker Nancy Pelosi, Senate Majority Leader Chuck Schumer, and key Republican negotiators who crossed party lines to support the measure."

        *For breaking news (91 words):*
        "Firefighters from three departments battle a massive blaze that engulfed two apartment buildings in Portland's Pearl District early Wednesday morning, sending thick black smoke across the downtown skyline. The three-alarm fire forced emergency evacuation of nearly 200 residents, with at least 15 families now homeless after losing everything in the flames that authorities suspect started in a ground-floor restaurant. Emergency responders rescued four people from upper floors using ladder trucks, while Red Cross volunteers set up temporary shelter at a nearby community center to assist displaced families through the night."

        **YOUR TASK:**
        Write ONE comprehensive caption following this structure:
        [WHO with titles] [ACTION VERB] [WHAT/WHERE] [WHEN] [DETAILED VISUAL DESCRIPTION] [BROADER CONTEXT/SIGNIFICANCE] [HUMAN IMPACT when relevant]

        Target 50-100 words. Make it a complete news story that stands alone.

        Output only the final caption - no reasoning, no extra text.
        """


        # return [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You are a professional journalist writing detailed photo captions for major news outlets. "
        #             "Your job is to describe what the image shows, explain the event it relates to, and why it matters. "
        #             "Use journalistic tone and AP style."
        #         )
        #     },
        #     {
        #         "role": "user",
        #         "content": user_message_content
        #     }
        # ]
        pass
    
    def provided_context(self, article_content: str, max_new_tokens: int = 1024):
        # messages = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You are a helpful assistant that contextualizes images within articles. "
        #             "Your job is to read the entire article and generate a detailed, well-written paragraph that explains the image caption in depth. "
        #             "The caption is marked inside angle brackets (<<...>>). Use the surrounding text — before and after the caption — to infer what the image is likely showing, "
        #             "why it matters, what events or people are related to it, and how it fits into the overall meaning of the article. "
        #             "Your response should be rich, informative, and grounded entirely in the article context."
        #         )
        #     },
        #     {
        #         "role": "user",
        #         "content": (
        #             "Here is the article with the image caption marked in angle brackets. "
        #             "Write a detailed paragraph that describes the image and its full meaning within the article context.\n\n"
        #             f"{article_content.strip()}"
        #         )
        #     }
        # ]
        messages = [
        {
            "role": "system",
            "content": (
                "You are a professional news captioning assistant trained to write detailed, human-like photo captions that reflect how people describe images in context. "
                "Your goal is to generate a clear, specific, and contextually grounded caption that connects the image to the article in a meaningful way. "
                "To do this, you must carefully read the full article and extract all relevant context — people, places, actions, events, or causes — that relate to the image. "
                "Focus especially on the sentences before and after the caption marker (<<...>>), as these are likely to explain what the image shows and why it matters. "
                "You should use natural, common phrasing (as many people would say it) and avoid speculation or hallucination."
            )
        },
        {
            "role": "user",
            "content": (
                "Below is the full article with the image caption marked using angle brackets <<...>>. "
                "Your task is to identify all the relevant context around the caption and write a final, high-quality caption as a single paragraph. "
                "Start by reasoning step by step through:\n"
                "1. Who or what is shown in the image?\n"
                "2. What is happening in the image?\n"
                "3. What event or issue does this image relate to in the article?\n"
                "4. Where and when did it take place?\n"
                "5. Why is this image important to the article?\n"
                "6. What visual elements or symbolism are significant?\n\n"
                "After this reasoning, write one paragraph caption that:\n"
                "- Clearly describes the image content and its context.\n"
                "- Uses phrasing that matches what several humans might naturally say.\n"
                "- Avoids vague or generic wording.\n"
                "- Connects directly to the article without adding new information.\n\n"
                "Only output the final caption paragraph.\n\n"
                f"{article_content.strip()}"
            )
        }
        ]


        
        # Convert messages to a prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Define termination tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Run the generation
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        full_output = outputs[0]["generated_text"]
        # 'prompt' variable contains the input you gave, so remove that part
        summary = full_output[len(prompt):].strip()

        return summary
        
    def enrich_caption_with_keywords_and_facts(self, image_description, keywords, news_structure_again, max_new_tokens = 1024):
        messages = self.build_enrich_caption_message_2(image_description, news_structure_again, keywords)
        
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Define termination tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

        full_output = outputs[0]["generated_text"]
        summary = full_output[len(prompt):].strip()
        pass
    
    def generate_context_optimized_caption(self, news_summary, image_description):
        system_prompt = """
        You are a professional visual news captioner. Your task is to write a highly specific, well-structured image caption that connects the **visual content** of the image with the **event described in the news summary**.

        You must first reason step by step about what is happening, then generate a single fluent paragraph caption using the exact caption framework below.

        ### STEP-BY-STEP CHAIN OF THOUGHT:

        1. **Event Analysis**: What exactly happened? Who is involved (people, organizations, locations)? When and where did it happen?
        2. **Visual Identification**: Who or what is in the image? What actions are taking place? What objects or physical elements are present?
        3. **Spatial Layout**: Describe what is in the foreground, midground, and background (or left/right/center).
        4. **Event Connection**: How do the visual elements directly reflect the news event? Which items clearly show damage, participation, or consequences?
        5. **Naming Entities**: List at least three specific named entities (e.g., people, cities, governments, buildings) to be used in full.

        Then, using this information, write a caption that follows the structure below — but present it as a **natural, continuous paragraph**, not numbered points.

        ### REQUIRED CAPTION FRAMEWORK:

        1. **Primary Action**: [Entity1] [action] [object] at [position]  
        2. **Secondary Interaction**: [Entity2] [action] with [object] [position]  
        3. **Event Evidence**: "[Visual element1] from [Event]" + "[Visual element2] showing [Event] impact"  
        4. **Environmental Proof**: "[Detail] and [detail] throughout scene confirming [Event consequence]"

        ### STRICT RULES:
        - Use **named entities repeatedly** (no pronouns like "he", "she", "they", "it").
        - Describe all **people, places, and objects** in detail and connect them to the event.
        - Include at least **3 spatial references**.
        - Connect **every visual detail** to the **news event**.
        - End with environmental evidence proving the consequence of the event.
        - Only return the **final caption paragraph**. Do not return the reasoning steps themselves.
        """

        user_prompt = f"""
        You are given the following:

        1. **News Event Summary**:
        \"\"\"{news_summary.strip()}\"\"\"

        2. **Image Description**:
        \"\"\"{image_description.strip()}\"\"\"

        Now apply the chain-of-thought reasoning internally and then output a **single rich caption paragraph** using the structured caption framework. Follow all constraints carefully. Only output the final caption paragraph.
        """


        return [system_prompt, user_prompt]
    
    def enrich_caption_2(self, image_description, news_summary, max_new_tokens: int = 2048):
        system_prompt, user_prompt = self.generate_context_optimized_caption(news_summary, image_description)
    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Define termination tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Run the generation
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

        full_output = outputs[0]["generated_text"]
        summary = full_output[len(prompt):].strip()

        return summary
    
    def assemble(self, prompt, max_new_tokens: int = 1500):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional journalist writing extended captions for major news outlets. "
                    "Each caption should not only describe what is visible in the photo but clearly link it to the related news story. "
                    "Your captions must answer: What does the image show? What is happening? Why does it matter? "
                    "Use a journalistic tone, follow AP style, and ensure that the caption can stand on its own without needing to read the article."
                )
            },
            {
                "role": "user",
                "content": prompt  
            }
        ]


        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Define termination tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Run the generation
        outputs = self.generator(
            prompt,
            min_new_tokens = 100,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

        full_output = outputs[0]["generated_text"]
        text = full_output[len(prompt):].strip()

        return text
    
    
    def question_answer(self, prompt, max_new_tokens: int = 2048):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an editorial assistant for a major news organization. "
                    "Your task is to generate clear, factual, and concise question-and-answer pairs that accompany images in news articles. "
                    "Each Q&A should help readers understand what the image shows, who is involved, what is happening, and why it matters. "
                    "Focus on contextual clarity by connecting visual elements to key facts from the news story. "
                    "Use a neutral, journalistic tone, and limit each answer to no more than 2–3 sentences."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Define termination tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Run the generation
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

        full_output = outputs[0]["generated_text"]
        summary = full_output[len(prompt):].strip()
        return summary
    
    def name_entity_extraction(self, prompt, max_new_tokens: int = 1024):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a named entity recognition assistant working for a major news organization. "
                    "Your job is to extract all relevant named entities from a news article, image description, and caption. "
                    "Focus on identifying and listing people, organizations, locations, dates, events, and other significant entities that are either mentioned in the article "
                    "or visually connected to the image. "
                    "Group entities by category, and ensure that the list is clean, factual, and concise. "
                    "Do not generate any additional commentary—only output the grouped list of named entities."
                )

            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Define termination tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Run the generation
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

        full_output = outputs[0]["generated_text"]
        summary = full_output[len(prompt):].strip()

        return summary