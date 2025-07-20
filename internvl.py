import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from conversation import get_conv_template
from typing import List, Optional, Tuple, Union
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
import torch.nn as nn


class CustonInternVLRetrievalModel():
    def __init__(self, model_name = "OpenGVLab/InternVL-14B-224px" , device='cuda:7'):
        
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).to(self.device).eval()
        


        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_name, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, add_eos_token=True, trust_remote_code=True)
        self.tokenizer.pad_token_id = 0  

    def encode_image(self, images, mode='InternVL-G', is_path = False):
        if is_path:
            images = [Image.open(path).convert('RGB') for path in images]

        pixel_values = self.image_processor(images=images, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
        embedding = self.model.encode_image(pixel_values, mode=mode)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding
    
    def encode_text(self, text):
        prefix = 'summarize:'
        text = prefix + text 
        input_ids = self.tokenizer([text], return_tensors='pt', max_length=80,
                      truncation=True, padding='max_length').input_ids.to(self.device)
        feature_text = self.model.encode_text(input_ids)
        feature_text = feature_text / feature_text.norm(dim=-1, keepdim=True)
        return feature_text

    def compute_image_text_probs(self, image, text, mode='InternVL-G', is_image_path = False, soft_max = True):
        with torch.no_grad():
            image = self.encode_image(image, mode=mode, is_path=is_image_path)
            text = self.encode_text(text)
            image = image / image.norm(dim=-1, keepdim=True)
            text = text / text.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            probs = logit_scale * image @ text.t()
            if soft_max:
                probs = probs.softmax(dim=-1)
            else:
                probs = probs
            return probs
    
   
    
    def compute_text_text_probs(self, text1, text2, soft_max = True):
        with torch.no_grad():
            text_feature_1 = self.encode_text(text1)
            text_feature_2 = self.encode_text(text2)
            text_feature_1 = text_feature_1 / text_feature_1.norm(dim=-1, keepdim=True)
            text_feature_2 = text_feature_2 / text_feature_2.norm(dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            probs = logit_scale * text_feature_1 @ text_feature_2.t()
            if soft_max:
                probs = probs.softmax(dim=-1)
            else:
                probs = probs
            return probs
        
    
    
    
    def crop_center(self, image, crop_width, crop_height):
        width, height = image.size
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        return image.crop((left, top, right, bottom))

    def generate_caption(self, image, is_path = False):
        with torch.no_grad():
            self.tokenizer.add_eos_token = False
            if is_path:
                image = Image.open(image).convert('RGB')
            
            
            pixel_values = self.image_processor(images=image, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
            tokenized = self.tokenizer("English caption:", return_tensors='pt')
            pred = self.model.generate(
                pixel_values=pixel_values,
                input_ids= tokenized.input_ids.to(self.device),
                attention_mask= tokenized.attention_mask.to(self.device),
                num_beams=5,
                max_new_tokens=32  # required
            )
            caption = self.tokenizer.decode(pred[0].cpu(), skip_special_tokens=True).strip()    
            return caption
        



class CustonInternVLCaptionModel():
    def __init__(self, model_name = 'OpenGVLab/InternVL2_5-8B' , device='cuda:0'):
        self.device = torch.device(device)
        
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn =True,
            trust_remote_code=True).eval().to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

        self.text_projection = nn.Parameter(torch.empty(4096, 1024)).to(device)  # frozen

    def build_transform(self, input_size, aug=False):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD

        if aug:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])
        return transform


        
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_image(self, image_file, input_size=448, max_num=12, aug=False):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size, aug=aug)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def pure_text_generation(self, question):
        with torch.no_grad():
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            response, history = self.model.chat(self.tokenizer, None, question, generation_config, history=None, return_history=True)
            return response
        
    
    def generate_caption(self, image_path):
        with torch.no_grad():
            image = self.load_image(image_path, max_num=12).to(torch.bfloat16).to(self.device)
            generation_config = dict(max_new_tokens=1024, do_sample=True)


            question = '<image>\nPlease describe detailed the image in a paragraph'
            response, history = self.model.chat(self.tokenizer, image, question, generation_config, history=None, return_history=True)
            return response
    def generate__short_caption(self, image_path):
        with torch.no_grad():
            image = self.load_image(image_path, max_num=12).to(torch.bfloat16).to(self.device)
            generation_config = dict(max_new_tokens=512, do_sample=True)


            question = '<image>\nPlease describe the unique features espescially the posting human or the features of that image of the image in one or two sentence'
            response, history = self.model.chat(self.tokenizer, image, question, generation_config, history=None, return_history=True)
            return response
    
    
    def generate_captions(self, image_paths):
        num_patches_list = []
        final_pixels = None  # <- initialize here

        for image_path in image_paths:
            
            pixels = self.load_image(image_path, max_num=12).to(torch.bfloat16).to(self.device)
            num_patches_list.append(pixels.size(0))
            final_pixels = torch.cat((final_pixels, pixels), dim=0) if final_pixels is not None else pixels

        questions = ['<image>\nPlease describe the image in a very paragraph'] * len(num_patches_list)
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        responses = self.model.batch_chat(
            self.tokenizer,
            final_pixels,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config= generation_config
        )
        return responses

    
    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False):
        
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')
        

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        

        model_inputs = tokenizer(query, return_tensors='pt')
        
        device = torch.device(self.model.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        
        
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.model.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.model.extract_feature(pixel_values)
            
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        outputs = self.model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    
    def get_inputs_embeddings(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ):
        assert self.model.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.model.extract_feature(pixel_values)
            
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        
        return input_embeds
    

    
    def get_embedding(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False):
        
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        

        model_inputs = tokenizer(query, return_tensors='pt')
        
        device = torch.device(self.model.language_model.device if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        
        
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        embeddings = self.get_inputs_embeddings(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        return embeddings, attention_mask
    
        
        

        
    
