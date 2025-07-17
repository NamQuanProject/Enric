from llama3 import Llama



class LLMAssembler:
    def __init__(self, device = 'cuda:4', model_type = 'llama3'):
        # if model_type == 'qwen':
        #     self.model = QwenModel(device = device)
        # elif model_type == 'llama3':
        #     self.model = Llama(device = device)
        self.model = Llama(device = device)

    
    def assemble(self, prompt: str) -> str:
        return self.model.assemble(prompt)
    
    def question_answer(self, prompt: str) -> str:
        return self.model.question_answer(prompt)
    
    def name_entity_extraction(self,prompt: str) -> str:
        return self.model.name_entity_extraction(prompt)
        