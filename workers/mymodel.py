from workers.base import *



class MyModelWorker(BaseWorker):

    def load_model_and_tokenizer(self, load_config, use_eagle):
        # TODO: load your model here
        hf_model_config = {"pretrained_model_name_or_path": load_config['config_dir'],'trust_remote_code': True, 'low_cpu_mem_usage': True}
        hf_tokenizer_config = {"pretrained_model_name_or_path": load_config['config_dir'], 'padding_side': 'left', 'trust_remote_code': True}
        precision = load_config.get('precision', 'fp16')
        device = load_config.get('device', 'cuda')
        assert device == "cuda", 'only supports CUDA inference'

        if precision == 'fp16':
            hf_model_config.update({"torch_dtype": torch.float16})
        if use_eagle is False:
            model = AutoModelForCausalLM.from_pretrained(**hf_model_config)
            tokenizer = AutoTokenizer.from_pretrained(**hf_tokenizer_config)
            model.eval()
        else:
            model = None
            tokenizer = AutoTokenizer.from_pretrained(**hf_tokenizer_config)
        return model, tokenizer
    
    @property
    def system_prompt(self):
        return "你是一个医疗领域的人工智能助手。"
    @property
    def instruction_template(self):
        return self.system_prompt + '问：{instruction}\n答：'
    @property
    def instruction_template_with_fewshot(self,):
        return self.system_prompt + '{fewshot_examples}问：{instruction}\n答：'
    @property
    def fewshot_template(self):
        return "问：{user}\n答：{gpt}\n"
    
    def collate_conv(self, data):
        raise NotImplementedError
    
    
