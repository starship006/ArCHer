import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from dotenv import load_dotenv
import torch.nn.functional as F


load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

def main():
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Global Rank {accelerator.process_index} init accelerator")

    model_path = os.path.expanduser("~/models/Llama-2-7b-hf")
    
    # if main path, load onto cpu
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        cache_dir="./cache",
        # token=HUGGINGFACE_API_KEY,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        cache_dir="./cache",
        # token=HUGGINGFACE_API_KEY,
    )
    accelerator.wait_for_everyone()
    print(f"Global Rank {accelerator.process_index} loaded model to RAM")

    # Prepare the model and tokenizer
    model, tokenizer = accelerator.prepare(model, tokenizer)
    accelerator.wait_for_everyone()
    print(f"Global Rank {accelerator.process_index} Model and tokenizer prepared across devices.")
    
    # example for forward-pass
    prompt = "Hey llama how are you?"
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"Global Rank {accelerator.process_index} model inputs: {inputs}")
    
    outputs = model(inputs, use_cache=False, output_hidden_states=False, output_attentions=False)
    pred_tokens = F.softmax(outputs.logits, dim=-1).argmax(dim=-1)
    print(f"Global Rank {accelerator.process_index} top-1 model outputs: {pred_tokens}")
    
    

if __name__ == "__main__":
    main()
    
    