from datetime import timedelta
from dotenv import load_dotenv
import os
from os.path import dirname, abspath
import sys

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from cycling_utils import TimestampedTimer, InterruptableDistributedSampler, MetricsTracker, AtomicDirectory
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
import wandb
print(os.getcwd())

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from archer.algorithms import offpolicy_train_loop
from archer.environment import TwentyQuestionsEnv, BatchedTwentyQuestionsEnv, BatchedGuessMyCityEnv, BatchedWebShopEnv, BatchedSellerEnv, LLMBatchedTwentyQuestionsEnv
from archer.models import ArcherAgent, CHAIAgent
from archer.prompts import MISTRAL_TWENTY_QUESTIONS_TEMPLATE, MISTRAL_TWENTY_QUESTIONS_SIMPLIFIED_TEMPLATE, mistral_twenty_questions_decode_actions, LLAMA_TWENTY_QUESTIONS_TEMPLATE
from archer.utils import colorful_print

sys.path.insert(0, dirname(dirname(abspath(__file__))))
transformers.logging.set_verbosity_error()
timer = TimestampedTimer()
load_dotenv()
WANDB_API_KEY = os.getenv('WANDB_API_KEY')


CONFIG_NAME = "archer_20q"
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    #colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    #colorful_print(OmegaConf.to_yaml(config), fg='red')
    
    #if config.use_bfloat16:
        #print("hi)")
        #torch.set_default_dtype(torch.bfloat16) # if we don't include this a very weird bug occurs where the lm_optimizer bugs out. dont ask me why this works.
    
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200)) # Set the timeout to 2 hours
    accelerator = Accelerator(kwargs_handlers=[timeout_kwargs])    
    device = accelerator.device
    
    if accelerator.is_main_process:  
        timer.report("imports done | loading envs")
        print("OH YEAH ALSO IM THE ONLY GUY HERE")
    

    # load environment
    if config.env_name == "twenty_questions":
        # LLMBatchedTwentyQuestionsEnv is broken?
        env = BatchedTwentyQuestionsEnv(env_load_path=config.env_load_path, 
                                        device=device, 
                                        cache_dir=config.cache_dir,
                                        bsize=config.batch_size,
                                        simplified=config.simplified)
        eval_env = env
    elif config.env_name == "adventure":
        raise NotImplementedError("Adventure environment is not implemented due to issue in Jericho import.")
        
    elif config.env_name == "guess_my_city":
        env = BatchedGuessMyCityEnv(env_load_path=config.env_load_path, 
                                        device=device, 
                                        cache_dir=config.cache_dir)
        eval_env = env
    elif config.env_name == "webshop":
        env = BatchedWebShopEnv(lower=config.webshop_lower,
                                upper=config.webshop_upper,
                                env_load_path=config.env_load_path)
        eval_env = env
    elif config.env_name == "seller_env":
        env_bs = config.batch_size
        if config.batch_size > 4:
            print("we are currently going to get rate limited! i will need to take this down to 4!")
            env_bs = 4
            
        env = BatchedSellerEnv(bsize = env_bs)
        eval_env = env
        
    else:
        raise NotImplementedError("Environment not implemented.")
    decode_f = lambda x:x
    
    if accelerator.is_main_process:  timer.report("envs done | loading agent")
    
    if config.model_path is not None:
        if config.model_path[0] == "~":
            global_model_path = os.path.expanduser(config.model_path)
        else:
            global_model_path = config.model_path
    else:
        global_model_path = None        
            
    # load decision model
    if config.agent_type.lower() == "chai":
        print(">>> Using CHAI agent")
        agent = CHAIAgent(device=device, accelerator=accelerator, 
                        temperature=config.temperature, 
                        do_sample=config.do_sample, policy_lm=config.policy_lm, 
                        critic_lm=config.critic_lm, cache_dir=config.cache_dir,
                        max_new_tokens=config.max_new_tokens, eos_str = config.eos_str)
        #if use chai, do not update the actor
        config.warmup_iter = config.iterations
    elif config.agent_type.lower() == "archer":
        print(">>> Using ArCHer agent")
        agent = ArcherAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens,
                            eos_str='\n', model_path = global_model_path, quantize=config.quantize)
    elif config.agent_type.lower() == "archer_llm":
        #only twenty questions is supported for LLM ArCHer
        print(">>> Using ArCHer agent with LLM")
        if config.env_name == "twenty_questions":
            if config.simplified:
                print(">>> Using simplified template and environment")
                if "llama" in config.policy_lm.lower():
                    raise NotImplementedError("Generation prompt not implemented.")
                else:
                    template = MISTRAL_TWENTY_QUESTIONS_SIMPLIFIED_TEMPLATE
                
            else:
                print(">>> Using regular template and environment")
                if "llama" in config.policy_lm.lower():
                    template = LLAMA_TWENTY_QUESTIONS_TEMPLATE
                else:
                    template = MISTRAL_TWENTY_QUESTIONS_TEMPLATE
        else:
            template = None
        
        
        agent = ArcherAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens,
                            TEMPLATE=template, use_lora=config.use_lora,
                            eos_str=config.eos_str, model_path = global_model_path,
                            use_bfloat16 = config.use_bfloat16, quantize=config.quantize)
        if config.env_name == "twenty_questions":
            print(">>> Using custom decoding for twenty questions")
            decode_f = mistral_twenty_questions_decode_actions 
        else:
            decode_f = None # mistral_twenty_questions_decode_actions # We don't want to use this!
        
        
    elif config.agent_type.lower() == "online_filteredbc":
        print(">>> Using Online FilteredBC agent")
        # the agent is the same as ArCHer, only the trainer will be different
        agent = ArcherAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens,
                            quantize=config.quantize)
    else:
        raise NotImplementedError("Agent not implemented.")

    accelerator.wait_for_everyone()
        
     
    if accelerator.is_main_process: timer.report("agent done | loading tokenizer")
    tokenizer = agent.tokenizer
    
    if config.checkpoint_path is not None:
        if config.checkpoint_path[-1] != '/':
            config.checkpoint_path += '/'
        print("loading in checkpoints!")
        try:
            state_dict = torch.load(config.checkpoint_path + 'trainer.pt', map_location=device)['model_state_dict'] # map to CPU so that it can fit on memory. should be prepared onto GPU later.
            agent.model.load_state_dict(state_dict)
        except Exception as e:
            print("no checkpoint found, continuing")
            print(e)
            
    # agent = accelerator.prepare(agent)
    if accelerator.is_main_process: timer.report("tokenizer done | loading wandb")
    
    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))
    if accelerator.is_main_process: timer.report("wand done | beginning training")

    offpolicy_train_loop(env = env,
                agent = agent,
                tokenizer = tokenizer,
                eval_env = eval_env,
                accelerator = accelerator,
                decode_f=decode_f,
                timer = timer,
                **config)


if __name__ == "__main__":
    print("ran")
    main()
