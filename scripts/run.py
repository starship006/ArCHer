import torch
import transformers
from tqdm import tqdm
import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from archer.environment import TwentyQuestionsEnv, BatchedTwentyQuestionsEnv, BatchedGuessMyCityEnv, BatchedWebShopEnv, SellerEnv
from archer.models import ArcherAgent, CHAIAgent
from archer.algorithms import offpolicy_train_loop
from archer.prompts import MISTRAL_TWENTY_QUESTIONS_TEMPLATE, mistral_twenty_questions_decode_actions
from archer.utils import colorful_print
import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
transformers.logging.set_verbosity_error()

CONFIG_NAME = "archer_20q"
@hydra.main(version_base=None, config_path="./config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    print("hiiiii")
    colorful_print(">>> Configuration file: "+CONFIG_NAME+"<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device

    # load environment
    if config.env_name == "twenty_questions":
        env = BatchedTwentyQuestionsEnv(env_load_path=config.env_load_path, 
                                        device=device, 
                                        cache_dir=config.cache_dir)
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
        raise NotImplementedError("Seller environment is not implemented.")
        
    else:
        raise NotImplementedError("Environment not implemented.")
    decode_f = lambda x:x
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
                            eos_str='\n')
    elif config.agent_type.lower() == "archer_llm":
        #only twenty questions is supported for LLM ArCHer
        print(">>> Using ArCHer agent with LLM")
        agent = ArcherAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens,
                            TEMPLATE=MISTRAL_TWENTY_QUESTIONS_TEMPLATE, use_lora=config.use_lora,
                            eos_str=config.eos_str)
        decode_f = mistral_twenty_questions_decode_actions
    elif config.agent_type.lower() == "online_filteredbc":
        print(">>> Using Online FilteredBC agent")
        # the agent is the same as ArCHer, only the trainer will be different
        agent = ArcherAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens)
    else:
        raise NotImplementedError("Agent not implemented.")
    tokenizer = agent.tokenizer
    if config.checkpoint_path is not None:
        state_dict = torch.load(config.checkpoint_path, map_location=device)['model_state_dict']
        agent.model.load_state_dict(state_dict)
    # agent = accelerator.prepare(agent)

    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    offpolicy_train_loop(env = env,
                agent = agent,
                tokenizer = tokenizer,
                eval_env = eval_env,
                accelerator = accelerator,
                decode_f=decode_f,
                **config)


if __name__ == "__main__":
    main()
