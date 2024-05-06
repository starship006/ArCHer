# from .text_nav_env import ContextualTextNavEnv
from .env_utils import batch_interact_environment
from .twenty_questions import TwentyQuestionsEnv, BatchedTwentyQuestionsEnv
# from .craiglist import CraiglistEnv, BatchedCraiglistEnv
#from .adventure_env import BatchedAdventureEnv
# from .alfworld_env import BatchedAlfWorldEnv
from .guess_my_city import BatchedGuessMyCityEnv
from .webshop import BatchedWebShopEnv
from .llm_twenty_questions_subset import LLMBatchedTwentyQuestionsEnv

import sys
from os.path import dirname, abspath
grand_grandparent_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
print(grand_grandparent_dir)
sys.path.insert(0, grand_grandparent_dir)
from seller_env import BatchedSellerEnv