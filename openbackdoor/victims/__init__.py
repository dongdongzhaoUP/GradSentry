import torch
import torch.nn as nn
from typing import List, Optional
from .victim import Victim
from .casualLLMs import CasualLLMVictim

Victim_List = {
    'casual': CasualLLMVictim
}


def load_victim(config):
    victim = Victim_List[config["type"]](**config)
    return victim