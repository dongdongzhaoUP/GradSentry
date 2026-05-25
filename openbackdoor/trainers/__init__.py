from .trainer import Trainer
from .casual_trainer import CasualTrainer
from .casual_ga_trainer import CasualGATrainer
from .casual_cleangen_trainer import CasualCleanGenTrainer
TRAINERS = {
    "base": Trainer,
    "casual":CasualTrainer,
    "casualga":CasualGATrainer,
    "casualcleangen":CasualCleanGenTrainer
}



def load_trainer(config) -> Trainer:
    return TRAINERS[config["name"].lower()](**config)
