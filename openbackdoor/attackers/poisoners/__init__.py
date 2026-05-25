from .poisoner import Poisoner
from .badnets_poisoner import BadNetsPoisoner, GenerativeBadnetsPoisoner
from .addsent_poisoner import AddSentPoisoner, GenerativeAddSentPoisoner
from .cba_poisoner import CBAPoisoner
from .stylebkd_poisoner import StyleBkdPoisoner, GenerativeStyleBkdPoisoner

POISONERS = {
    "base": Poisoner,
    "badnets": BadNetsPoisoner,
    "addsent": AddSentPoisoner,
    'cba': CBAPoisoner,
    'generativebadnets': GenerativeBadnetsPoisoner,
    'generativeaddsent': GenerativeAddSentPoisoner,
    'stylebkd': StyleBkdPoisoner,
    'generativestylebkd': GenerativeStyleBkdPoisoner,
}

def load_poisoner(config) -> Poisoner:
    return POISONERS[config["name"].lower()](**config)
