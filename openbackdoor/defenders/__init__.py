from .defender import Defender
from .cube_defender import CUBEDefender, CasualCUBEDefender
from .graceful_defender import GraCeFulDefender
from .svd_defender import SVDDefender
from .onion_defender import ONIONDefender
from .strip_defender import STRIPDefender
from .rap_defender import RAPDefender

DEFENDERS = {
    "base": Defender,
    'cube': CUBEDefender,
    'casualcube': CasualCUBEDefender,
    'graceful': GraCeFulDefender,
    'svd': SVDDefender,
    'onion': ONIONDefender,
    'strip': STRIPDefender,
    'rap': RAPDefender,
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
