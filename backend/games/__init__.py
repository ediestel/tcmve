from typing import Dict, Any, Callable
import logging
from typing import Dict, Any, Callable
from .auction import auction_nash
from .prisoner import play_prisoner_dilemma
from .stackelberg import stackelberg_equilibrium
from .evolution import evolutionary_stable_strategy
from .regret import regret_minimization
from .shadow import shadow_play
from .multiplay import multi_agent_play
from .chicken import play_chicken
from .stag_hunt import play_stag_hunt
from .repeated_pd import play_repeated_pd
from .ultimatum import play_ultimatum


logger = logging.getLogger("nTGT.games")
GAME_REGISTRY: Dict[str, Callable] = {
    "auction": auction_nash,
    "prisoner": play_prisoner_dilemma,
    "stackelberg": stackelberg_equilibrium,
    "evolution": evolutionary_stable_strategy,
    "regret_min": regret_minimization,
    "shadow_play": shadow_play,
    "multiplay": multi_agent_play,
    "chicken": play_chicken,
    "stag_hunt": play_stag_hunt,
    "repeated_pd": play_repeated_pd,
    "ultimatum": play_ultimatum,
}

def register_game(name: str):
    def decorator(func: Callable):
        GAME_REGISTRY[name] = func
        return func
    return decorator

def play_game(game_type: str, query: str, context: Any = None) -> Dict[str, Any]:
    logger.info(f"Game requested: {game_type} | Query: {query}")
    if game_type not in GAME_REGISTRY:
        logger.error(f"Unknown game: {game_type}")
        raise ValueError(f"Game '{game_type}' not found")
    return GAME_REGISTRY[game_type](query, context)