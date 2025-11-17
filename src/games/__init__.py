from typing import Dict, Any, Callable
import logging
from typing import Dict, Any, Callable
from .auction import auction_nash
from .stackelberg import stackelberg_equilibrium
from .evolution import evolutionary_stable_strategy
from .regret import regret_minimization
from .shadow import shadow_play
from .multiplay import multi_agent_play


logger = logging.getLogger("nTGT.games")
GAME_REGISTRY: Dict[str, Callable] = {
    "auction": auction_nash,
    "stackelberg": stackelberg_equilibrium,
    "evolution": evolutionary_stable_strategy,
    "regret_min": regret_minimization,
    "shadow_play": shadow_play,
    "muliplay": multi_agent_play,
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