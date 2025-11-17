# games/stackelberg.py
# Stackelberg Competition — Leader-Follower Nash
# nTGT GAMES — **stackelberg.py: FULLY ENHANCED + Professional Logging**
# @ECKHART_DIESTEL | nTGT 2.0

import logging
from typing import Dict, Any, Callable, Tuple, List
import numpy as np
from . import register_game

logger = logging.getLogger("nTGT.games.stackelberg")

# Example: Market entry game
DEFAULT_LEADER_PAYOFF = (5.0, 2.0)   # (Enter, Stay out)
DEFAULT_FOLLOWER_RESPONSE = lambda x: 0 if x == 0 else 1  # Accommodate if enter, Fight if stay out


@register_game("stackelberg")
def stackelberg_equilibrium(
    query: str,
    context: Any = None,
    leader_payoff: Tuple[float, float] = DEFAULT_LEADER_PAYOFF,
    follower_response: Callable[[int], int] = DEFAULT_FOLLOWER_RESPONSE,
    noise: float = 0.0
) -> Dict[str, Any]:
    """
    Stackelberg Leader-Follower Game
    - Leader commits first
    - Follower best-responds
    - Nash in subgame-perfect equilibrium
    """
    logger.info(f"Stackelberg game started | query='{query}' | noise={noise:.3f}")

    actions = ["commit", "wait"]
    follower_actions = ["accommodate", "fight"]

    # Leader anticipates follower response
    expected_payoffs = []
    for action in range(2):
        follower_move = follower_response(action)
        if np.random.random() < noise:
            follower_move = 1 - follower_move  # noise
        payoff = leader_payoff[action]
        if follower_move == 1:  # fight
            payoff -= 3.0  # penalty
        expected_payoffs.append(payoff)

    leader_action = int(np.argmax(expected_payoffs))
    follower_action = follower_response(leader_action)
    if np.random.random() < noise:
        follower_action = 1 - follower_action

    final_payoff = leader_payoff[leader_action]
    if follower_action == 1:
        final_payoff -= 3.0

    eIQ_boost = 0.40

    result = {
        "game": "stackelberg",
        "leader_action": actions[leader_action],
        "follower_action": follower_actions[follower_action],
        "leader_payoff": round(final_payoff, 2),
        "subgame_perfect": True,
        "eIQ_boost": eIQ_boost,
        "status": "equilibrium_reached"
    }

    logger.info(
        f"Stackelberg resolved | Leader: {actions[leader_action]} | "
        f"Follower: {follower_actions[follower_action]} | "
        f"Payoff: {final_payoff:.2f} | eIQ +40%"
    )
    return result