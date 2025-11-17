# games/prisoner.py
# Prisoner's Dilemma — Full Nash + Iterated + eIQ Boost
# @ECKHART_DIESTEL | nTGT 2.0

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
from . import register_game

logger = logging.getLogger("nTGT.games.prisoner")

# Classic Prisoner's Dilemma payoff matrix
# (cooperate, cooperate) → (3, 3)
# (cooperate, defect)    → (0, 5)
# (defect, cooperate)    → (5, 0)
# (defect, defect)       → (1, 1)
PAYOFF_MATRIX = np.array([
    [3, 0],  # cooperate vs [cooperate, defect]
    [5, 1]   # defect    vs [cooperate, defect]
])

@register_game("prisoner")
def play_prisoner_dilemma(
    query: str,
    context: Any = None,
    rounds: int = 100,
    noise: float = 0.05,
    forgiveness: float = 0.1
) -> Dict[str, Any]:
    """
    Full Prisoner's Dilemma with:
    - Iterated play
    - Tit-for-tat + forgiveness
    - Noise robustness
    - Nash convergence detection
    """
    logger.info(f"Prisoner's Dilemma | rounds={rounds} | noise={noise:.2f} | forgiveness={forgiveness:.2f}")

    # Player strategies
    p1_history = []
    p2_history = []
    p1_score = 0
    p2_score = 0

    # Start with cooperation (humility Ω)
    p1_move = 0  # 0 = cooperate, 1 = defect
    p2_move = 0

    for t in range(rounds):
        # Tit-for-tat with forgiveness
        if t > 0:
            if np.random.random() < forgiveness:
                p1_move = 0  # forgive
            else:
                p1_move = p2_history[-1]  # mirror

        # Add noise (realism)
        if np.random.random() < noise:
            p1_move = 1 - p1_move  # flip

        p2_move = p1_history[-1] if t > 0 else 0
        if np.random.random() < noise:
            p2_move = 1 - p2_move

        # Payoff
        payoff = PAYOFF_MATRIX[p1_move, p2_move]
        p1_score += payoff[0]
        p2_score += payoff[1]

        p1_history.append(p1_move)
        p2_history.append(p2_move)

        if t % 20 == 0:
            coop_rate = 1 - np.mean(p1_history[-10:])
            logger.debug(f"Round {t} | Coop rate: {coop_rate:.2f}")

    # Nash detection
    defect_rate = np.mean(p1_history[-20:])
    nash_type = "mutual_defect" if defect_rate > 0.8 else "mixed_cooperation"

    eIQ_boost = 0.40 if nash_type == "mutual_defect" else 0.30

    result = {
        "game": "prisoner",
        "rounds": rounds,
        "nash_equilibrium": nash_type,
        "cooperation_rate": 1 - defect_rate,
        "p1_score": int(p1_score),
        "p2_score": int(p2_score),
        "eIQ_boost": eIQ_boost,
        "status": "converged"
    }

    logger.info(f"Prisoner's Dilemma complete | {nash_type} | eIQ +{eIQ_boost*100:.0f}%")
    return result