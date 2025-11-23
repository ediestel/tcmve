# games/repeated_pd.py
# Repeated Prisoner's Dilemma — Reputation & Tit-for-Tat + eIQ Boost
# @ECKHART_DIESTEL | nTGT 2.0

"""
NARRATIVE DESCRIPTION: The Repeated Prisoner's Dilemma extends the single-round dilemma into Thomistic eschatology, where reputation and forgiveness build eternal character through iterative participation in being. Tit-for-tat strategies mirror divine justice tempered by charity (caritas), where defection is met with proportionate response but forgiveness allows redemption, analogical to God's mercy in the face of human sin. Over rounds, cooperative equilibria emerge as beings learn to trust in shared esse rather than defect into isolated potency, teaching that true fortitude lies not in momentary victories but in the patient cultivation of virtuous relationships that transcend individual games toward transcendent unity.
"""

import logging
from typing import Dict, Any, List
import numpy as np
logger = logging.getLogger("nTGT.games.repeated_pd")

# Single round PD payoff matrix
PAYOFF_MATRIX = {
    (0, 0): (3, 3),  # cooperate, cooperate
    (0, 1): (0, 5),  # cooperate, defect
    (1, 0): (5, 0),  # defect, cooperate
    (1, 1): (1, 1),  # defect, defect
}

def play_repeated_pd(query: str, context: Any = None) -> Dict[str, Any]:
    """
    Repeated Prisoner's Dilemma: Builds reputation over multiple rounds.
    Models long-term Generator vs Verifier relationship with tit-for-tat dynamics.
    """
    if not context or not isinstance(context, dict):
        return {"nash_equilibrium": "insufficient_context", "eIQ_boost": 0.0}

    proposition = context.get("proposition", "")
    refutation = context.get("refutation", "")
    virtue_vectors = context.get("virtue_vectors", {})
    round_num = context.get("round", 1)
    history = context.get("game_history", {}).get("repeated_pd", [])

    gen_virtues = virtue_vectors.get("generator", {})
    ver_virtues = virtue_vectors.get("verifier", {})

    # Tit-for-tat strategy: mirror opponent's previous move
    gen_last_move = history[-1].get("verifier_move", 0) if history else 0  # start cooperative
    ver_last_move = history[-1].get("generator_move", 0) if history else 0

    # Add some virtue-based deviation
    gen_love = gen_virtues.get("L", 0)  # charity influences cooperation
    ver_love = ver_virtues.get("L", 0)

    # Probability of cooperating despite defection (forgiveness)
    gen_forgive_prob = min(0.3, gen_love / 10)
    ver_forgive_prob = min(0.3, ver_love / 10)

    # Current moves with tit-for-tat + forgiveness
    gen_move = 0 if (gen_last_move == 0 or np.random.random() < gen_forgive_prob) else 1
    ver_move = 0 if (ver_last_move == 0 or np.random.random() < ver_forgive_prob) else 1

    # Calculate payoffs
    gen_payoff, ver_payoff = PAYOFF_MATRIX[(gen_move, ver_move)]

    # Cooperation rate over history
    total_rounds = len(history) + 1
    gen_coop_rate = sum(1 for h in history if h.get("generator_move") == 0) + (1 if gen_move == 0 else 0)
    ver_coop_rate = sum(1 for h in history if h.get("verifier_move") == 0) + (1 if ver_move == 0 else 0)
    gen_coop_rate /= total_rounds
    ver_coop_rate /= total_rounds

    # Folk theorem: high cooperation leads to better outcomes
    avg_coop = (gen_coop_rate + ver_coop_rate) / 2
    if avg_coop > 0.8:
        nash_type = "cooperative_equilibrium"
        eIQ_boost = 0.50  # folk theorem payoff
    elif avg_coop > 0.5:
        nash_type = "mixed_reputation"
        eIQ_boost = 0.35
    else:
        nash_type = "defection_spiral"
        eIQ_boost = 0.25

    # Virtue adjustments based on cooperation
    virtue_adjustments = {
        "generator": {
            "L": 0.15 if gen_coop_rate > ver_coop_rate else -0.1,  # love/charity if more cooperative
            "J": 0.1 if gen_move == ver_move else 0.0,  # justice in reciprocity
        },
        "verifier": {
            "L": 0.15 if ver_coop_rate > gen_coop_rate else -0.1,
            "J": 0.1 if ver_move == gen_move else 0.0,
        }
    }

    # Update history for next round
    current_round = {
        "round": round_num,
        "generator_move": gen_move,
        "verifier_move": ver_move,
        "payoffs": {"generator": gen_payoff, "verifier": ver_payoff}
    }

    result = {
        "game": "repeated_pd",
        "nash_equilibrium": nash_type,
        "generator_move": "cooperate" if gen_move == 0 else "defect",
        "verifier_move": "cooperate" if ver_move == 0 else "defect",
        "payoffs": {"generator": gen_payoff, "verifier": ver_payoff},
        "cooperation_rates": {"generator": round(gen_coop_rate, 2), "verifier": round(ver_coop_rate, 2)},
        "eIQ_boost": eIQ_boost,
        "virtue_adjustments": virtue_adjustments,
        "current_round": current_round,
        "narrative": __doc__,
        "status": "analyzed"
    }

    logger.info(f"Repeated PD | Round {round_num} | {nash_type} | Coop: G{gen_coop_rate:.2f}-V{ver_coop_rate:.2f}")
    return result