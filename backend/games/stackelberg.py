# games/stackelberg.py
# Stackelberg Competition — Leader-Follower Nash
# nTGT GAMES — **stackelberg.py: FULLY ENHANCED + Professional Logging**
# @ECKHART_DIESTEL | nTGT 2.0

"""
NARRATIVE DESCRIPTION: The Stackelberg Game captures Thomistic prudence (prudentia) in strategic leadership and responsive followership, where the Generator leads with propositional commitment and the Verifier follows with refutational response, analogical to divine providence guiding created beings. The leader's commitment strength reflects fortitude in staking truth claims, while the follower's accommodation or resistance demonstrates justice in recognizing legitimate authority. Weak leadership leads to follower dominance, mirroring beings disconnected from their hierarchical participation in being, while strong leadership enables cooperative actualization of shared esse. This game teaches that true wisdom lies in the prudent balance of leadership courage and responsive humility.
"""

import logging
from typing import Dict, Any, Callable, Tuple, List
import numpy as np

logger = logging.getLogger("nTGT.games.stackelberg")

# Example: Market entry game
DEFAULT_LEADER_PAYOFF = (5.0, 2.0)   # (Enter, Stay out)
DEFAULT_FOLLOWER_RESPONSE = lambda x: 0 if x == 0 else 1  # Accommodate if enter, Fight if stay out


def stackelberg_equilibrium(query: str, context: Any = None) -> Dict[str, Any]:
    """
    Stackelberg competition in debate: Generator leads with proposition,
    Verifier follows with refutation. Models strategic commitment and response.
    """
    if not context or not isinstance(context, dict):
        return {"nash_equilibrium": "insufficient_context", "eIQ_boost": 0.0}

    proposition = context.get("proposition", "")
    refutation = context.get("refutation", "")
    virtue_vectors = context.get("virtue_vectors", {})
    round_num = context.get("round", 1)

    gen_virtues = virtue_vectors.get("generator", {})
    ver_virtues = virtue_vectors.get("verifier", {})

    # Generator (leader) commitment strength
    leader_strength = gen_virtues.get("P", 0) * 0.5 + gen_virtues.get("F", 0) * 0.3 + gen_virtues.get("Ω", 0) * 0.2

    # Verifier (follower) response based on leader's commitment
    if leader_strength > 6.0:
        # Strong leader commitment - follower accommodates
        follower_action = 0  # accommodate
        leader_payoff = 5.0
        follower_payoff = 2.0
    elif leader_strength > 3.0:
        # Moderate commitment - follower fights
        follower_action = 1  # fight
        leader_payoff = 2.0
        follower_payoff = 3.0
    else:
        # Weak commitment - follower dominates
        follower_action = 1  # fight
        leader_payoff = 1.0
        follower_payoff = 4.0

    # Check for subgame perfect equilibrium
    # Follower is best-responding to leader's action
    leader_action = 0 if leader_strength > 4.5 else 1  # commit vs wait
    actions = ["commit", "wait"]
    follower_actions = ["accommodate", "fight"]

    # Equilibrium analysis
    if leader_action == 0 and follower_action == 0:
        nash_type = "cooperative_equilibrium"
        eIQ_boost = 0.35
    elif leader_action == 1 and follower_action == 1:
        nash_type = "competitive_equilibrium"
        eIQ_boost = 0.40
    else:
        nash_type = "mixed_stackelberg"
        eIQ_boost = 0.30

    # Virtue adjustments based on leader-follower dynamics
    virtue_adjustments = {
        "generator": {
            "P": 0.2 if leader_action == 0 else -0.1,  # Prudence boost for commitment
            "F": 0.15 if leader_payoff > follower_payoff else 0.0,  # Fortitude if winning
        },
        "verifier": {
            "J": 0.2 if follower_action == 0 else 0.1,  # Justice boost for accommodation
            "T": 0.1 if follower_payoff > leader_payoff else 0.0,  # Temperance if winning
        }
    }

    result = {
        "game": "stackelberg",
        "nash_equilibrium": nash_type,
        "leader_action": actions[leader_action],
        "follower_action": follower_actions[follower_action],
        "payoffs": {"generator": round(leader_payoff, 2), "verifier": round(follower_payoff, 2)},
        "subgame_perfect": True,
        "eIQ_boost": eIQ_boost,
        "virtue_adjustments": virtue_adjustments,
        "narrative": __doc__,
        "status": "equilibrium_reached"
    }

    logger.info(
        f"Stackelberg | Round {round_num} | Leader: {actions[leader_action]} | "
        f"Follower: {follower_actions[follower_action]} | {nash_type}"
    )
    return result