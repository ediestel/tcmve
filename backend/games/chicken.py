# games/chicken.py
# Chicken (Game of Chicken / Hawk-Dove) — Brinkmanship + eIQ Boost
# @ECKHART_DIESTEL | nTGT 2.0

"""
NARRATIVE DESCRIPTION: The Game of Chicken operationalizes Thomistic fortitude (fortitudo) versus temperance (temperantia) in the face of mutual destruction, analogical to beings testing their participation in being against the abyss of non-existence. The Generator and Verifier charge toward confrontation like cars on a collision course, where holding firm demonstrates courageous adherence to truth's essence, while swerving reveals prudent restraint that preserves esse. Mutual destruction (both continuing) mirrors the chaos of beings disconnected from their final cause, while mutual swerving represents temperate harmony. This game teaches that true fortitude is not reckless bravado but the wisdom to know when to stand firm for transcendent truth and when to yield for the greater good of continued participation in being.
"""

import logging
from typing import Dict, Any
import numpy as np
logger = logging.getLogger("nTGT.games.chicken")

# Chicken payoff matrix
# Returns (player1_payoff, player2_payoff)
# 0 = swerve (cooperate), 1 = continue (defect)
PAYOFF_MATRIX = {
    (0, 0): (2, 2),  # both swerve: tie, some loss of face
    (0, 1): (0, 3),  # swerve vs continue: swerver loses face, continuer wins
    (1, 0): (3, 0),  # continue vs swerve: continuer wins, swerver loses
    (1, 1): (1, 1),  # both continue: crash, both lose badly
}

def play_chicken(query: str, context: Any = None) -> Dict[str, Any]:
    """
    Chicken game analysis: Brinkmanship in debate.
    Models Generator vs Verifier in high-stakes confrontation.
    Who will back down first? Who will hold firm?
    """
    if not context or not isinstance(context, dict):
        return {"nash_equilibrium": "insufficient_context", "eIQ_boost": 0.0}

    proposition = context.get("proposition", "")
    refutation = context.get("refutation", "")
    virtue_vectors = context.get("virtue_vectors", {})
    round_num = context.get("round", 1)

    gen_virtues = virtue_vectors.get("generator", {})
    ver_virtues = virtue_vectors.get("verifier", {})

    # Fortitude (F) = willingness to hold firm (continue)
    # Temperance (T) = restraint, willingness to swerve
    gen_fortitude = gen_virtues.get("F", 0)
    gen_temperance = gen_virtues.get("T", 0)
    ver_fortitude = ver_virtues.get("F", 0)
    ver_temperance = ver_virtues.get("T", 0)

    # Decision: higher fortitude = more likely to continue (defect)
    gen_threshold = 4.0 + np.random.normal(0, 0.5)  # some randomness
    ver_threshold = 4.0 + np.random.normal(0, 0.5)

    gen_move = 1 if gen_fortitude > gen_threshold else 0  # 1=continue, 0=swerve
    ver_move = 1 if ver_fortitude > ver_threshold else 0

    # Calculate payoffs
    gen_payoff, ver_payoff = PAYOFF_MATRIX[(gen_move, ver_move)]

    # Nash equilibrium analysis
    if gen_move == 0 and ver_move == 0:
        nash_type = "mutual_swerve"
        eIQ_boost = 0.25  # both back down, compromise
    elif gen_move == 1 and ver_move == 1:
        nash_type = "mutual_crash"
        eIQ_boost = 0.40  # Nash equilibrium, both hold firm
    else:
        nash_type = "asymmetric_outcome"
        eIQ_boost = 0.35  # one wins, one loses

    # Virtue adjustments
    virtue_adjustments = {
        "generator": {
            "F": 0.2 if gen_move == 1 else -0.1,  # fortitude boost if held firm
            "T": 0.1 if gen_move == 0 else -0.1,  # temperance if backed down
        },
        "verifier": {
            "F": 0.2 if ver_move == 1 else -0.1,
            "T": 0.1 if ver_move == 0 else -0.1,
        }
    }

    result = {
        "game": "chicken",
        "nash_equilibrium": nash_type,
        "generator_move": "continue" if gen_move == 1 else "swerve",
        "verifier_move": "continue" if ver_move == 1 else "swerve",
        "payoffs": {"generator": gen_payoff, "verifier": ver_payoff},
        "eIQ_boost": eIQ_boost,
        "virtue_adjustments": virtue_adjustments,
        "narrative": __doc__,
        "status": "analyzed"
    }

    logger.info(f"Chicken | Round {round_num} | {nash_type} | Payoffs: G{gen_payoff}-V{ver_payoff}")
    return result