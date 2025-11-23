# games/prisoner.py
# Prisoner's Dilemma — Full Nash + Iterated + eIQ Boost
# @ECKHART_DIESTEL | nTGT 2.0

"""
NARRATIVE DESCRIPTION: The Prisoner's Dilemma in TCMVE embodies the fundamental tension between individual self-interest and collective truth-seeking harmony. In Thomistic terms, mutual cooperation represents analogical participation in divine unity (unum), where Generator and Verifier collaborate toward transcendent truth (verum) rather than defecting into isolated, self-serving falsehoods. Mutual defection, the Nash equilibrium of distrust, mirrors the fallen state of beings disconnected from their final cause, while mixed strategies reveal the potency-act dynamics of imperfect virtue. This game operationalizes actus essendi by rewarding cooperative actualization of shared esse over competitive essence-hoarding, teaching that true fortitude (fortitudo) lies in trusting participation rather than defensive isolation.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
logger = logging.getLogger("nTGT.games.prisoner")

# Classic Prisoner's Dilemma payoff matrix
# Returns (generator_payoff, verifier_payoff)
PAYOFF_MATRIX = {
    (0, 0): (3, 3),  # cooperate, cooperate
    (0, 1): (0, 5),  # cooperate, defect
    (1, 0): (5, 0),  # defect, cooperate
    (1, 1): (1, 1),  # defect, defect
}

def play_prisoner_dilemma(query: str, context: Any = None) -> Dict[str, Any]:
    """
    Prisoner's Dilemma analysis of current debate state.
    Models Generator vs Verifier as prisoners deciding to cooperate or defect.
    """
    if not context or not isinstance(context, dict):
        return {"nash_equilibrium": "insufficient_context", "eIQ_boost": 0.0}

    proposition = context.get("proposition", "")
    refutation = context.get("refutation", "")
    virtue_vectors = context.get("virtue_vectors", {})
    round_num = context.get("round", 1)

    # Analyze debate moves as cooperation/defection
    gen_virtues = virtue_vectors.get("generator", {})
    ver_virtues = virtue_vectors.get("verifier", {})

    # Cooperation metrics (high virtues = cooperation)
    gen_coop = (gen_virtues.get("Ω", 0) + gen_virtues.get("L", 0) + gen_virtues.get("P", 0)) / 3
    ver_coop = (ver_virtues.get("Ω", 0) + ver_virtues.get("L", 0) + ver_virtues.get("P", 0)) / 3

    # Defect detection (aggressive refutation = defection)
    defect_words = ["contradiction", "false", "invalid", "wrong", "fail"]
    ver_defect_score = sum(1 for word in defect_words if word.lower() in refutation.lower())

    # Determine moves
    gen_move = 0 if gen_coop > 5.0 else 1  # cooperate if virtuous
    ver_move = 0 if ver_defect_score < 2 else 1  # cooperate if not too aggressive

    # Calculate payoff
    gen_payoff, ver_payoff = PAYOFF_MATRIX[(gen_move, ver_move)]

    # Nash equilibrium analysis
    if gen_move == 0 and ver_move == 0:
        nash_type = "mutual_cooperation"
        eIQ_boost = 0.35
    elif gen_move == 1 and ver_move == 1:
        nash_type = "mutual_defection"
        eIQ_boost = 0.40  # Nash equilibrium
    else:
        nash_type = "mixed_strategy"
        eIQ_boost = 0.25

    # Virtue adjustments based on game outcome
    virtue_adjustments = {
        "generator": {
            "F": 0.2 if gen_payoff > ver_payoff else -0.1,  # Fortitude boost if winning
            "J": 0.1 if gen_move == ver_move else 0.0,  # Justice alignment
        },
        "verifier": {
            "F": 0.2 if ver_payoff > gen_payoff else -0.1,
            "J": 0.1 if ver_move == gen_move else 0.0,
        }
    }

    result = {
        "game": "prisoner",
        "nash_equilibrium": nash_type,
        "generator_move": "cooperate" if gen_move == 0 else "defect",
        "verifier_move": "cooperate" if ver_move == 0 else "defect",
        "payoffs": {"generator": gen_payoff, "verifier": ver_payoff},
        "eIQ_boost": eIQ_boost,
        "virtue_adjustments": virtue_adjustments,
        "narrative": __doc__,
        "status": "analyzed"
    }

    logger.info(f"Prisoner's Dilemma | Round {round_num} | {nash_type} | Payoffs: G{gen_payoff}-V{ver_payoff}")
    return result