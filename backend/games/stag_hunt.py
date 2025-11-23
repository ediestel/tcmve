# games/stag_hunt.py
# Stag Hunt (Assurance Game) — Cooperation vs Risk + eIQ Boost
# @ECKHART_DIESTEL | nTGT 2.0

"""
NARRATIVE DESCRIPTION: The Stag Hunt embodies Thomistic faith (fides) and hope (spes) in coordinating toward transcendent truth, where individual safety (hunting hare) conflicts with collective participation in divine plenitude. The Generator and Verifier must choose between the assured mediocrity of isolated skepticism and the risky grandeur of coordinated truth-seeking, analogical to beings deciding whether to pursue individual esse or participate in higher unities. Successful coordination yields abundant truth (the stag), while mismatched efforts result in mutual failure, teaching that true prudence (prudentia) lies not in fearful isolation but in faithful trust that others share the same orientation toward the final cause of being itself.
"""

import logging
from typing import Dict, Any
logger = logging.getLogger("nTGT.games.stag_hunt")

# Stag Hunt payoff matrix
# 0 = hunt hare (safe, low payoff), 1 = hunt stag (risky, high payoff if coordinated)
PAYOFF_MATRIX = {
    (0, 0): (1, 1),  # both hunt hare: safe but low
    (0, 1): (0, 0),  # one hare, one stag: stag hunter gets nothing
    (1, 0): (0, 0),  # one stag, one hare: stag hunter gets nothing
    (1, 1): (3, 3),  # both hunt stag: high payoff
}

def play_stag_hunt(query: str, context: Any = None) -> Dict[str, Any]:
    """
    Stag Hunt analysis: Assurance game in debate.
    Models Generator vs Verifier coordinating on ambitious truth-seeking vs safe skepticism.
    """
    if not context or not isinstance(context, dict):
        return {"nash_equilibrium": "insufficient_context", "eIQ_boost": 0.0}

    proposition = context.get("proposition", "")
    refutation = context.get("refutation", "")
    virtue_vectors = context.get("virtue_vectors", {})
    round_num = context.get("round", 1)

    gen_virtues = virtue_vectors.get("generator", {})
    ver_virtues = virtue_vectors.get("verifier", {})

    # Faith (V) and Hope (H) = willingness to coordinate on ambitious goals
    # Prudence (P) = preference for safe, incremental progress
    gen_faith_hope = (gen_virtues.get("V", 0) + gen_virtues.get("H", 0)) / 2
    gen_prudence = gen_virtues.get("P", 0)
    ver_faith_hope = (ver_virtues.get("V", 0) + ver_virtues.get("H", 0)) / 2
    ver_prudence = ver_virtues.get("P", 0)

    # Decision: higher faith/hope = more likely to hunt stag (coordinate)
    gen_threshold = (gen_faith_hope + gen_prudence) / 2
    ver_threshold = (ver_faith_hope + ver_prudence) / 2

    gen_move = 1 if gen_faith_hope > gen_threshold else 0  # 1=stag, 0=hare
    ver_move = 1 if ver_faith_hope > ver_threshold else 0

    # Calculate payoffs
    gen_payoff, ver_payoff = PAYOFF_MATRIX[(gen_move, ver_move)]

    # Nash equilibrium analysis
    if gen_move == 0 and ver_move == 0:
        nash_type = "mutual_hare"
        eIQ_boost = 0.20  # safe but low ambition
    elif gen_move == 1 and ver_move == 1:
        nash_type = "mutual_stag"
        eIQ_boost = 0.45  # coordinated high payoff
    else:
        nash_type = "miscoordination"
        eIQ_boost = 0.15  # failed coordination, both lose

    # Virtue adjustments
    virtue_adjustments = {
        "generator": {
            "V": 0.2 if gen_move == 1 and ver_move == 1 else (-0.1 if gen_move == 1 else 0.1),  # faith rewarded if coordinated
            "H": 0.2 if gen_move == 1 and ver_move == 1 else (-0.1 if gen_move == 1 else 0.1),  # hope rewarded if coordinated
            "P": 0.1 if gen_move == 0 else -0.1,  # prudence if played safe
        },
        "verifier": {
            "V": 0.2 if ver_move == 1 and gen_move == 1 else (-0.1 if ver_move == 1 else 0.1),
            "H": 0.2 if ver_move == 1 and gen_move == 1 else (-0.1 if ver_move == 1 else 0.1),
            "P": 0.1 if ver_move == 0 else -0.1,
        }
    }

    result = {
        "game": "stag_hunt",
        "nash_equilibrium": nash_type,
        "generator_move": "stag" if gen_move == 1 else "hare",
        "verifier_move": "stag" if ver_move == 1 else "hare",
        "payoffs": {"generator": gen_payoff, "verifier": ver_payoff},
        "eIQ_boost": eIQ_boost,
        "virtue_adjustments": virtue_adjustments,
        "narrative": __doc__,
        "status": "analyzed"
    }

    logger.info(f"Stag Hunt | Round {round_num} | {nash_type} | Payoffs: G{gen_payoff}-V{ver_payoff}")
    return result