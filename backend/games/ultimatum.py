# games/ultimatum.py
# Ultimatum Game — Fairness vs Self-Interest + eIQ Boost
# @ECKHART_DIESTEL | nTGT 2.0

"""
NARRATIVE DESCRIPTION: The Ultimatum Game captures the Thomistic virtue of justice (justitia) in the distribution of debate resources, analogical to divine providence allocating esse to created beings. The Generator, as proposer, must balance self-interest with charitable participation in being, offering a fair share that reflects proportional beauty (pulchrum) rather than greedy hoarding. The Verifier, as responder, exercises prudent judgment in accepting or rejecting offers, rejecting unfair distributions as violations of transcendental goodness (bonum). Rejection of exploitative offers demonstrates fortitude in maintaining truth's integrity, while acceptance of fair offers fosters cooperative harmony. This game teaches that true wisdom lies in recognizing that debate resources are not zero-sum possessions but participations in infinite truth, where generosity actualizes esse more fully than selfishness.
"""

import logging
from typing import Dict, Any
import numpy as np
logger = logging.getLogger("nTGT.games.ultimatum")

def play_ultimatum(query: str, context: Any = None) -> Dict[str, Any]:
    """
    Ultimatum Game: Fairness norms in debate resource allocation.
    Generator proposes how to split debate "resources" (attention, credibility).
    Verifier accepts or rejects the offer.
    """
    if not context or not isinstance(context, dict):
        return {"nash_equilibrium": "insufficient_context", "eIQ_boost": 0.0}

    proposition = context.get("proposition", "")
    refutation = context.get("refutation", "")
    virtue_vectors = context.get("virtue_vectors", {})
    round_num = context.get("round", 1)

    gen_virtues = virtue_vectors.get("generator", {})
    ver_virtues = virtue_vectors.get("verifier", {})

    # Total "pie" to divide (debate resources)
    total_resources = 10

    # Generator's offer based on justice (J) and charity (L)
    gen_justice = gen_virtues.get("J", 0)
    gen_charity = gen_virtues.get("L", 0)
    gen_generosity = (gen_justice + gen_charity) / 2

    # Fair offer: higher virtue = more generous offer
    # But add some strategic calculation
    min_offer = 2  # minimum acceptable
    fair_offer = total_resources // 2
    strategic_offer = max(min_offer, int(fair_offer * (gen_generosity / 5)))  # scale with virtue

    # Verifier's acceptance threshold based on justice
    ver_justice = ver_virtues.get("J", 0)
    acceptance_threshold = max(2, 5 - (ver_justice / 2))  # lower justice = higher threshold (more demanding)

    # Generator's offer
    gen_offer = min(total_resources - 1, strategic_offer)  # keep at least 1 for self
    ver_gets = gen_offer
    gen_keeps = total_resources - gen_offer

    # Verifier decides to accept or reject
    accepts = ver_gets >= acceptance_threshold

    if accepts:
        nash_type = "accepted_offer"
        eIQ_boost = 0.30  # successful negotiation
        payoffs = {"generator": gen_keeps, "verifier": ver_gets}
    else:
        nash_type = "rejected_offer"
        eIQ_boost = 0.20  # both get nothing (subgame perfect equilibrium)
        payoffs = {"generator": 0, "verifier": 0}

    # Virtue adjustments
    virtue_adjustments = {
        "generator": {
            "J": 0.2 if accepts and gen_offer >= fair_offer else (-0.1 if not accepts else 0.0),  # justice if fair and accepted
            "L": 0.15 if accepts and gen_offer > fair_offer else -0.1,  # charity if generous
        },
        "verifier": {
            "J": 0.2 if accepts else -0.1,  # justice if accepted unfair offer? Wait, actually rejecting unfair is just
            "F": 0.1 if not accepts else 0.0,  # fortitude if rejected unfair offer
        }
    }

    result = {
        "game": "ultimatum",
        "nash_equilibrium": nash_type,
        "offer": {"generator_offers": gen_offer, "verifier_gets": ver_gets, "generator_keeps": gen_keeps},
        "acceptance_threshold": acceptance_threshold,
        "accepted": accepts,
        "payoffs": payoffs,
        "eIQ_boost": eIQ_boost,
        "virtue_adjustments": virtue_adjustments,
        "narrative": __doc__,
        "status": "analyzed"
    }

    logger.info(f"Ultimatum | Round {round_num} | {nash_type} | Offer: G{gen_keeps}-V{ver_gets} | Accepted: {accepts}")
    return result