# games/auction.py
# Auction game theory applied to debate rounds

"""
NARRATIVE DESCRIPTION: The Auction Game operationalizes Thomistic justice (justitia) in the valuation and allocation of truth claims, where Generator and Verifier bid their virtue-strength on propositions, analogical to beings offering their participation in being to divine judgment. The reserve price represents the minimum esse required for a claim to participate in truth, while overbidding demonstrates imprudent excess and underbidding reveals insufficient commitment. Successful auctions reward proportional virtue with truth's possession, teaching that true wisdom lies in bidding not with greedy self-interest but with measured justice that reflects the intrinsic worth of each truth-claim as a participation in transcendent being.
"""

from typing import Dict, Any
import numpy as np
import re

def auction_nash(query: str, context: Any = None) -> Dict[str, Any]:
    """
    Auction theory applied to debate: Generator and Verifier "bid" on truth claims.
    Higher virtue bids win rounds, but overbidding leads to inefficiency.
    """
    if not context or not isinstance(context, dict):
        return {"nash_equilibrium": "insufficient_context", "eIQ_boost": 0.0}

    proposition = context.get("proposition", "")
    refutation = context.get("refutation", "")
    virtue_vectors = context.get("virtue_vectors", {})

    # Calculate "bids" based on virtue strength
    gen_virtues = virtue_vectors.get("generator", {})
    ver_virtues = virtue_vectors.get("verifier", {})

    gen_bid = gen_virtues.get("P", 0) * 0.4 + gen_virtues.get("J", 0) * 0.3 + gen_virtues.get("F", 0) * 0.3
    ver_bid = ver_virtues.get("P", 0) * 0.4 + ver_virtues.get("J", 0) * 0.3 + ver_virtues.get("F", 0) * 0.3

    # Reserve price based on query complexity
    complexity_indicators = len(re.findall(r'\b(and|or|but|however|therefore|thus|hence)\b', query.lower()))
    reserve_price = 2.0 + complexity_indicators * 0.5

    # Determine winner and payment
    bids = [gen_bid, ver_bid]
    winner_idx = np.argmax(bids)
    winner = "generator" if winner_idx == 0 else "verifier"
    loser = "verifier" if winner_idx == 0 else "generator"

    # First-price auction: winner pays their bid
    payment = bids[winner_idx]

    # Check if bids meet reserve
    if payment < reserve_price:
        nash_type = "no_sale_reserve_not_met"
        eIQ_boost = 0.20
        winner = None
    else:
        # Nash equilibrium analysis
        bid_difference = abs(gen_bid - ver_bid)
        if bid_difference < 1.0:
            nash_type = "competitive_equilibrium"
            eIQ_boost = 0.35
        elif winner == "generator":
            nash_type = "generator_dominance"
            eIQ_boost = 0.30
        else:
            nash_type = "verifier_dominance"
            eIQ_boost = 0.30

    # Virtue adjustments
    virtue_adjustments = {}
    if winner:
        # Winner gains confidence, loser learns caution
        virtue_adjustments[winner] = {"F": 0.15, "P": 0.1}
        virtue_adjustments[loser] = {"T": 0.1, "V": 0.05}  # Temperance and vigilance
    else:
        # Both learn from failed auction
        virtue_adjustments["generator"] = {"J": 0.1, "P": 0.05}
        virtue_adjustments["verifier"] = {"J": 0.1, "P": 0.05}

    result = {
        "game": "auction",
        "nash_equilibrium": nash_type,
        "winner": winner,
        "bids": {"generator": round(gen_bid, 2), "verifier": round(ver_bid, 2)},
        "reserve_price": round(reserve_price, 2),
        "payment": round(payment, 2) if winner else 0,
        "eIQ_boost": eIQ_boost,
        "virtue_adjustments": virtue_adjustments,
        "narrative": __doc__,
        "status": "auctioned"
    }

    return result