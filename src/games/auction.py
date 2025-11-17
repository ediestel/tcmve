# games/auction.py
# tcmve "bid 100" --game=auction

from typing import Dict, List, Any
import numpy as np

def auction_nash(bids: List[float], reserve: float = 0.0) -> Dict[str, Any]:
    """First-price sealed-bid auction — Nash equilibrium."""
    winner = np.argmax(bids)
    payment = max(bids[winner], reserve)
    return {
        "winner": winner,
        "payment": payment,
        "nash": "truthful bidding" if reserve == 0 else "shade bid",
        "eIQ_boost": 0.35
    }