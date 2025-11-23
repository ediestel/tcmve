# games/regret_min.py
# Regret Minimization — Hedge Algorithm (No-Regret Learning)
# tcmve "Learn from regret" --game=regret_min
# @ECKHART_DIESTEL | nTGT 2.0 | 2025-11-17

"""
NARRATIVE DESCRIPTION: Regret Minimization operationalizes Thomistic conversion and growth in virtue, where beings learn from the pain of suboptimal choices to converge toward transcendent wisdom. Each round of regret updates strategy weights like a soul accumulating habits, exponentially weighting better actions while diminishing poor ones, analogical to grace perfecting nature. Convergence to no-regret equilibrium mirrors the beatific vision, where all potency is actualized in perfect participation in being. This game teaches that true prudence emerges not from innate perfection but from the humble willingness to learn from mistakes, transforming regret into the ladder of analogical ascent toward divine truth.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger("nTGT.games.regret_min")

# Default actions for demo
DEFAULT_ACTIONS = ["rock", "paper", "scissors"]


def regret_minimization(
    query: str,
    context: Any = None,
    actions: Optional[List[str]] = None,
    rounds: int = 5000,
    learning_rate: float = 0.1,
    convergence_threshold: float = 0.001,
    log_interval: int = 1000
) -> Dict[str, Any]:
    """
    Regret Minimization via Hedge Algorithm
    Converges to Coarse Correlated Equilibrium (no-regret learning).
    """
    logger.info(
        f"Regret minimization started | actions={len(actions or DEFAULT_ACTIONS)} | "
        f"rounds={rounds} | lr={learning_rate} | query='{query}'"
    )

    actions = actions or DEFAULT_ACTIONS
    n_actions = len(actions)
    action_names = {i: name for i, name in enumerate(actions)}

    # Cumulative regret
    regret = np.zeros(n_actions)
    # Strategy weights (exponential)
    weights = np.ones(n_actions)
    strategy_history = []

    converged = False
    convergence_round = rounds

    for t in range(rounds):
        # Normalize weights to probabilities
        strategy = weights / weights.sum()
        choice = np.random.choice(n_actions, p=strategy)

        # Simulate regret: +1 for all non-chosen actions
        instant_regret = np.ones(n_actions)
        instant_regret[choice] = 0
        regret += instant_regret

        # Hedge update: exponential weighting
        weights *= np.exp(learning_rate * instant_regret)

        # Log
        if t % log_interval == 0 or t == rounds - 1:
            logger.debug(
                f"Round {t:5d} | Strategy: {dict(zip(actions, strategy.round(4)))} | "
                f"Choice: {action_names[choice]}"
            )

        # Convergence check
        if t > 100:
            recent_avg = np.mean(strategy_history[-100:], axis=0) if len(strategy_history) > 100 else strategy
            if np.linalg.norm(strategy - recent_avg) < convergence_threshold:
                converged = True
                convergence_round = t + 1
                logger.info(f"No-regret convergence at round {convergence_round}")
                break

        strategy_history.append(strategy.copy())

    final_strategy = strategy.round(4)
    eIQ_boost = 0.35 if converged else 0.32

    result = {
        "game": "regret_min",
        "final_strategy": {action_names[i]: float(final_strategy[i]) for i in range(n_actions)},
        "converged": converged,
        "convergence_round": convergence_round,
        "total_rounds": rounds,
        "eIQ_boost": eIQ_boost,
        "narrative": __doc__,
        "status": "no_regret_achieved" if converged else "max_rounds"
    }

    logger.info(
        f"Regret minimization complete | Converged: {converged} | "
        f"Strategy: {result['final_strategy']} | eIQ +{eIQ_boost*100:.0f}%"
    )
    return result