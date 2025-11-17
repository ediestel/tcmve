# games/evolution.py
# Evolutionary Stable Strategy — Full Replicator Dynamics
# tcmve "Evolve strategy" --game=evolution
# @ECKHART_DIESTEL | nTGT 2.0 | 2025-11-17

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from . import register_game

logger = logging.getLogger("nTGT.games.evolution")

# Default Hawk-Dove payoff matrix
HAWK_DOVE = np.array([
    [1, 3],  # Hawk vs [Hawk, Dove]
    [0, 2]   # Dove vs [Hawk, Dove]
], dtype=float)


@register_game("evolution")
def evolutionary_stable_strategy(
    query: str,
    context: Any = None,
    payoff_matrix: Optional[np.ndarray] = None,
    rounds: int = 5000,
    mutation_rate: float = 0.01,
    convergence_threshold: float = 0.001,
    log_interval: int = 1000
) -> Dict[str, Any]:
    """
    Evolutionary Game Theory — Replicator Dynamics
    Converges to Evolutionary Stable Strategy (ESS).
    """
    logger.info(
        f"Evolution started | rounds={rounds} | mutation={mutation_rate:.3f} | "
        f"threshold={convergence_threshold} | query='{query}'"
    )

    payoff = payoff_matrix if payoff_matrix is not None else HAWK_DOVE
    n_strategies = payoff.shape[0]
    strategy_names = [f"strategy_{i}" for i in range(n_strategies)]

    # Initial population (uniform)
    population = np.ones(n_strategies) / n_strategies
    history: List[Dict[str, Any]] = []

    converged = False
    convergence_round = rounds

    for t in range(rounds):
        # Fitness = payoff × population
        fitness = payoff @ population
        avg_fitness = np.dot(population, fitness)

        # Replicator equation
        delta = population * (fitness - avg_fitness)
        population += delta

        # Mutation (genetic drift)
        if np.random.random() < mutation_rate:
            mutation = np.random.dirichlet(np.ones(n_strategies) * 0.1)
            population = 0.95 * population + 0.05 * mutation

        # Normalize
        population = np.clip(population, 1e-10, None)
        population /= population.sum()

        # Log
        if t % log_interval == 0 or t == rounds - 1:
            logger.debug(
                f"Generation {t:5d} | Population: {dict(zip(strategy_names, population.round(4)))}"
            )

        # Convergence check
        if t > 100 and len(history) > 50:
            recent = np.mean([h["population"] for h in history[-50:]], axis=0)
            if np.linalg.norm(population - recent) < convergence_threshold:
                converged = True
                convergence_round = t + 1
                logger.info(f"ESS converged at generation {convergence_round}")
                break

        history.append({"generation": t, "population": population.copy()})

    # Determine dominant strategy
    dominant = np.argmax(population)
    ess_type = strategy_names[dominant] if population[dominant] > 0.8 else "mixed"

    eIQ_boost = 0.40 if converged else 0.38

    result = {
        "game": "evolution",
        "ess_type": ess_type,
        "final_population": {strategy_names[i]: round(float(p), 4) for i, p in enumerate(population)},
        "converged": converged,
        "convergence_generation": convergence_round,
        "total_generations": rounds,
        "eIQ_boost": eIQ_boost,
        "status": "ess_reached" if converged else "max_generations"
    }

    logger.info(
        f"Evolution complete | ESS: {ess_type} | "
        f"Population: {result['final_population']} | eIQ +{eIQ_boost*100:.0f}%"
    )
    return result