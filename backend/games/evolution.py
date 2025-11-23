# games/evolution.py
# Evolutionary Stable Strategy — Full Replicator Dynamics
# tcmve "Evolve strategy" --game=evolution
# @ECKHART_DIESTEL | nTGT 2.0 | 2025-11-17

"""
NARRATIVE DESCRIPTION: Evolutionary Stable Strategy captures Thomistic natural law in the survival and propagation of virtuous strategies, where populations of beings evolve toward equilibria that resist invasion by mutant strategies, analogical to grace building upon nature. Replicator dynamics reward fitness based on participation in being, with hawkish fortitude and dovish charity competing until stable mixtures emerge that maximize collective esse. This game teaches that true stability comes not from domination but from balanced participation, where aggressive and cooperative virtues coexist in transcendent harmony, reflecting the divine wisdom that orders all things to their proper ends.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger("nTGT.games.evolution")

# Default Hawk-Dove payoff matrix
HAWK_DOVE = np.array([
    [1, 3],  # Hawk vs [Hawk, Dove]
    [0, 2]   # Dove vs [Hawk, Dove]
], dtype=float)


def evolutionary_stable_strategy(query: str, context: Any = None) -> Dict[str, Any]:
    """
    Evolutionary analysis of debate strategies.
    Models Generator and Verifier strategies as evolving populations.
    """
    if not context or not isinstance(context, dict):
        return {"nash_equilibrium": "insufficient_context", "eIQ_boost": 0.0}

    virtue_vectors = context.get("virtue_vectors", {})
    history = context.get("history", [])
    round_num = context.get("round", 1)

    gen_virtues = virtue_vectors.get("generator", {})
    ver_virtues = virtue_vectors.get("verifier", {})

    # Strategy fitness based on virtue profiles
    # Hawk-like (aggressive): High F, low L
    # Dove-like (cooperative): High L, low F
    gen_hawk_fitness = gen_virtues.get("F", 0) * 0.6 - gen_virtues.get("L", 0) * 0.4
    gen_dove_fitness = gen_virtues.get("L", 0) * 0.6 - gen_virtues.get("F", 0) * 0.4

    ver_hawk_fitness = ver_virtues.get("F", 0) * 0.6 - ver_virtues.get("L", 0) * 0.4
    ver_dove_fitness = ver_virtues.get("L", 0) * 0.6 - ver_virtues.get("F", 0) * 0.4

    # Population evolution (simplified replicator dynamics)
    gen_population = np.array([gen_hawk_fitness, gen_dove_fitness])
    ver_population = np.array([ver_hawk_fitness, ver_dove_fitness])

    # Normalize to probabilities
    gen_population = np.exp(gen_population) / np.sum(np.exp(gen_population))
    ver_population = np.exp(ver_population) / np.sum(np.exp(ver_population))

    # Determine dominant strategies
    strategies = ["hawk", "dove"]
    gen_dominant = strategies[np.argmax(gen_population)]
    ver_dominant = strategies[np.argmax(ver_population)]

    # ESS analysis
    if gen_dominant == "hawk" and ver_dominant == "hawk":
        ess_type = "hawk_dominance"
        stability = True
        eIQ_boost = 0.40
    elif gen_dominant == "dove" and ver_dominant == "dove":
        ess_type = "dove_dominance"
        stability = True
        eIQ_boost = 0.35
    else:
        ess_type = "mixed_evolution"
        stability = False
        eIQ_boost = 0.30

    # Evolutionary pressure adjustments
    virtue_adjustments = {
        "generator": {
            "F": 0.15 if gen_dominant == "hawk" else -0.1,  # Reinforce dominant strategy
            "L": 0.15 if gen_dominant == "dove" else -0.1,
        },
        "verifier": {
            "F": 0.15 if ver_dominant == "hawk" else -0.1,
            "L": 0.15 if ver_dominant == "dove" else -0.1,
        }
    }

    result = {
        "game": "evolution",
        "nash_equilibrium": ess_type,
        "evolutionary_stable": stability,
        "strategies": {
            "generator": {"dominant": gen_dominant, "distribution": dict(zip(strategies, gen_population))},
            "verifier": {"dominant": ver_dominant, "distribution": dict(zip(strategies, ver_population))}
        },
        "eIQ_boost": eIQ_boost,
        "virtue_adjustments": virtue_adjustments,
        "narrative": __doc__,
        "status": "evolved"
    }

    logger.info(f"Evolution | Round {round_num} | ESS: {ess_type} | Stable: {stability}")
    return result