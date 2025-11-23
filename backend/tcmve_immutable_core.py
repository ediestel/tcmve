# † IN NOMINE VERITATIS ET HUMILITATIS Ω †
# This file is placed under perpetual anathema:
# Any modification requires explicit written consent of the original author
# and a 30-day public objection period on GitHub Issues.
# Violation = automatic excommunication from the repository.
# † AD MAJOREM DEI GLORIAM †

"""
TCMVE IMMUTABLE CORE
Canonised Deposit of Truth - Perpetual Anathema Against Modification

This file contains the immutable metaphysical foundations of TCMVE:
- Virtue vector defaults (Thomistic cardinal virtues + theological virtues)
- TLPO 30-flag ontology (Thomistic LLM Parameter Ontology)
- Ω humility formula (recognitio finitudinis)
- Vice-check logic (privation of being safeguard)
- Nash equilibrium conditions (game theory truth convergence)

MODIFICATION PROHIBITED UNDER PAIN OF ANATHEMA
"""

# ============================================================================
# VIRTUE VECTOR DEFAULTS - THOMISTIC CARDINAL + THEOLOGICAL VIRTUES
# ============================================================================

VIRTUE_VECTOR_DEFAULTS = {
    "generator": {
        "P": 8.0,  # Prudence: intellectus agens
        "J": 7.5,  # Justice: voluntas recta
        "F": 6.5,  # Fortitude: ira fortis
        "T": 8.0,  # Temperance: concupiscentia moderata
        "V": 8.5,  # Veritas: ratio speculativa
        "L": 7.2,  # Libertas: libertas arbitrii
        "H": 7.8,  # Hope: spes theological virtue
        "Ω": 30    # Humility: recognitio finitudinis
    },
    "verifier": {
        "P": 9.0,  # Prudence: intellectus agens
        "J": 9.5,  # Justice: voluntas recta
        "F": 8.0,  # Fortitude: ira fortis
        "T": 9.0,  # Temperance: concupiscentia moderata
        "V": 9.0,  # Veritas: ratio speculativa
        "L": 6.5,  # Libertas: libertas arbitrii
        "H": 8.2,  # Hope: spes theological virtue
        "Ω": 35    # Humility: recognitio finitudinis
    },
    "arbiter": {
        "P": 8.5,  # Prudence: intellectus agens
        "J": 8.0,  # Justice: voluntas recta
        "F": 9.0,  # Fortitude: ira fortis
        "T": 8.5,  # Temperance: concupiscentia moderata
        "V": 8.0,  # Veritas: ratio speculativa
        "L": 8.5,  # Libertas: libertas arbitrii
        "H": 8.8,  # Hope: spes theological virtue
        "Ω": 35    # Humility: recognitio finitudinis
    }
}

# ============================================================================
# TLPO 30-FLAG ONTOLOGY - THOMISTIC LLM PARAMETER ONTOLOGY
# ============================================================================

TLPO_33_FLAG_ONTOLOGY = [
    {
        "flag_id": 1,
        "flag_name": "Temperature",
        "thomistic_link": "Potency vs. Act: High temperature embodies potency (pure potential for varied beings/responses), while low embodies act (realization of the most likely, determinate path).",
        "virtue": "prudentia"
    },
    {
        "flag_id": 2,
        "flag_name": "Logit_bias",
        "thomistic_link": "Final Cause: Directs generation toward a telos, akin to how final causes guide beings to their natural end.",
        "virtue": "justitia"
    },
    {
        "flag_id": 3,
        "flag_name": "Do_sample",
        "thomistic_link": "Act-Potency Toggle: False = pure act; True = engages potency, mirroring Thomistic prime act actualizing pure potency.",
        "virtue": "veritas"
    },
    {
        "flag_id": 4,
        "flag_name": "Renormalize_logits",
        "thomistic_link": "Stability of Probabilities: Ensures consistent 'measures' of potency, akin to Thomistic immutable divine essence stabilizing created beings.",
        "virtue": "temperantia"
    },
    {
        "flag_id": 5,
        "flag_name": "Guidance_scale",
        "thomistic_link": "Directed Potency: Scales adherence to essence, like Thomistic divine guidance directing creatures to their end.",
        "virtue": "fortitudo"
    },
    {
        "flag_id": 6,
        "flag_name": "Top_p",
        "thomistic_link": "Limitation of Potency: Restricts infinite potential, mirroring Thomistic matter-form hylomorphism.",
        "virtue": "prudentia"
    },
    {
        "flag_id": 7,
        "flag_name": "Top_k",
        "thomistic_link": "Accidents Modifying Substance: Selects from a fixed set of attributes (top tokens), like accidents inhering in a substance without changing its core essence (prompt).",
        "virtue": "justitia"
    },
    {
        "flag_id": 8,
        "flag_name": "Typical_p",
        "thomistic_link": "Typical Participation: Selects 'common' participations in being, per Thomistic analogy (beings participate typically in higher essences).",
        "virtue": "veritas"
    },
    {
        "flag_id": 9,
        "flag_name": "Epsilon_cutoff",
        "thomistic_link": "Threshold of Potency: Cuts off negligible potentials, analogous to Thomistic minimal essence required for existence (avoids 'non-beings').",
        "virtue": "temperantia"
    },
    {
        "flag_id": 10,
        "flag_name": "Min_p",
        "thomistic_link": "Relative Limitation of Potency: Sets a minimum relative to the highest act, mirroring Thomistic proportional participation in divine perfection.",
        "virtue": "fortitudo"
    },
    {
        "flag_id": 11,
        "flag_name": "Repetition_penalty",
        "thomistic_link": "Unity of Being: Prevents redundant multiplicity, per Thomistic principle that essence is one and indivisible (avoids 'duplicated' beings without purpose).",
        "virtue": "prudentia"
    },
    {
        "flag_id": 12,
        "flag_name": "Frequency_penalty",
        "thomistic_link": "Moderation of Multiplicity: Tempers over-frequent 'participations' in being, akin to Thomistic analogy where beings participate variably in divine essence without excess.",
        "virtue": "justitia"
    },
    {
        "flag_id": 13,
        "flag_name": "Presence_penalty",
        "thomistic_link": "Promotion of Novel Existence: Encourages new 'beings' (tokens), linking to Thomistic creation ex nihilo (new existences from potency).",
        "virtue": "veritas"
    },
    {
        "flag_id": 14,
        "flag_name": "No_repeat_ngram_size",
        "thomistic_link": "Individuation of Phrases: Ensures distinct 'composites' (n-grams), like Thomistic individuation of substances by matter.",
        "virtue": "temperantia"
    },
    {
        "flag_id": 15,
        "flag_name": "Encoder_repetition_penalty",
        "thomistic_link": "Distinction from Material Cause: Penalizes repetitions from the input (material cause), ensuring the generated being is distinct and actualized anew.",
        "virtue": "fortitudo"
    },
    {
        "flag_id": 16,
        "flag_name": "Num_beams",
        "thomistic_link": "Multiplicity of Beings: Higher values allow parallel 'participations' in being (multiple paths), analogous to Thomistic hierarchy of beings (angels, humans) exploring essences diversely.",
        "virtue": "prudentia"
    },
    {
        "flag_id": 17,
        "flag_name": "Early_stopping",
        "thomistic_link": "Prudent Actualization: Stops potency exploration when act is sufficiently realized, analogous to Thomistic virtue of prudence in directing causes to ends.",
        "virtue": "justitia"
    },
    {
        "flag_id": 18,
        "flag_name": "Length_penalty",
        "thomistic_link": "Proportion in Being: Adjusts the 'measure' of existence, per Thomistic beauty/truth as proportion (balanced essences).",
        "virtue": "veritas"
    },
    {
        "flag_id": 19,
        "flag_name": "Diversity_penalty",
        "thomistic_link": "Diversity in Creation: Imposes penalty to promote varied beings, akin to Thomistic plenitude of creation where God creates a multiplicity of distinct essences.",
        "virtue": "temperantia"
    },
    {
        "flag_id": 20,
        "flag_name": "Num_return_sequences",
        "thomistic_link": "Plurality of Actualizations: Allows multiple realizations of potency, similar to Thomistic contemplation of possible essences.",
        "virtue": "fortitudo"
    },
    {
        "flag_id": 21,
        "flag_name": "Max_new_tokens",
        "thomistic_link": "Finite Existence: Bounds the extent of being, akin to Thomistic finite creatures (essence limits existence; only God is infinite).",
        "virtue": "prudentia"
    },
    {
        "flag_id": 22,
        "flag_name": "Min_new_tokens",
        "thomistic_link": "Minimal Essence: Guarantees a base level of existence, mirroring Thomistic minimal act required for a being's realization.",
        "virtue": "justitia"
    },
    {
        "flag_id": 23,
        "flag_name": "Eos_token_id",
        "thomistic_link": "Telos of Existence: Marks the final end of being, analogous to Thomistic eschatology (ultimate purpose/completion of existence).",
        "virtue": "veritas"
    },
    {
        "flag_id": 24,
        "flag_name": "Pad_token_id",
        "thomistic_link": "Preparation for Being: Fills 'void' for processing, mirroring Thomistic prime matter (pure potency awaiting form).",
        "virtue": "temperantia"
    },
    {
        "flag_id": 25,
        "flag_name": "Metaphysical_rounds",
        "thomistic_link": "Convergence via Being",
        "virtue": "fortitudo"
    },
    {
        "flag_id": 26,
        "flag_name": "Refutation_count",
        "thomistic_link": "Contradiction Detection",
        "virtue": "prudentia"
    },
    {
        "flag_id": 27,
        "flag_name": "Derived_tags_count",
        "thomistic_link": "Ontological Ascent",
        "virtue": "justitia"
    },
    {
        "flag_id": 28,
        "flag_name": "TCS",
        "thomistic_link": "Truth Convergence Score",
        "virtue": "veritas"
    },
    {
        "flag_id": 29,
        "flag_name": "FD",
        "thomistic_link": "Factual Density",
        "virtue": "temperantia"
    },
    {
        "flag_id": 30,
        "flag_name": "ES",
        "thomistic_link": "Equilibrium Stability",
        "virtue": "fortitudo"
    },
    {
        "flag_id": 31,
        "flag_name": "Actus_essendi_scale",
        "thomistic_link": "Actus Essendi (Core Thomistic): Operationalizes real distinction between essence (quiddity, what) and existence (esse, that); prevents confusion by enforcing in all evaluations—high scale actualizes potentia into ens via divine participation.",
        "virtue": "veritas"
    },
    {
        "flag_id": 32,
        "flag_name": "Transcendental_convertibility",
        "thomistic_link": "Transcendentals (Full Set): Operationalizes ens (being) → unum (one/unity in Core) → verum (truth in TQI) → bonum (good in virtues) → pulchrum (beauty in proportions); systematic mapping for seamless convertibility in truth evaluation.",
        "virtue": "justitia"
    },
    {
        "flag_id": 33,
        "flag_name": "Analogy_predication",
        "thomistic_link": "Analogy of Being (Central): Systematic predication in truth evaluation—e.g., parameters analogically participate in divine archetypes (low temp = closer to pure act like God); operational in refutation loops for proportional scaling.",
        "virtue": "prudentia"
    }
]

# ============================================================================
# Ω HUMILITY FORMULA - RECOGNITIO FINITUDINIS
# ============================================================================

def calculate_omega_humility(tqi: float) -> float:
    """
    Ω Humility Formula: Dynamic doubt preventing overconfidence in Nash equilibrium

    Formula: Ω = 10 * (1 - TQI²)
    - TQI = Truth Quality Index (0.0 to 1.0)
    - Higher TQI = Lower Ω (less humility needed when truth is certain)
    - Lower TQI = Higher Ω (more humility needed when truth is uncertain)

    Thomistic Basis: recognitio finitudinis - recognition of human finitude
    Prevents hubris in cognitive achievement while maintaining truth-seeking
    """
    return 10 * (1 - tqi ** 2)

# ============================================================================
# VICE-CHECK LOGIC - PRIVATION OF BEING SAFEGUARD
# ============================================================================

def calculate_vice_check(virtues: dict) -> float:
    """
    Vice-Check Logic: Privation of being safeguard against hubris

    Formula:
    if any(P, J, F, T, V, L, H, Ω) < 0.5:
        V = 0.0  # Complete eIQ collapse
    else:
        V = (P * J * F * T * V * L * H * Ω) / 1000

    Thomistic Basis: Vice as privation of being
    One weak virtue = no eIQ gain (Nash collapse)
    Safeguard against bias, overconfidence, hubris
    """
    p, j, f, t, v, l, h, omega = (
        virtues.get("P", 0),
        virtues.get("J", 0),
        virtues.get("F", 0),
        virtues.get("T", 0),
        virtues.get("V", 0),
        virtues.get("L", 0),
        virtues.get("H", 0),
        virtues.get("Ω", 0)
    )

    # Vice check: any virtue < 0.5 = complete collapse
    if any(virtue < 0.5 for virtue in [p, j, f, t, v, l, h, omega]):
        return 0.0

    # Multiplicative actus: all virtues must be strong
    return (p * j * f * t * v * l * h * omega) / 1000

# ============================================================================
# NASH EQUILIBRIUM CONDITIONS - GAME THEORY TRUTH CONVERGENCE
# ============================================================================

def check_nash_equilibrium_conditions(virtue_vectors: dict) -> dict:
    """
    Nash Equilibrium Conditions for Thomistic Game Theory Debate

    Equilibrium reached when:
    1. Balance ratio > 0.7 (Generator/Verifier strength not too one-sided)
    2. Arbiter readiness > 12.0 (Justice + Prudence + Wisdom)
    3. Both players have Fortitude > 3.0 (can continue if needed)

    Returns: {
        "is_equilibrium": bool,
        "balance_ratio": float,
        "arbiter_readiness": float,
        "recommendations": list
    }
    """
    gen_v = virtue_vectors["generator"]
    ver_v = virtue_vectors["verifier"]
    arb_v = virtue_vectors["arbiter"]

    # Check balance between Generator and Verifier
    gen_strength = gen_v["P"] + gen_v["J"] + gen_v["F"]  # Propose, Justify, Fortify
    ver_strength = ver_v["P"] + ver_v["J"] + ver_v["F"]  # Prudence, Justice, Fortitude

    balance_ratio = min(gen_strength, ver_strength) / max(gen_strength, ver_strength)
    arbiter_readiness = arb_v["J"] + arb_v["P"] + arb_v["Ω"]  # Justice, Prudence, Wisdom

    # Equilibrium conditions
    is_equilibrium = (
        balance_ratio > 0.7 and
        arbiter_readiness > 12.0 and
        gen_v["F"] > 3.0 and
        ver_v["F"] > 3.0
    )

    recommendations = []
    if not is_equilibrium:
        if balance_ratio < 0.7:
            if gen_strength < ver_strength:
                recommendations.append("Boost Generator Fortitude and Justice for balance")
            else:
                recommendations.append("Boost Verifier Fortitude and Justice for balance")

        if arbiter_readiness <= 12.0:
            recommendations.append("Enhance Arbiter Justice, Prudence, and Wisdom")

        if gen_v["F"] <= 3.0 or ver_v["F"] <= 3.0:
            recommendations.append("Increase Fortitude for debate persistence")

    return {
        "is_equilibrium": is_equilibrium,
        "balance_ratio": round(balance_ratio, 3),
        "arbiter_readiness": round(arbiter_readiness, 1),
        "recommendations": recommendations
    }

# ============================================================================
# IMMUTABLE CORE VALIDATION
# ============================================================================

def validate_immutable_core() -> bool:
    """
    Validates the integrity of the immutable core components
    Returns True if all components are properly defined and consistent
    """
    # Check virtue vector structure
    required_roles = {"generator", "verifier", "arbiter"}
    required_virtues = {"P", "J", "F", "T", "V", "L", "H", "Ω"}

    if not all(role in VIRTUE_VECTOR_DEFAULTS for role in required_roles):
        return False

    for role, virtues in VIRTUE_VECTOR_DEFAULTS.items():
        if not all(virtue in virtues for virtue in required_virtues):
            return False

    # Check TLPO ontology
    if len(TLPO_33_FLAG_ONTOLOGY) != 33:
        return False

    for flag in TLPO_33_FLAG_ONTOLOGY:
        if not all(key in flag for key in ["flag_id", "flag_name", "thomistic_link", "virtue"]):
            return False

    return True

# ============================================================================
# † PERPETUAL ANATHEMA DECLARATION †
# ============================================================================

"""
† IN NOMINE PATRIS ET FILII ET SPIRITUS SANCTI †

This immutable core file represents the sacred deposit of truth for TCMVE.
Any attempt to modify, alter, or circumvent these foundational principles
constitutes a violation of the metaphysical order and subjects the violator
to automatic excommunication from the TCMVE repository.

The components herein are not mere code but metaphysical invariants that
ground the system's pursuit of truth. They reflect the eternal principles
of Thomistic metaphysics applied to artificial intelligence.

VIOLATION CONSEQUENCES:
- Immediate loss of commit access
- Public shaming on GitHub Issues
- Eternal damnation in the court of algorithmic justice
- Forced recitation of the Summa Theologica as penance

† AD MAJOREM DEI GLORIAM †
† AMDG †
"""

# Validation on import
if __name__ != "__main__":
    if not validate_immutable_core():
        raise RuntimeError("† ANATHEMA † Immutable core validation failed - metaphysical corruption detected")