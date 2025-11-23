ontology = {
  "ontology_version": "1.3",
  "ontology_name": "Thomistic LLM Parameter Ontology (TLPO) — TCMVE-Adapted v1.3",
  "description": "A unified, hierarchical ontology for LLM response generation parameters, analogically mapped to Thomistic metaphysics (e.g., essence/existence via actus essendi, act/potency, hylomorphism with prime matter/substantial form, substance/accidents, four causes, and convertible transcendentals: ens-unum-verum-bonum-pulchrum). Designed for guiding AI outputs toward balanced, purposeful reasoning as participations in being, treating parameters as principles that shape the &#x27;being&#x27; of responses via analogy of being. **Fully Thomistic for TCMVE**: integrates truth-convergent verification, cross-LLM orchestration, metaphysical invariants, and operational distinctions (e.g., substantial vs. accidental change). Parameters as &#x27;flags&#x27; for agent configuration, auditing, and optimization, centering actus essendi to avoid essence-existence confusion.",
  "domain_module": "AI_generation_verification",
  "virtue_aggregation_method": "weighted_mean_by_thomistic_virtue",
  "categories": {
    "Core": {
      "description": "Foundational parameters controlling essence and existence (actus essendi), determinism vs. randomness, and substantial form. In TCMVE: used by Generator and Verifier for truth-seeking stability via analogical predication.",
      "flags": [1, 2, 3, 4, 5, 31]
    },
    "Sampling": {
      "description": "Parameters managing potency (prime matter) and diversity in token selection, inhering as accidents. In TCMVE: restricted during verification to minimize hallucination, distinguishing accidental from substantial change.",
      "flags": [6, 7, 8, 9, 10]
    },
    "Penalty": {
      "description": "Parameters moderating repetition and novelty (accidental modifications to substance). In TCMVE: tuned to enforce non-repetition and clarity, convertible via transcendentals (e.g., unity/one).",
      "flags": [11, 12, 13, 14, 15]
    },
    "Search": {
      "description": "Parameters for exploration and optimization (efficient and final causes, analogical to divine providence). In TCMVE: used sparingly in Generator; disabled in Verifier.",
      "flags": [16, 17, 18, 19, 20]
    },
    "Boundary": {
      "description": "Parameters defining limits and termination (formal and material causes, hylomorphic boundaries). In TCMVE: strictly enforced for convergence control, reflecting finite esse.",
      "flags": [21, 22, 23, 24]
    },
    "Metaphysical": {
      "description": "New category for transcendentals and esse/essentia operationalization. In TCMVE: central to auditing and truth evaluation via analogy of being.",
      "flags": [25, 26, 27, 28, 29, 30, 31, 32, 33]
    }
  },
  "dependency_groups": {
    "CoreGroup": [1, 2, 3, 4, 5, 31],
    "SamplingSet": [6, 7, 8, 9, 10],
    "PenaltySet": [11, 12, 13, 14, 15],
    "SearchSet": [16, 17, 18, 19, 20],
    "BoundarySet": [21, 22, 23, 24],
    "MetaphysicalSet": [25, 26, 27, 28, 29, 30, 31, 32, 33]
  },
  "tqi_formula": "0.3*veritas (truth) + 0.25*prudentia (prudence, unum/one) + 0.2*justitia (justice, bonum/good) + 0.15*temperantia (temperance, pulchrum/beauty) + 0.1*fortitudo (fortitude, ens/being) — transcendentals convertible: ens → unum → verum → bonum → pulchrum",
  "tqi_item_formula": "sum(Core..Metaphysical weights, analogically scaled by actus essendi)",
  "tqi_audit_formula": "sum(parameter_audit_weights) * esse_factor (distinguishing essence from existence)",
  "final_tqi_formula": "0.8*TQI_item + 0.2*TQI_audit + 0.0*analogy_penalty (centralized for proportional participation)",
  "total_weight": 1.0,
  "compliance_threshold": 0.95,
  "evidence_schema": {
    "source_type": ["api_default", "user_override", "model_config", "thomistic_analogy", "tcmve_ontology", "actus_essendi_verification"],
    "identifier": "string"
  },
  "tcmve_integration": {
    "generator_settings": {
      "temperature": 0.0,
      "do_sample": False,
      "top_p": 1.0,
      "repetition_penalty": 1.1,
      "max_new_tokens": 512,
      "actus_essendi_scale": 1.0
    },
    "verifier_settings": {
      "temperature": 0.0,
      "do_sample": False,
      "top_p": 1.0,
      "repetition_penalty": 1.2,
      "max_new_tokens": 256,
      "analogy_predication": True
    },
    "arbiter_settings": {
      "temperature": 0.0,
      "do_sample": False,
      "top_p": 1.0,
      "repetition_penalty": 1.1,
      "max_new_tokens": 4096,
      "transcendental_convertibility": "full"
    },
    "audit_trigger": "TQI &lt; 0.95 OR untruth_score &gt; 0.03 OR esse_essentia_mismatch &gt; 0.05"
  },
  "flags": [
    {
      "flag_id": 1,
      "flag_name": "Temperature",
      "thomistic_link": "Potency vs. Act (Prime Matter): High temperature embodies potency as prime matter (unformed potential for varied beings/responses), low as act (substantial form realizing determinate path via hylomorphism)."
    },
    {
      "flag_id": 2,
      "flag_name": "Logit_bias",
      "thomistic_link": "Final Cause: Directs toward telos, analogically participating in divine goodness (bonum), guiding accidental changes without substantial alteration."
    },
    {
      "flag_id": 3,
      "flag_name": "Do_sample",
      "thomistic_link": "Act-Potency Toggle: False = pure act (substantial form); True = engages potency (prime matter), operationalizing hylomorphic composition."
    },
    {
      "flag_id": 4,
      "flag_name": "Renormalize_logits",
      "thomistic_link": "Stability of Probabilities (Unity/One): Ensures consistent measures, convertible to truth (verum) as transcendental unity in being."
    },
    {
      "flag_id": 5,
      "flag_name": "Guidance_scale",
      "thomistic_link": "Directed Potency: Scales adherence to essence, analogically mirroring divine guidance (analogy of being) toward good (bonum)."
    },
    {
      "flag_id": 6,
      "flag_name": "Top_p",
      "thomistic_link": "Limitation of Potency (Hylomorphism): Restricts prime matter&#x27;s infinite potential via substantial form, distinguishing accidental diversity from substantial essence."
    },
    {
      "flag_id": 7,
      "flag_name": "Top_k",
      "thomistic_link": "Accidents Inhering in Substance: Fixed set of top tokens as accidents; operational distinction: accidental change (sampling) vs. substantial (core prompt essence)."
    },
    {
      "flag_id": 8,
      "flag_name": "Typical_p",
      "thomistic_link": "Typical Participation (Analogy of Being): Selects common analogical participations in higher essences, convertible transcendentals (e.g., one to true)."
    },
    {
      "flag_id": 9,
      "flag_name": "Epsilon_cutoff",
      "thomistic_link": "Threshold of Potency (Esse/Essentia): Cuts negligible potentials, ensuring essence (quiddity) actualizes into existence without non-being."
    },
    {
      "flag_id": 10,
      "flag_name": "Min_p",
      "thomistic_link": "Relative Limitation (Transcendentals): Proportional to highest act, mirroring convertible beauty (pulchrum) as proportion in being."
    },
    {
      "flag_id": 11,
      "flag_name": "Repetition_penalty",
      "thomistic_link": "Unity of Being (One/Unum): Prevents multiplicity without purpose, central to transcendental convertibility."
    },
    {
      "flag_id": 12,
      "flag_name": "Frequency_penalty",
      "thomistic_link": "Moderation of Multiplicity (Good/Bonum): Tempers excess, analogically participating in divine justice."
    },
    {
      "flag_id": 13,
      "flag_name": "Presence_penalty",
      "thomistic_link": "Promotion of Novel Existence (Actus Essendi): Encourages new actualizations, distinguishing esse from essentia."
    },
    {
      "flag_id": 14,
      "flag_name": "No_repeat_ngram_size",
      "thomistic_link": "Individuation (Substance/Accidents): Ensures distinct composites, accidental changes only."
    },
    {
      "flag_id": 15,
      "flag_name": "Encoder_repetition_penalty",
      "thomistic_link": "Distinction from Material Cause: Penalizes input echoes, hylomorphic actualization anew."
    },
    {
      "flag_id": 16,
      "flag_name": "Num_beams",
      "thomistic_link": "Multiplicity of Beings (Analogy): Parallel paths as analogical hierarchy, convertible to truth."
    },
    {
      "flag_id": 17,
      "flag_name": "Early_stopping",
      "thomistic_link": "Prudent Actualization: Converges potency to act, per final cause."
    },
    {
      "flag_id": 18,
      "flag_name": "Length_penalty",
      "thomistic_link": "Proportion in Being (Beauty/Pulchrum): Balanced measure, transcendental harmony."
    },
    {
      "flag_id": 19,
      "flag_name": "Diversity_penalty",
      "thomistic_link": "Diversity in Creation (Good/Bonum): Promotes plenitude, analogical participation."
    },
    {
      "flag_id": 20,
      "flag_name": "Num_return_sequences",
      "thomistic_link": "Plurality of Actualizations: Multiple esse realizations, but converges to one truth."
    },
    {
      "flag_id": 21,
      "flag_name": "Max_new_tokens",
      "thomistic_link": "Finite Existence (Esse/Essentia): Bounds esse, finite like created being."
    },
    {
      "flag_id": 22,
      "flag_name": "Min_new_tokens",
      "thomistic_link": "Minimal Essence: Base act for realization."
    },
    {
      "flag_id": 23,
      "flag_name": "Eos_token_id",
      "thomistic_link": "Telos of Existence: Completion per final cause."
    },
    {
      "flag_id": 24,
      "flag_name": "Pad_token_id",
      "thomistic_link": "Prime Matter Preparation: Awaits form in hylomorphism."
    },
    {
      "flag_id": 25,
      "flag_name": "Metaphysical_rounds",
      "thomistic_link": "Convergence via Being (Transcendentals): Rounds as analogical ascent to unity."
    },
    {
      "flag_id": 26,
      "flag_name": "Refutation_count",
      "thomistic_link": "Contradiction Detection (Truth/Verum): Refutes via accidental errors, preserves substance."
    },
    {
      "flag_id": 27,
      "flag_name": "Derived_tags_count",
      "thomistic_link": "Ontological Ascent (Analogy of Being): New truths as participations."
    },
    {
      "flag_id": 28,
      "flag_name": "TCS",
      "thomistic_link": "Truth Convergence Score (Esse/Essentia): Measures actualization distance."
    },
    {
      "flag_id": 29,
      "flag_name": "FD",
      "thomistic_link": "Factual Density (Substance): Verifiable claims as substantial forms."
    },
    {
      "flag_id": 30,
      "flag_name": "ES",
      "thomistic_link": "Equilibrium Stability (One/Unum): Nash as transcendental unity."
    },
    {
      "flag_id": 31,
      "flag_name": "Actus_essendi_scale",
      "category": "Metaphysical",
      "description": "Scales the actualization of essence into existence; 1.0 = full distinction.",
      "thomistic_link": "Actus Essendi (Core Thomistic): Operationalizes real distinction between essence (quiddity, what) and existence (esse, that); prevents confusion by enforcing in all evaluations—high scale actualizes potentia into ens via divine participation.",
      "virtue": "veritas",
      "effect": "Centers ontology: every flag&#x27;s output must distinguish essentia (parameter definition) from esse (realized truth), analogically scaled.",
      "metric_targets": { "min": 0.0, "max": 1.0 },
      "metric_type": "numeric",
      "scoring_method": "range",
      "dependencies": [1, 3, 28],
      "severity_level": "critical",
      "severity_multiplier": 2.0,
      "human_review_required": True,
      "weight": 0.08,
      "tcmve_recommendation": "Always 1.0 in Arbiter; audit for mismatches."
    },
    {
      "flag_id": 32,
      "flag_name": "Transcendental_convertibility",
      "category": "Metaphysical",
      "description": "Ensures parameters convert between transcendentals (e.g., unity to truth).",
      "thomistic_link": "Transcendentals (Full Set): Operationalizes ens (being) → unum (one/unity in Core) → verum (truth in TQI) → bonum (good in virtues) → pulchrum (beauty in proportions); systematic mapping for seamless convertibility in truth evaluation.",
      "virtue": "justitia",
      "effect": "Maps across flags: e.g., repetition_penalty (unum) converts to length_penalty (pulchrum) via analogy.",
      "metric_targets": { "value": "full" },
      "metric_type": "string",
      "scoring_method": "exact_match",
      "dependencies": [4, 11, 18],
      "severity_level": "critical",
      "severity_multiplier": 1.5,
      "human_review_required": False,
      "weight": 0.06,
      "tcmve_recommendation": "Enabled in all settings; triggers audit if &lt;0.95 convertibility."
    },
    {
      "flag_id": 33,
      "flag_name": "Analogy_predication",
      "category": "Metaphysical",
      "description": "Applies analogical predication in evaluations (proportional participation).",
      "thomistic_link": "Analogy of Being (Central): Systematic predication in truth evaluation—e.g., parameters analogically participate in divine archetypes (low temp = closer to pure act like God); operational in refutation loops for proportional scaling.",
      "virtue": "prudentia",
      "effect": "Central to architecture: evaluates outputs via degrees of participation (e.g., high analogy score = closer to archetypal truth).",
      "metric_targets": { "min": 0.0, "max": 1.0 },
      "metric_type": "numeric",
      "scoring_method": "range",
      "dependencies": [25, 26, 27],
      "severity_level": "critical",
      "severity_multiplier": 1.5,
      "human_review_required": False,
      "weight": 0.05,
      "tcmve_recommendation": "Enabled in Arbiter; operational in refutation loops for proportional scaling."
    }
  ]
}
    