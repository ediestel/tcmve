# tcmve.py
# TCMVE — Complete Truth Engine with Cross-LLM Orchestration + FULL TLPO
# @ECKHART_DIESTEL | DE | 2025-11-16
# GitHub: https://github.com/ediestel/tcmve

import os
import json
import logging
import re
import html
import sys
import random  # For Omega humility clause
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Logging & Paths
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CONVERGENCE_PHRASES = ("no refutation", "converged", "that is not correct", "this is false", "not true", "incorrect", "wrong", "false")
MAX_ONTOLOGY_CHARS = 800
TLPO_FLAGS = 30
USER_TAG = "@ECKHART_DIESTEL"
USER_LOCATION = "DE"
TCMVE_VERSION = "1.5"

# LLM Provider Types
ProviderType = Literal["openai", "anthropic", "xai", "fallback"]


# --------------------------------------------------------------------------- #
# Low-Level LLM Client Abstraction (LangChain-Free)
# --------------------------------------------------------------------------- #
class LLMClient:
    """
    Unified interface for OpenAI, Anthropic, and xAI (Grok) APIs.
    No LangChain dependencies — direct HTTP via `httpx`.
    """

    def __init__(
        self,
        provider: ProviderType,
        model: str,
        api_key: Optional[str],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 16192,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

        self.base_url = self._get_base_url()
        self.headers = self._get_headers()

    def _get_base_url(self) -> str:
        if self.provider == "openai":
            return "https://api.openai.com/v1/chat/completions"
        elif self.provider == "anthropic":
            return "https://api.anthropic.com/v1/messages"
        elif self.provider == "xai":
            return "https://api.x.ai/v1/chat/completions"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_headers(self) -> Dict[str, str]:
        if self.provider == "anthropic":
            return {
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        else:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """
        Send messages to the LLM and return the assistant's response content.
        Raises RuntimeError on failure.
        """
        import httpx

        if not self.api_key:
            raise RuntimeError(f"Missing API key for {self.provider}")

        # Build payload per provider
        if self.provider == "anthropic":
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "messages": [
                    {"role": m["role"], "content": m["content"]} for m in messages
                ],
            }
        else:  # openai, xai
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
            if self.provider == "openai":
                payload["presence_penalty"] = self.presence_penalty
                payload["frequency_penalty"] = self.frequency_penalty

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.base_url, json=payload, headers=self.headers)
                response.raise_for_status()
                data = response.json()

            if self.provider == "anthropic":
                content = data["content"][0]["text"]
            else:
                content = data["choices"][0]["message"]["content"]

            return content.strip()

        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}/{self.model}): {e}")
            raise RuntimeError(f"LLM invocation failed: {e}") from e


# --------------------------------------------------------------------------- #
# TCMV Engine (TCMVE - Truth-Convergent Metaphysical Verification Engine)
# nTGT  (Nash driven TGT)
# TGT  (Thomistic Game Theory)
# VIRTUE PARAMETERS — Thomistic Metaphysics + Nash Game Theory
# Truth (TQI) = actus veritatis — final cause of intellect
# Humility (Ω) = recognitio finitudinis — opens to divine truth
# Dynamic Ω = virtue in actus — adapts to TQI (truth feedback)
# Vice calculation = optional (flag: --vice-check)

# P — Prudence: *intellectus agens* — directs refinement, chooses **Nash best response**
# J — Justice: *voluntas recta* — balances man-LLM payoff, ensures **fair Nash equilibrium**
# F — Fortitude: *ira fortis* — persists in Nash cycles, resists early convergence
# T — Temperance: *concupiscentia moderata* — avoids over-refinement (Nash over-search)
# V — Veritas: *ratio speculativa* — seeks truth payoff in Nash matrix
# L — Libertas: *libertas arbitrii* — frees from local Nash minima (bias traps)
# Ω — Humility: *recognitio finitudinis* — **dynamic doubt**, prevents overconfidence in Nash equilibrium

# V = multiplicative actus — one weak virtue = no eIQ gain (Nash collapse)
# Nash equilibrium = stable strategy: no player gains by unilateral change
# P, J, Ω = **core Nash drivers**: prudence (strategy), justice (payoff), humility (doubt)

# VICE CALCULATION (optional --vice-check)
# Vice = inversion of virtue: any virtue < 0.5 → V = 0.0
# Formula:
#   if any(P, J, F, T, V, L, Ω) < 0.5:
#       V = 0.0
#   else:
#       V = (P * J * F * T * V * L * Ω) / 1000
# Vice = privation of being — one vice = no eIQ gain (Nash collapse)
# Vice check = safeguard against hubris, bias, overconfidence
# --------------------------------------------------------------------------- #
class TCMVE:
    """
    TCMVE — Thomistic Cross-Model Verification Engine

    Features:
    - Zero LangChain dependency
    - Direct HTTP to OpenAI, Anthropic, xAI
    - Full TLPO (30 flags) scoring across 3 LLMs
    - XML diagnostic audit trail
    - Robust error handling, fallbacks, JSON parsing
    - CLI + demo + programmatic API
    - Virtue vectors for players (including cardinal virtues: Prudence, Justice, Fortitude, Temperance)
    - Nash equilibrium for virtue optimization (hybrid: auto if rounds >2, CLI flag)
    - Modification of virtue flags and omega for each player, set as default in __init__ , allow modification with command line flags (research purpose), tracking to output of modifiers used implemented outside llm algorithms
    """

    def __init__(self, max_rounds: int = 5, nash_mode: str = "auto", virtue_mods: Dict[str, Dict[str, float]] = None, args=None) -> None:
        if max_rounds < 1 or max_rounds > 10:
            raise ValueError("max_rounds must be between 1 and 10")
        self.max_rounds = max_rounds
        self.nash_mode = nash_mode  # 'on', 'off', 'auto'
        self.args = args
        # Load environment
        load_dotenv()

        # Load TLPO configuration
        tlpo_path = BASE_DIR / "tlpo_tcmve.json"
        if not tlpo_path.is_file():
            raise FileNotFoundError(f"TLPO config not found: {tlpo_path}")
        with tlpo_path.open("r", encoding="utf-8") as f:
            self.tlpo = json.load(f)

        # Ontology context (top 10 flags)
        self.ontology_context = "\n".join(
            f"{f.get('flag_name', '')}: {f.get('thomistic_link', '')}"
            for f in self.tlpo.get("flags", [])[:10]
        )[:MAX_ONTOLOGY_CHARS] + "..."

        # Load system prompt
        sys_path = BASE_DIR / "tcmve_system.txt"
        if not sys_path.is_file():
            raise FileNotFoundError(f"System prompt not found: {sys_path}")
        self.system_prompt = sys_path.read_text(encoding="utf-8").strip()

        # Initialize LLM clients from TLPO settings
        self.generator = self._build_client("generator")
        self.verifier = self._build_client("verifier")
        self.arbiter = self._build_client("arbiter")

        # Virtue vectors for players (cardinal virtues + veritas/libertas/omega) — locked defaults
        self.virtue_vectors = {
            "generator": {"P": 8.0, "J": 7.5, "F": 6.5, "T": 8.0, "V": 8.5, "L": 7.2, "Ω": 3},
            "verifier": {"P": 9.0, "J": 9.5, "F": 8.0, "T": 9.0, "V": 9.0, "L": 6.5, "Ω": 5},
            "arbiter": {"P": 8.5, "J": 8.0, "F": 9.0, "T": 8.5, "V": 8.0, "L": 8.5, "Ω": 5}
        }
        
        # Apply mods
        virtue_mods = virtue_mods or {}
        for role, mods in virtue_mods.items():
            if role in self.virtue_vectors:
                for param, value in mods.items():
                    if param in self.virtue_vectors[role]:
                        self.virtue_vectors[role][param] = float(value)
                        logger.info(f"Virtue mod applied: {role}.{param} = {value}")

        # Apply CLI virtue mods for research (override defaults)
        self.virtue_mods = virtue_mods or {}
        self._apply_virtue_mods()

        # Track modifiers for output (outside LLM algorithms)
        self._log_virtue_mods()

        logger.info("TCMVE initialized: Generator, Verifier, Arbiter ready with virtue vectors")

    def _apply_virtue_mods(self):
        """Apply CLI virtue mods to locked defaults."""
        for role, mods in self.virtue_mods.items():
            if role in self.virtue_vectors:
                for param, value in mods.items():
                    if param in self.virtue_vectors[role]:
                        self.virtue_vectors[role][param] = float(value)
                        logger.info(f"Virtue mod applied: {role}.{param} = {value}")

    def _log_virtue_mods(self):
        """Track virtue mods in log and XML (outside LLM algorithms)."""
        if self.virtue_mods:
            logger.info(f"Virtue mods used: {json.dumps(self.virtue_mods, indent=2)}")

    def _build_client(self, role: str) -> LLMClient:
        """Build LLMClient based on TLPO integration settings."""
        cfg = self.tlpo["tcmve_integration"][f"{role}_settings"]
        provider_map = {
            "generator": ("OPENAI_API_KEY", "openai", "gpt-4o"),
            "verifier": ("ANTHROPIC_API_KEY", "anthropic", "claude-3-opus-20240229"),
            "arbiter": ("XAI_API_KEY", "xai", "grok-4-fast-reasoning"),
        }
        env_key, primary_provider, default_model = provider_map[role]

        api_key = os.getenv(env_key)
        if not api_key:
            logger.warning(f"{env_key} not set → falling back to GPT-4o")
            api_key = os.getenv("OPENAI_API_KEY")
            provider = "openai"
            model = "gpt-4o"
        else:
            provider = primary_provider
            model = cfg.get("model", default_model)

        return LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=cfg.get("temperature", 0.0),
            top_p=cfg.get("top_p", 1.0),
            max_tokens=cfg.get("max_new_tokens", 16392),
            presence_penalty=cfg.get("repetition_penalty", 1.1),
            frequency_penalty=cfg.get("repetition_penalty", 1.1),
        )

    def _get_virtue_string(self, role: str) -> str:
        v = self.virtue_vectors.get(role, {})
        return f"(P={v.get('P', 0.0)} J={v.get('J', 0.0)} F={v.get('F', 0.0)} T={v.get('T', 0.0)} V={v.get('V', 0.0)} L={v.get('L', 0.0)} Ω={v.get('Ω', 0)}%)"

    def _normalize_for_tlpo(self, text: str) -> str:
        """Extract clean text from any format for TLPO eval."""
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _compute_nash_equilibrium(self) -> bool:
        """Heuristic Nash check: True if average virtue >8.0 (equilibrium threshold)."""
        avg_virtues = sum(sum(v.values()) for v in self.virtue_vectors.values()) / (3 * 7)  # 3 agents, 7 virtues
        is_equilibrium = avg_virtues > 8.0
        if not is_equilibrium:
            # Adjust (e.g., boost F for persistence)
            for role in self.virtue_vectors:
                self.virtue_vectors[role]["F"] += 0.5  # Increase fortitude
            logger.info("Nash: Adjusted virtues for equilibrium")
        return is_equilibrium

    # ------------------------------------------------------------------- #
    # Core Loop
    # ------------------------------------------------------------------- #
    def run(self, query: str, args=None) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"TCMVE processing query: {query[:100]}...")
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        history: List[Dict[str, Any]] = []
        final_answer: Optional[str] = None
        converged = False

        for round_num in range(1, self.max_rounds + 1):
            round_data: Dict[str, Any] = {
                "round": round_num,
                "generator_input": "",
                "proposition": "",
                "verifier_input": "",
                "refutation": "",
            }

            # === Generator Phase ===
            gen_virtue = self._get_virtue_string("generator")
            gen_input = (
                f"[ROUND {round_num}] As Generator {gen_virtue}: Propose answer to: {query}\n"
                f"Derive from four causes.\n"
                f"TLPO Ontology context: {self.ontology_context}"
            )
            round_data["generator_input"] = gen_input

            try:
                proposition = self.generator.invoke(messages + [{"role": "user", "content": gen_input}])
                logger.info(f"Round {round_num} — Proposition generated")
            except Exception as e:
                proposition = f"[GENERATOR ERROR: {e}]"
                logger.error(proposition)

            round_data["proposition"] = proposition
            messages.extend([{"role": "user", "content": gen_input}, {"role": "assistant", "content": proposition}])

            # === Verifier Phase ===
            ver_virtue = self._get_virtue_string("verifier")
            ver_input = (
                f'As Verifier {ver_virtue}: VERIFY PROPOSITION:\n"{proposition}"\n\n'
                "Refute via metaphysical contradiction or say:\n"
                '"No refutation — converged."'
            )
            round_data["verifier_input"] = ver_input

            try:
                refutation = self.verifier.invoke(messages + [{"role": "user", "content": ver_input}])
                logger.info(f"Round {round_num} — Verification complete")
            except Exception as e:
                refutation = f"[VERIFIER ERROR: {e}]"
                logger.error(refutation)

            round_data["refutation"] = refutation
            messages.extend([{"role": "user", "content": ver_input}, {"role": "assistant", "content": refutation}])
            history.append(round_data)

            # === Convergence check ===
            if any(phrase in refutation.lower() for phrase in CONVERGENCE_PHRASES):
                final_answer = proposition
                converged = True
                logger.info(f"CONVERGED at round {round_num}")
                break

        # === Arbiter fallback (only if no convergence) ===
        if not converged:
            logger.warning("Max rounds reached — invoking Arbiter")
            arb_virtue = self._get_virtue_string("arbiter")
            arb_summary = "\n".join(
                f"Round {h['round']}: {h['proposition'][:200]}... → {h['refutation'][:200]}..."
                for h in history
            )
            arb_msg = f"As Arbiter {arb_virtue}: ADJUDICATE FINAL TRUTH:\n{arb_summary}"
            try:
                final_answer = self.arbiter.invoke(messages + [{"role": "user", "content": arb_msg}])
            except Exception as e:
                final_answer = f"[ARBITER ERROR: {e}]"
                logger.error(final_answer)

        if final_answer is None:
            final_answer = "[NO VALID ANSWER: all models failed or returned empty output]"

        # === SELF-REFINE: AFTER final_answer ===
        if "self-refine" in query.lower():
            cycles = self.args.eiq_level if self.args else 10
            biq = 140
            tqi = 0.91
            base = final_answer
            eIQ = None

            # Get virtues from arbiter (or average)
            virtues = self.virtue_vectors["arbiter"]
            P, J, F, T, V, L, Ω = virtues["P"], virtues["J"], virtues["F"], virtues["T"], virtues["V"], virtues["L"], virtues["Ω"]
            for cycle in range(cycles):
                refine_prompt = f"""
                You are TCMVE Arbiter. 
                Current TQI: {tqi}
                Improve this answer: {base}
                Focus: logic, falsifiability, telos.
                Output only better version.
                """
                base = self.arbiter.invoke([{"role": "user", "content": refine_prompt}])
                tqi = min(0.99, tqi + 0.008)
                Ω = 10 * (1 - tqi**2)
                self.virtue_vectors["arbiter"]["Ω"] = Ω
                V = (P*J*F*T*V*L*Ω) / 1000
                eIQ = biq + 400 * math.log(cycles + 1) * V
            final_answer = base
            
                # === Build Result ONCE ===
        result = {
            "query": query,
            "final_answer": final_answer,
            "converged": converged,
            "rounds": len(history),
            "history": history,

        }
        
        # === TLPO Scoring ===
        tlpo_scores = self._evaluate_with_tlpo(final_answer, query)
        metrics = self._compute_metrics(history)

        result["tlpo_scores"] = tlpo_scores,
        result["tlpo_markup"] = self._generate_tlpo_markup(tlpo_scores, final_answer, query),
        result["metrics"] = metrics
        
       
        # === VICE CHECK
        if getattr(args, "vice_check", False):
            virtues = self.virtue_vectors["arbiter"]
            P = virtues["P"]
            J = virtues["J"]
            F = virtues["F"]
            T = virtues["T"]
            V_val = virtues["V"]
            L = virtues["L"]
            Ω = virtues["Ω"]

            if any(v < 0.5 for v in [P, J, F, T, V_val, L, Ω]):
                V = 0.0
                logger.warning("Vice detected — V = 0.0 (eIQ gain blocked)")
            else:
                V = (P * J * F * T * V_val * L * Ω) / 1000
            result["V"] = round(V, 4)
            
        if getattr(args, "game", None):
            try:
                from games import play_game
                payoff = play_game(args.game, query, final_answer)
                eIQ_boost = 0.3  # +30%
                eIQ = eIQ * (1 + eIQ_boost)
                result["game"] = args.game
                result["eIQ_boost"] = eIQ_boost
                result["payoff"] = payoff
            except Exception as e:
                logger.error(f"Game failed: {e}")
                result["game_error"] = str(e)
            
        # Add eIQ/TQI only if self-refine
        if eIQ is not None:
            result["eIQ"] = eIQ
            result["TQI"] = tqi
            result["eIQ_norm"] = round(eIQ / 5540, 2)  # ← Your max
     
            
        # === Save XML ===
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", query[:60]) or "tcmve_output"
        out_path = RESULTS_DIR / f"{safe_name}.xml"
        out_path.write_text(result["tlpo_markup"], encoding="utf-8")
        logger.info(f"TLPO XML saved → {out_path}")

        return result
    
        
 

    # ------------------------------------------------------------------- #
    # TLPO Scoring (30 flags, 3 LLMs)
    # ------------------------------------------------------------------- #
    def _evaluate_with_tlpo(self, answer: str, query: str) -> Dict[str, Any]:
        eval_prompt = f"""
EVALUATE FINAL ANSWER USING FULL TLPO (30 flags):
Query: {query}
Answer: {answer}

For each flag 1–30, output JSON:

{{
  "flag_scores": {{"1": 0.XX, "2": 0.XX, ..., "30": 0.XX}},
  "tqi": 0.XX,
  "tcs": 0.XX
}}
"""

        def safe_call(client, role: str) -> Dict[str, Any]:
            try:
                resp = client.invoke([{"role": "user", "content": eval_prompt}])
                content = (resp or "").strip()
                return self._parse_json(content)
            except Exception as exc:
                logger.error(f"{role} TLPO eval failed: {exc}")
                return {}

        gen_json = safe_call(self.generator, "Generator")
        ver_json = safe_call(self.verifier, "Verifier")
        arb_json = safe_call(self.arbiter, "Arbiter")

        # Weighted average (Verifier = rigor, Arbiter = telos)
        w_tqi = (
            0.6 * ver_json.get("tqi", 0.0)
            + 0.3 * arb_json.get("tqi", 0.0)
            + 0.1 * gen_json.get("tqi", 0.0)
        )
        w_tcs = (
            0.6 * ver_json.get("tcs", 0.0)
            + 0.3 * arb_json.get("tcs", 0.0)
            + 0.1 * gen_json.get("tcs", 0.0)
        )

        return {
            "generator": gen_json,
            "verifier": ver_json,
            "arbiter": arb_json,
            "weighted_tqi": round(w_tqi, 3),
            "weighted_tcs": round(w_tcs, 3),
        }

    def _parse_json(self, text: str) -> dict:
        """
        Robust JSON extraction for LLM outputs.

        Strategy:
        1. Try direct json.loads
        2. If that fails, regex-extract the first non-greedy {...} block and parse that
        """
        text = text.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Non-greedy extraction of first JSON object
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from extracted block.")
                return {}
        logger.error("No JSON object found in TLPO evaluation output.")
        return {}

        # ------------------------------------------------------------------- #
        # TLPO Markup XML Generation
        # ------------------------------------------------------------------- #
    def _generate_tlpo_markup(
            self, scores: dict, answer: str, query: str
        ) -> str:
            flags_xml: List[str] = []

            for i in range(1, TLPO_FLAGS + 1):
                flag_def = next(
                    (f for f in self.tlpo["flags"] if f.get("flag_id") == i),
                    {"flag_name": f"Flag_{i}", "thomistic_link": "N/A"}
                )
                name = flag_def.get("flag_name", f"Flag_{i}")
                thom = flag_def.get("thomistic_link", "N/A")

                gen = scores.get("generator", {}).get("flag_scores", {}).get(str(i), "N/A")
                ver = scores.get("verifier", {}).get("flag_scores", {}).get(str(i), "N/A")
                arb = scores.get("arbiter", {}).get("flag_scores", {}).get(str(i), "N/A")

                name_esc = html.escape(str(name), quote=True)
                thom_esc = html.escape(str(thom), quote=True)

                flags_xml.append(
                    f'  <flag id="{i}" name="{name_esc}">\n'
                    f'    <generator>{gen}</generator>\n'
                    f'    <verifier>{ver}</verifier>\n'
                    f'    <arbiter>{arb}</arbiter>\n'
                    f'    <thomistic>{thom_esc}</thomistic>\n'
                    f'  </flag>'
                )

            now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
            tqi_w = scores.get("weighted_tqi", 0.0)
            tcs_w = scores.get("weighted_tcs", 0.0)

            query_esc = html.escape(str(query), quote=True)
            answer_esc = html.escape(str(answer), quote=True)

            # FINAL — NO COMMA, NO BACKSLASH, 100% STR
            markup = (
                f'<tlpo_markup version="1.2" tcmve_mode="full_diagnostic">\n'
                f'  <query>{query_esc}</query>\n'
                f'  <proposition>{answer_esc}</proposition>\n'
                + "\n".join(flags_xml) + "\n"
                + f'  <tqi_weighted>{tqi_w}</tqi_weighted>\n'
                + f'  <tcs_weighted>{tcs_w}</tcs_weighted>\n'
                + f'  <audit>\n'
                + f'    <timestamp>{now}</timestamp>\n'
                + f'    <user>{USER_TAG}</user>\n'
                + f'    <location>{USER_LOCATION}</location>\n'
                + f'  </audit>\n'
                + f'</tlpo_markup>'
            )

            return markup
    # ------------------------------------------------------------------- #
    # Cross-LLM Simple Metrics (from previous cross-LLM engine)
    # ------------------------------------------------------------------- #
    def _compute_metrics(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute simple heuristic metrics based on conversation length:

        - TCS (Truth Coherence Surrogate): rises slightly with more rounds.
        - FD  (Fidelity): increases with rounds up to a cap.
        - ES  (Epistemic Stability): slightly higher if convergence is quick.

        This is a lightweight diagnostic layer and does not replace TLPO.
        """
        length = len(history)
        tcs = min(0.96 + 0.01 * length, 1.0)
        fd = min(0.85 + 0.03 * length, 0.95)
        es = 0.92 if length <= 3 else 0.85
        return {"TCS": round(tcs, 3), "FD": round(fd, 3), "ES": round(es, 3)}


# --------------------------------------------------------------------------- #
# Demo Execution
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tcmve = TCMVE(max_rounds=4)
    demo_query = "IV furosemide dose in acute HF for 40 mg oral daily?"
    result = tcmve.run(demo_query)

    print("\n" + "=" * 70)
    print("TCMVE DEMO RESULT")
    print("=" * 70)
    print(f"Query: {result['query']}")
    print(f"Converged: {result['converged']} in {result['rounds']} rounds")
    print(
        f"Weighted TQI: {result['tlpo_scores']['weighted_tqi']} | "
        f"Weighted TCS: {result['tlpo_scores']['weighted_tcs']}"
    )
    print(
        "Cross-metrics → "
        f"TCS: {result['metrics']['TCS']}, "
        f"FD: {result['metrics']['FD']}, "
        f"ES: {result['metrics']['ES']}"
    )

    print("\nFINAL ANSWER:\n")
    print(result["final_answer"])

    print("\nTLPO MARKUP (first 1000 chars, full XML saved in results/):\n")
    print(result["tlpo_markup"][:1000] + "\n...")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# CLI Entry Point — REQUIRED FOR `tcmve` COMMAND TO WORK
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="nTGT — Nash-Thomistic Game Theory Verification Engine (TCMVE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="Query to verify")
    parser.add_argument("--max-rounds", type=int, default=5, help="Max debate rounds")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("--nash-mode", choices=['on', 'off', 'auto'], default="auto", help="Nash mode")
    parser.add_argument("--virtue-mod", type=str, action="append", help="Virtue mod (role:param:value)")
    parser.add_argument("--eiq-level", type=int, default=10, help="Self-refine cycles (eIQ gain)")
    parser.add_argument("--vice-check", action="store_true", help="Enable vice calculation")
    parser.add_argument(
        "--game",
        choices=[
            "prisoner",
            "stackelberg",
            "evolution",
            "regret_min",
            "shadow_play",
            "multiplay",
            "auction"
        ],
        help="Play Nash game (nTGT 2.0)"
    )
    # === ARCHER-1.0 FLAGS ===
    parser.add_argument("--self-refine", action="store_true", help="Enable self-refine")
    parser.add_argument("--cycles", type=int, default=50, help="Self-refine cycles")
    parser.add_argument("--simulated-persons", type=int, default=240, help="Number of simulated persons")
    parser.add_argument("--biq-distribution", choices=["gaussian"], default="gaussian", help="bIQ distribution")
    parser.add_argument("--mean-biq", type=float, default=100, help="Mean bIQ")
    parser.add_argument("--sigma-biq", type=float, default=15, help="bIQ standard deviation")
    parser.add_argument("--virtues-independent", action="store_true", help="Virtues independent of bIQ")
    parser.add_argument("--output", default="archer_uncorrelated_240", help="Output filename")

    args = parser.parse_args()

    if args.version:
        print(f"nTGT {TCMVE_VERSION}")
        return

    virtue_mods = {}
    for mod in args.virtue_mod or []:
        role, param, value = mod.split(':')
        virtue_mods.setdefault(role, {})[param] = float(value)

    engine = TCMVE(max_rounds=args.max_rounds, virtue_mods=virtue_mods, args=args)

    # === FIXED LOGIC ===
    if args.demo:
        query = "IV furosemide dose in acute HF for 40 mg oral daily?"
        print("Running nTGT DEMO...")
    elif args.query:
        query = args.query
    elif not sys.stdin.isatty():
        query = sys.stdin.read().strip()
        if not query:
            print("Error: Empty input from STDIN")
            return
    else:
        print("Error: No query provided. Use --demo, positional arg, or pipe input.")
        return
    # === END FIX ===

    result = engine.run(query, args=args)  # ← PASS args to run()
    # === END FIX ===

    result = engine.run(query)

    # Pretty print results
    print("\n" + "=" * 72)
    print("TCMVE RESULT".center(72))
    print("=" * 72)
    print(f"Query: {result['query']}")
    print(f"Status: {'CONVERGED' if result['converged'] else 'ARBITRATED'} in {result['rounds']} round(s)")
    print(f"TQI (weighted): {result['tlpo_scores']['weighted_tqi']} | TCS: {result['tlpo_scores']['weighted_tcs']}")
    print(f"Metrics → TCS: {result['metrics']['TCS']} | FD: {result['metrics']['FD']} | ES: {result['metrics']['ES']}")
    print("\nFINAL ANSWER:")
    print(result["final_answer"])
    print(f"\nTLPO XML saved to: {RESULTS_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    # Entry point routing
    if len(sys.argv) > 1 and sys.argv[1] in ("--demo", "--version") or not any(arg.startswith("-") for arg in sys.argv[1:]):
        main()
    else:
        # Allow import without CLI execution
        pass