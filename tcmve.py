# tcmve.py
# TCMVE — Complete Truth Engine with Cross-LLM Orchestration + FULL TLPO
# @ECKHART_DIESTEL | DE | 2025-11-15
# Run: python tcmve.py
# GitHub: https://github.com/ECKHART_DIESTEL/tcmve

import os
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import html
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# --------------------------------------------------------------------------- #
# Logging & Paths
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CONVERGENCE_PHRASES = ("no refutation", "converged")
MAX_ONTOLOGY_CHARS = 800
TLPO_FLAGS = 30
USER_TAG = "@ECKHART_DIESTEL"
USER_LOCATION = "DE"


# --------------------------------------------------------------------------- #
# TCMVE Engine
# --------------------------------------------------------------------------- #
class TCMVE:
    """
    TCMVE — Thomistic Cross-Model Verification Engine

    Combines:
    - Cross-LLM orchestration (Generator / Verifier / Arbiter)
    - Thomistic Logical Purity Ontology (TLPO, 30 flags)
    - Full TLPO scoring (TQI/TCS, weighted across 3 LLMs)
    - XML diagnostic markup output for audits
    """

    def __init__(self, max_rounds: int = 5) -> None:
        self.max_rounds = max_rounds

        # --- .env (for API keys etc.) ----------------------------------- #
        load_dotenv()

        # --- Load TLPO (30 flags + LLM integration settings) ------------ #
        tlpo_path = BASE_DIR / "tlpo_tcmve.json"
        if not tlpo_path.exists():
            raise FileNotFoundError("tlpo_tcmve.json missing")
        with tlpo_path.open("r", encoding="utf-8") as f:
            self.tlpo = json.load(f)

        # --- LLM configs from TLPO -------------------------------------- #
        gen_cfg = self.tlpo["tcmve_integration"]["generator_settings"]
        ver_cfg = self.tlpo["tcmve_integration"]["verifier_settings"]
        arb_cfg = self.tlpo["tcmve_integration"]["arbiter_settings"]

        # NOTE:
        # We respect TLPO-provided settings (temperature, max_tokens etc.)
        # and only supply the `model` argument explicitly.
        try:
            self.generator = ChatOpenAI(model="gpt-4o", **gen_cfg)
            self.verifier = ChatAnthropic(model="claude-3-opus-20240229", **ver_cfg)
            self.arbiter = ChatGroq(model="grok-4", **arb_cfg)
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise

        # --- System prompt (pure metaphysics) --------------------------- #
        sys_path = BASE_DIR / "tcmve_system.txt"
        if not sys_path.exists():
            raise FileNotFoundError("tcmve_system.txt missing")
        self.system_prompt = sys_path.read_text(encoding="utf-8")

        # --- Ontology (for enriched context, as in cross-LLM version) --- #
        onto_path = BASE_DIR / "ontology.txt"
        if not onto_path.exists():
            raise FileNotFoundError("ontology.txt missing")
        self.ontology = onto_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------- #
    # Core Loop (Cross-LLM Orchestration + TLPO Integration)
    # ------------------------------------------------------------------- #
    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute TCMVE loop on a query.

        Flow per round:
        1. Generator proposes a metaphysical answer (from four causes + ontology).
        2. Verifier attacks the proposition; if no refutation is found → converged.
        3. After convergence or max rounds, Arbiter adjudicates if needed.
        4. Final answer is scored with FULL TLPO (30 flags, 3 LLMs).
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        history: List[Dict[str, Any]] = []

        logger.info(f"Starting TCMVE on: {query}")

        final_answer: str | None = None
        converged = False

        for r in range(1, self.max_rounds + 1):
            round_data: Dict[str, Any] = {
                "round": r,
                "generator_input": "",
                "proposition": "",
                "verifier_input": "",
                "refutation": "",
            }

            # ---------- Generator (GPT-4o) ---------- #
            gen_msg = (
                f"[ROUND {r}] Propose answer to: {query}\n"
                f"Derive from four causes.\n"
                f"Ontology context: {self.ontology[:MAX_ONTOLOGY_CHARS]}..."
            )
            round_data["generator_input"] = gen_msg

            try:
                prop_resp = self.generator.invoke(
                    messages + [{"role": "user", "content": gen_msg}]
                )
                proposition = (prop_resp.content or "").strip()
                logger.info(f"Round {r} - Proposition generated")
                messages.append({"role": "user", "content": gen_msg})
                messages.append({"role": "assistant", "content": proposition})
            except Exception as e:
                proposition = f"[ERROR: Generator failed: {e}]"
                logger.error(proposition)
                messages.append({"role": "assistant", "content": proposition})

            round_data["proposition"] = proposition

            # ---------- Verifier (Claude-3 Opus) ---------- #
            ver_msg = (
                f'VERIFY PROPOSITION:\n"{proposition}"\n\n'
                "Refute via metaphysical contradiction or say:\n"
                '"No refutation — converged."'
            )
            round_data["verifier_input"] = ver_msg

            try:
                ver_resp = self.verifier.invoke(
                    messages + [{"role": "user", "content": ver_msg}]
                )
                refutation = (ver_resp.content or "").strip()
                logger.info(f"Round {r} - Verification complete")
                messages.append({"role": "user", "content": ver_msg})
                messages.append({"role": "assistant", "content": refutation})
            except Exception as e:
                refutation = f"[ERROR: Verifier failed: {e}]"
                logger.error(refutation)
                messages.append({"role": "assistant", "content": refutation})

            round_data["refutation"] = refutation
            history.append(round_data)

            # ---------- Convergence check ---------- #
            if any(phrase in refutation.lower() for phrase in CONVERGENCE_PHRASES):
                final_answer = proposition
                converged = True
                logger.info(f"CONVERGED at round {r}")
                break

        # ---------- Arbiter fallback (Groq) ---------- #
        if not converged:
            logger.warning("Max rounds reached → invoking Arbiter")
            arb_summary = "\n".join(
                f"Round {h['round']}: {h['proposition'][:200]}... → {h['refutation'][:200]}..."
                for h in history
            )
            arb_msg = "ADJUDICATE FINAL TRUTH:\n" + arb_summary
            try:
                arb_resp = self.arbiter.invoke(
                    messages + [{"role": "user", "content": arb_msg}]
                )
                final_answer = (arb_resp.content or "").strip()
            except Exception as e:
                final_answer = f"[ARBITER ERROR: {e}]"
                logger.error(final_answer)

        if final_answer is None:
            final_answer = "[NO VALID ANSWER: all models failed or returned empty output]"

        # ---------- Post-convergence TLPO scoring ---------- #
        tlpo_scores = self._evaluate_with_tlpo(final_answer, query)

        # ---------- Cross-LLM simple metrics (from previous version) ---- #
        metrics = self._compute_metrics(history)

        # ---------- Build result ---------- #
        result: Dict[str, Any] = {
            "query": query,
            "final_answer": final_answer,
            "converged": converged,
            "rounds": len(history),
            "history": history,
            "tlpo_scores": tlpo_scores,
            "tlpo_markup": self._generate_tlpo_markup(
                tlpo_scores, final_answer, query
            ),
            "metrics": metrics,
        }

        # ---------- Save XML output ---------- #
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", query[:60]) or "tcmve_output"
        out_path = RESULTS_DIR / f"{safe_name}.xml"
        out_path.write_text(result["tlpo_markup"], encoding="utf-8")
        logger.info(f"TLPO XML output saved → {out_path}")

        return result

    # ------------------------------------------------------------------- #
    # TLPO Scoring (30 flags, 3 LLMs)
    # ------------------------------------------------------------------- #
    def _evaluate_with_tlpo(self, answer: str, query: str) -> Dict[str, Any]:
        """
        Evaluate the final answer under the FULL TLPO:

        Each LLM scores:
        - 30 individual flags (0.00–1.00)
        - TQI (Truth Quality Index)
        - TCS (Thomistic Coherence Score)

        Then we compute a weighted aggregate:
        - Verifier (Claude) = rigor (weight 0.6)
        - Arbiter (Groq)    = telos (weight 0.3)
        - Generator (GPT-4) = creativity (weight 0.1)
        """
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
                content = (resp.content or "").strip()
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
        """
        Generate a full XML TLPO diagnostic document:

        - 30 <flag> nodes, one per Thomistic flag
        - Per-LLM scores (generator / verifier / arbiter)
        - Weighted TQI and TCS
        - Audit trail (timestamp, user, location)
        """
        flags_xml: List[str] = []

        for i in range(1, TLPO_FLAGS + 1):
            flag_def = next(
                (f for f in self.tlpo["flags"] if f.get("flag_id") == i), {}
            )
            name = flag_def.get("flag_name", f"Flag_{i}")
            thom = flag_def.get("thomistic_link", "N/A")

            gen = scores.get("generator", {}).get("flag_scores", {}).get(str(i), "N/A")
            ver = scores.get("verifier", {}).get("flag_scores", {}).get(str(i), "N/A")
            arb = scores.get("arbiter", {}).get("flag_scores", {}).get(str(i), "N/A")

            name_esc = html.escape(str(name), quote=True)
            thom_esc = html.escape(str(thom), quote=True)

            flags_xml.append(
                f"""  <flag id="{i}" name="{name_esc}">
    <generator>{gen}</generator>
    <verifier>{ver}</verifier>
    <arbiter>{arb}</arbiter>
    <thomistic>{thom_esc}</thomistic>
  </flag>"""
            )

        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        tqi_w = scores.get("weighted_tqi", 0.0)
        tcs_w = scores.get("weighted_tcs", 0.0)

        query_esc = html.escape(str(query), quote=True)
        answer_esc = html.escape(str(answer), quote=True)

        return f"""<tlpo_markup version="1.2" tcmve_mode="full_diagnostic">
  <query>{query_esc}</query>
  <proposition>{answer_esc}</proposition>
{os.linesep.join(flags_xml)}
  <tqi_weighted>{tqi_w}</tqi_weighted>
  <tcs_weighted>{tcs_w}</tcs_weighted>
  <audit>
    <timestamp>{now}</timestamp>
    <user>{USER_TAG}</user>
    <location>{USER_LOCATION}</location>
  </audit>
</tlpo_markup>"""

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
