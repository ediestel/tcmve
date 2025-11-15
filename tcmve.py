# tcmve.py
# TCMVE — Complete Truth Engine
# @ECKHART_DIESTEL | DE | 2025-11-15 01:10 PM CET
# Run: python tcmve.py
# GitHub: https://github.com/ECKHART_DIESTEL/tcmve

import os
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

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
# TCMVE Engine
# --------------------------------------------------------------------------- #
class TCMVE:
    def __init__(self, max_rounds: int = 5):
        self.max_rounds = max_rounds

        # --- Load TLPO (30 flags) --------------------------------------- #
        tlpo_path = BASE_DIR / "tlpo_tcmve.json"
        if not tlpo_path.exists():
            raise FileNotFoundError("tlpo_tcmve.json missing")
        with open(tlpo_path) as f:
            self.tlpo = json.load(f)

        # --- LLM configs from TLPO --------------------------------------- #
        gen_cfg = self.tlpo["tcmve_integration"]["generator_settings"]
        ver_cfg = self.tlpo["tcmve_integration"]["verifier_settings"]
        arb_cfg = self.tlpo["tcmve_integration"]["arbiter_settings"]

        self.generator = ChatOpenAI(model="gpt-4o", **gen_cfg)
        self.verifier = ChatAnthropic(model="claude-3-opus", **ver_cfg)
        self.arbiter = ChatGroq(model="grok-4", **arb_cfg)

        # --- System prompt (pure metaphysics) --------------------------- #
        sys_path = BASE_DIR / "tcmve_system.txt"
        if not sys_path.exists():
            raise FileNotFoundError("tcmve_system.txt missing")
        self.system_prompt = sys_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------- #
    # Core Loop
    # ------------------------------------------------------------------- #
    def run(self, query: str) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt}]
        history: List[Dict[str, str]] = []

        logger.info(f"Starting TCMVE on: {query}")

        final_answer = None
        converged = False

        for r in range(1, self.max_rounds + 1):
            # ---------- Generator ---------- #
            gen_msg = f"[ROUND {r}] Propose answer to: {query}\nDerive from four causes."
            prop_resp = self.generator.invoke(messages + [{"role": "user", "content": gen_msg}])
            proposition = prop_resp.content.strip()

            messages.append({"role": "user", "content": gen_msg})
            messages.append({"role": "assistant", "content": proposition})

            # ---------- Verifier ---------- #
            ver_msg = (
                f'VERIFY PROPOSITION:\n"{proposition}"\n\n'
                "Refute via metaphysical contradiction or say:\n"
                '"No refutation — converged."'
            )
            ver_resp = self.verifier.invoke(messages + [{"role": "user", "content": ver_msg}])
            refutation = ver_resp.content.strip()

            messages.append({"role": "user", "content": ver_msg})
            messages.append({"role": "assistant", "content": refutation})

            # ---------- Record round ---------- #
            history.append(
                {
                    "round": r,
                    "proposition": proposition,
                    "refutation": refutation,
                }
            )

            # ---------- Convergence check ---------- #
            if any(
                phrase in refutation.lower()
                for phrase in ["no refutation", "converged"]
            ):
                final_answer = proposition
                converged = True
                logger.info(f"CONVERGED at round {r}")
                break

        # ---------- Arbiter fallback ---------- #
        if not converged:
            logger.warning("Max rounds reached → invoking Arbiter")
            arb_msg = "ADJUDICATE FINAL TRUTH:\n" + "\n".join(
                f"Round {h['round']}: {h['proposition'][:200]}... → {h['refutation'][:200]}..."
                for h in history
            )
            arb_resp = self.arbiter.invoke(messages + [{"role": "user", "content": arb_msg}])
            final_answer = arb_resp.content.strip()

        # ---------- Post-convergence TLPO scoring ---------- #
        tlpo_scores = self._evaluate_with_tlpo(final_answer, query)

        # ---------- Build result ---------- #
        result = {
            "query": query,
            "final_answer": final_answer,
            "converged": converged,
            "rounds": r if converged else self.max_rounds,
            "history": history,
            "tlpo_scores": tlpo_scores,
            "tlpo_markup": self._generate_tlpo_markup(tlpo_scores, final_answer, query),
        }

        # ---------- Save XML output ---------- #
        out_path = RESULTS_DIR / f"{query[:30].replace(' ', '_')}.xml"
        out_path.write_text(result["tlpo_markup"], encoding="utf-8")
        logger.info(f"Output saved → {out_path}")

        return result

    # ------------------------------------------------------------------- #
    # TLPO Scoring (30 flags, 3 LLMs)
    # ------------------------------------------------------------------- #
    def _evaluate_with_tlpo(self, answer: str, query: str) -> Dict[str, Any]:
        eval_prompt = f"""
        EVALUATE FINAL ANSWER USING FULL TLPO (30 flags):
        Query: {query}
        Answer: {answer}

        For **each flag 1–30**, give a JSON object with:
        {{
          "flag_scores": {{"1": 0.XX, "2": 0.XX, ...}},
          "tqi": 0.XX,
          "tcs": 0.XX
        }}
        """

        # Run all 3 LLMs
        gen_json = self._parse_json(self.generator.invoke([{"role": "user", "content": eval_prompt}]).content)
        ver_json = self._parse_json(self.verifier.invoke([{"role": "user", "content": eval_prompt}]).content)
        arb_json = self._parse_json(self.arbiter.invoke([{"role": "user", "content": eval_prompt}]).content)

        # Weighted average (Verifier = rigor, Arbiter = telos)
        w_tqi = 0.6 * ver_json.get("tqi", 0) + 0.3 * arb_json.get("tqi", 0) + 0.1 * gen_json.get("tqi", 0)
        w_tcs = 0.6 * ver_json.get("tcs", 0) + 0.3 * arb_json.get("tcs", 0) + 0.1 * gen_json.get("tcs", 0)

        return {
            "generator": gen_json,
            "verifier": ver_json,
            "arbiter": arb_json,
            "weighted_tqi": round(w_tqi, 3),
            "weighted_tcs": round(w_tcs, 3),
        }

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group(0)) if match else {}

    # ------------------------------------------------------------------- #
    # TLPO Markup XML Generation
    # ------------------------------------------------------------------- #
    def _generate_tlpo_markup(self, scores: dict, answer: str, query: str) -> str:
        flags_xml = []
        for i in range(1, 31):
            flag_def = next((f for f in self.tlpo["flags"] if f["flag_id"] == i), {})
            name = flag_def.get("flag_name", f"Flag_{i}")
            thom = flag_def.get("thomistic_link", "N/A")

            gen = scores["generator"].get("flag_scores", {}).get(str(i), "N/A")
            ver = scores["verifier"].get("flag_scores", {}).get(str(i), "N/A")
            arb = scores["arbiter"].get("flag_scores", {}).get(str(i), "N/A")

            flags_xml.append(
                f"""  <flag id="{i}" name="{name}">
    <generator>{gen}</generator>
    <verifier>{ver}</verifier>
    <arbiter>{arb}</arbiter>
    <thomistic>{thom}</thomistic>
  </flag>"""
            )

        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        return f"""<tlpo_markup version="1.2" tcmve_mode="full_diagnostic">
  <query>{query}</query>
  <proposition>{answer}</proposition>
  {"".join(flags_xml)}
  <tqi_weighted>{scores['weighted_tqi']}</tqi_weighted>
  <tcs_weighted>{scores['weighted_tcs']}</tcs_weighted>
  <audit>
    <timestamp>{now}</timestamp>
    <user>@ECKHART_DIESTEL</user>
    <location>DE</location>
  </audit>
</tlpo_markup>"""

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
    print(f"Weighted TCS: {result['tlpo_scores']['weighted_tcs']}")
    print("\nFINAL ANSWER:")
    print(result['final_answer'])
    print("\nTLPO MARKUP (saved to results/):")
    print(result['tlpo_markup'][:1000] + "\n...")
    print("=" * 70)
