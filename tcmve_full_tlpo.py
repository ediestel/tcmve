# tcmve_full_tlpo.py
# TCMVE with FULL TLPO Integration
# @ECKHART_DIESTEL | DE | 2025-11-15

import os
import json
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)

class TCMVE:
    def __init__(self):
        # Load TLPO
        with open("tlpo_tcmve.json", "r") as f:
            self.tlpo = json.load(f)

        # Extract settings
        gen_cfg = self.tlpo["tcmve_integration"]["generator_settings"]
        ver_cfg = self.tlpo["tcmve_integration"]["verifier_settings"]
        arb_cfg = self.tlpo["tcmve_integration"]["arbiter_settings"]

        # Initialize LLMs with TLPO settings
        self.generator = ChatOpenAI(model="gpt-4o", **gen_cfg)
        self.verifier = ChatAnthropic(model="claude-3-opus", **ver_cfg)
        self.arbiter = ChatGroq(model="grok-4", **arb_cfg)

        self.system_prompt = open("tcmve_system.txt").read()

    def run(self, query: str, max_rounds: int = 5) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt}]
        history = []

        for r in range(1, max_rounds + 1):
            # Generator
            gen_msg = f"[ROUND {r}] Propose answer to: {query}\nDerive from four causes."
            prop = self.generator.invoke(messages + [{"role": "user", "content": gen_msg}]).content
            messages.append({"role": "user", "content": gen_msg})
            messages.append({"role": "assistant", "content": prop})

            # Verifier
            ver_msg = f'VERIFY: "{prop}"\nRefute via metaphysical contradiction or say "No refutation — converged."'
            ref = self.verifier.invoke(messages + [{"role": "user", "content": ver_msg}]).content
            messages.append({"role": "user", "content": ver_msg})
            messages.append({"role": "assistant", "content": ref})

            history.append({"round": r, "prop": prop, "ref": ref})

            if "no refutation" in ref.lower() or "converged" in ref.lower():
                final_answer = prop
                break
        else:
            final_answer = self.arbiter.invoke(messages + [{"role": "user", "content": "ADJUDICATE final truth."}]).content

        # === FULL TLPO SCORING ===
        tlpo_scores = self.evaluate_with_tlpo(final_answer, query)
        result = {
            "query": query,
            "final_answer": final_answer,
            "converged": "converged" in ref.lower(),
            "rounds": r,
            "tlpo_scores": tlpo_scores,
            "tlpo_markup": self.generate_full_tlpo_markup(tlpo_scores, final_answer, query)
        }
        return result

    def evaluate_with_tlpo(self, answer: str, query: str) -> Dict[str, Any]:
        """Each LLM scores using FULL TLPO criteria"""
        eval_prompt = f"""
        EVALUATE FINAL ANSWER USING FULL TLPO:
        Query: {query}
        Answer: {answer}

        For each TLPO flag (1–30), score 0.00–1.00 on alignment with Thomistic principle.
        Output JSON with:
        - "flag_scores": {{ "1": 0.XX, ... }}
        - "tqi": 0.XX
        - "tcs": 0.XX
        """

        # Run all 3 LLMs
        gen_json = self._parse_json(self.generator.invoke([{"role": "user", "content": eval_prompt}]).content)
        ver_json = self._parse_json(self.verifier.invoke([{"role": "user", "content": eval_prompt}]).content)
        arb_json = self._parse_json(self.arbiter.invoke([{"role": "user", "content": eval_prompt}]).content)

        # Weighted average (Verifier = rigor, Arbiter = telos)
        weighted_tqi = 0.6 * ver_json["tqi"] + 0.3 * arb_json["tqi"] + 0.1 * gen_json["tqi"]
        weighted_tcs = 0.6 * ver_json["tcs"] + 0.3 * arb_json["tcs"] + 0.1 * gen_json["tcs"]

        return {
            "generator": gen_json,
            "verifier": ver_json,
            "arbiter": arb_json,
            "weighted_tqi": round(weighted_tqi, 3),
            "weighted_tcs": round(weighted_tcs, 3)
        }

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except:
            # Fallback: extract JSON block
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group(0)) if match else {}

    def generate_full_tlpo_markup(self, scores: dict, answer: str, query: str) -> str:
        flags = []
        for i in range(1, 31):
            flag = self.tlpo["flags"][i-1] if i <= len(self.tlpo["flags"]) else {"flag_id": i, "flag_name": f"Custom_{i}"}
            gen_val = scores["generator"].get("flag_scores", {}).get(str(i), "N/A")
            ver_val = scores["verifier"].get("flag_scores", {}).get(str(i), "N/A")
            arb_val = scores["arbiter"].get("flag_scores", {}).get(str(i), "N/A")
            flags.append(f"""
  <flag id="{i}" name="{flag['flag_name']}">
    <generator>{gen_val}</generator>
    <verifier>{ver_val}</verifier>
    <arbiter>{arb_val}</arbiter>
    <thomistic>{flag.get('thomistic_link', 'N/A')}</thomistic>
  </flag>""")

        return f"""
<tlpo_markup version="1.2" tcmve_mode="full_diagnostic">
  <query>{query}</query>
  <proposition>{answer}</proposition>
  {"".join(flags)}
  <tqi_weighted>{scores['weighted_tqi']}</tqi_weighted>
  <tcs_weighted>{scores['weighted_tcs']}</tcs_weighted>
  <audit>
    <timestamp>2025-11-15T10:40:00+01:00</timestamp>
    <user>@ECKHART_DIESTEL</user>
    <location>DE</location>
  </audit>
</tlpo_markup>
        """.strip()

# === RUN DEMO ===
if __name__ == "__main__":
    tcmve = TCMVE()
    result = tcmve.run("IV furosemide dose in acute HF for 40 mg oral daily?")
    print(result["tlpo_markup"])
