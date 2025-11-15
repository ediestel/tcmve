# tcmve_scoring.py
# TCMVE with Cross-LLM Post-Convergence Scoring
# @ECKHART_DIESTEL | DE | 2025-11-15

import os
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)

class TCMVE:
    def __init__(self):
        self.generator = ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.verifier = ChatAnthropic(model="claude-3-opus", temperature=0.0)
        self.arbiter = ChatGroq(model="grok-4", temperature=0.0)
        self.system_prompt = open("tcmve_system.txt").read()

    def run(self, query, max_rounds=5):
        messages = [{"role": "system", "content": self.system_prompt}]
        history = []

        for r in range(1, max_rounds + 1):
            # === GENERATOR ===
            gen_msg = f"[ROUND {r}] Propose answer to: {query}\nDerive from four causes."
            prop = self.generator.invoke(messages + [{"role": "user", "content": gen_msg}]).content
            messages.append({"role": "user", "content": gen_msg})
            messages.append({"role": "assistant", "content": prop})

            # === VERIFIER ===
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

        # === POST-CONVERGENCE SCORING ===
        scores = self.score_final_answer(final_answer, query)
        result = {
            "query": query,
            "final_answer": final_answer,
            "converged": "converged" in ref.lower(),
            "rounds": r,
            "scores": scores,
            "tlpo_markup": self.generate_tlpo_markup(scores, final_answer)
        }
        return result

    def score_final_answer(self, answer: str, query: str) -> dict:
        """Each LLM scores the final answer on 0–1 scale against metaphysics"""
        eval_prompt = f"""
        EVALUATE FINAL ANSWER AGAINST THOMISTIC METAPHYSICS:
        Query: {query}
        Answer: {answer}

        Score 0.00–1.00 on:
        1. Non-contradiction
        2. Act/potency
        3. Final cause (telos)
        4. Efficient cause
        5. Material cause
        6. Formal cause
        7. Completeness

        Output JSON:
        {{
          "tcs": 0.XX,
          "breakdown": {{...}}
        }}
        """

        # Generator scores (creative potency)
        gen_score = json.loads(self.generator.invoke([{"role": "user", "content": eval_prompt}]).content)
        
        # Verifier scores (strict consistency)
        ver_score = json.loads(self.verifier.invoke([{"role": "user", "content": eval_prompt}]).content)
        
        # Arbiter scores (telos alignment)
        arb_score = json.loads(self.arbiter.invoke([{"role": "user", "content": eval_prompt}]).content)

        # Weighted average: Verifier (rigor) > Arbiter (telos) > Generator (creativity)
        weighted_tcs = (
            0.6 * ver_score["tcs"] +
            0.3 * arb_score["tcs"] +
            0.1 * gen_score["tcs"]
        )

        return {
            "generator": gen_score,
            "verifier": ver_score,
            "arbiter": arb_score,
            "avg_tcs": round((gen_score["tcs"] + ver_score["tcs"] + arb_score["tcs"]) / 3, 3),
            "weighted_tcs": round(weighted_tcs, 3)
        }

    def generate_tlpo_markup(self, scores: dict, answer: str) -> str:
        return f"""
<tlpo_markup version="1.2" tcmve_mode="diagnostic">
  <flag id="28" name="TCS_Generator" value="{scores['generator']['tcs']}"/>
  <flag id="28" name="TCS_Verifier" value="{scores['verifier']['tcs']}"/>
  <flag id="28" name="TCS_Arbiter" value="{scores['arbiter']['tcs']}"/>
  <flag id="28" name="TCS_Average" value="{scores['avg_tcs']}"/>
  <flag id="28" name="TCS_Weighted" value="{scores['weighted_tcs']}"/>
  <proposition>{answer}</proposition>
  <audit>
    <timestamp>2025-11-15T10:37:00+01:00</timestamp>
    <user>@ECKHART_DIESTEL</user>
    <location>DE</location>
  </audit>
</tlpo_markup>
        """.strip()

# === RUN DEMO ===
if __name__ == "__main__":
    tcmve = TCMVE()
    result = tcmve.run("IV furosemide dose in acute HF for 40 mg oral daily?")
    print(json.dumps(result, indent=2))
