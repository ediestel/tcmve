# === tcmve_crossllm.py ===
cat > tcmve_crossllm.py << 'EOF'
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

class TCMVE:
    def __init__(self):
        self.generator = ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.verifier = ChatAnthropic(model="claude-3-opus", temperature=0.0)
        self.arbiter = ChatGroq(model="grok-4", temperature=0.0)

    def run(self, query, max_rounds=5):
        system_prompt = open("tcmve_system.txt").read()
        messages = [{"role": "system", "content": system_prompt}]
        history = []

        for r in range(1, max_rounds + 1):
            gen_msg = f"[ROUND {r}] Propose answer to: {query}"
            prop = self.generator.invoke(messages + [{"role": "user", "content": gen_msg}]).content
            messages.append({"role": "user", "content": gen_msg})
            messages.append({"role": "assistant", "content": prop})

            ver_msg = f'VERIFY: "{prop}"\nRefute or say "No refutation — converged."'
            ref = self.verifier.invoke(messages + [{"role": "user", "content": ver_msg}]).content
            messages.append({"role": "user", "content": ver_msg})
            messages.append({"role": "assistant", "content": ref})

            history.append({"round": r, "prop": prop, "ref": ref})

            if "no refutation" in ref.lower() or "converged" in ref.lower():
                return {"final": prop, "history": history, "converged": True}

        arb = self.arbiter.invoke(messages + [{"role": "user", "content": "ADJUDICATE final truth."}]).content
        return {"final": arb, "history": history, "converged": False}

if __name__ == "__main__":
    tcmve = TCMVE()
    result = tcmve.run("IV furosemide dose in acute HF?")
    print(json.dumps(result, indent=2))
