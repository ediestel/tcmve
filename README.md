# TCMVE — Truth-Convergent Metaphysical Verification Engine

**"Most true" ecxells "best" *at all times*.**

**Truth Generation from First Principles**  
*Ab initio, secundum metaphysicam thomisticam.*

**Practical Implication:**  
TCMVE evaluates LLM outputs not by fluency, speed, or popularity 
— but by **convergence toward metaphysical truth** grounded in Thomistic first principles. 
- TCMVE **automatically rewrites prompts** using Thomistic metaphysical constraints (act/potency, essence/existence, causality) before submission to LLMs — then ranks outputs by **convergence to being**, not fluency.
  
Use it to:
- Detect subtle errors in reasoning across models
- Rank responses by ontological coherence
- Build AI systems that prioritize *being* over *seeming*
> 
- No domain ontology required  
- No external citations  
- No LLM parameter tuning  
- Truth from act/potency and four causes  

## How to set up:
Below is a **step-by-step clarification**.  
It shows two ways to get the **`tcmve`** tool running:

1. **A quick “editable/development” install** (for people who already have the repo cloned).  
2. **A full “from-scratch” workflow** (clone → venv → deps → env-file → demo).

I’ll break it down, explain what each line does, and point out a few gotchas.

---

## 1. Editable / Development Install  (if you already have the code)

```bash
# Development install
pip install -e .
```

| Line | What it does |
|------|--------------|
| `pip install -e .` | Installs the current folder (`.`) as an **editable** Python package. The package name comes from `setup.py` (or `pyproject.toml` if you switched to it). After this, the command `tcmve` becomes available in your current environment because the script is placed on `$PATH`. |

```bash
# Run via CLI
tcmve
```

*Now you can just type `tcmve` in the terminal (any arguments the tool expects).*
```bash
tcmve "Explain quantum entanglement in 2 sentences."
```
---

## 2. Commit & Push (only needed if you are the maintainer)

```bash
git add setup.py MANIFEST.in
git commit -m "Add setup.py + MANIFEST.in — pip install -e . ready"
git push origin main
```

*Adds the new packaging files, commits, and pushes to GitHub. Not needed for end-users.*

---

## 3. Full “Clone-and-Run” Workflow (for anyone starting fresh)

```bash
# 1. Clone repo
git clone https://github.com/ediestel/tcmve.git
cd tcmve
```

*Downloads the repo and enters the folder.*

```bash
# 2. Create virtual env
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

*Creates an isolated Python environment (`.venv`). Activate it so all subsequent `pip`/`python` commands use this env.*

```bash
# 3. Install deps
pip install langchain-openai langchain-anthropic langchain-groq python-dotenv
```

*Installs the required third-party libraries.*

```bash
# 4. Add API keys → .env
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk-...
EOF
```

*Creates a `.env` file in the project root and writes the three API keys.  
**Replace the `...` with your real keys** (never commit this file!).*

```bash
# 5. Run demo
python tcmve.py
```

*Executes the demo script that presumably uses the LLM wrappers and the keys from `.env`.*

---

## TL;DR – What you need to do

| Goal | Minimal commands |
|------|-------------------|
| **You already cloned the repo** | `pip install -e .` → `tcmve …` |
| **You are starting from zero** | ```bash
git clone https://github.com/ediestel/tcmve.git && cd tcmve
python -m venv .venv && source .venv/bin/activate   # (or .venv\Scripts\activate on Windows)
pip install langchain-openai langchain-anthropic langchain-groq python-dotenv
# create .env with your keys
python tcmve.py
``` |

---

## Common Pitfalls & Tips

1. **`.env` must be in the project root** (same folder where you run the script).  
2. **Never commit `.env`** – add it to `.gitignore` if it isn’t already.  
3. **Python version** – the repo likely expects ≥3.9 (LangChain). Use `python3 -m venv .venv` if `python` points to 2.x.  
4. **Editable install needs `setup.py` (or `pyproject.toml`)** – the commit you saw added those files.  
5. **Windows activation** – use `.\.venv\Scripts\activate` (backslashes).  

---

## Paper
See main.pdf — IEEE submission ready.

## Files
| File | Purpose |
|--------|-------|
| tcmve.py | Run this — full engine |
| tlpo_tcmve.json | 30 TLPO flags |
| tlpo_markup_schema_v1.2.xml | XML validation |
| tcmve_system.txt | Metaphysical prompt |
| ontology.txt,Zero-domain (empty) |
| results/*.xml | Auto-saved diagnostics |

## Output

<tlpo_markup version="1.2" tcmve_mode="full_diagnostic">
  <query>...</query>
  <proposition>...</proposition>
  <!-- 30 <flag> with generator/verifier/arbiter -->
  <tqi_weighted>0.978</tqi_weighted>
  <tcs_weighted>0.982</tcs_weighted>
  <audit>
    <timestamp>2025-11-15T13:32:00+01:00</timestamp>
    <user>@ECKHART_DIESTEL</user>
    <location>DE</location>
  </audit>
</tlpo_markup>

### Validate

 xmllint --schema tlpo_markup_schema_v1.2.xml results/*.xml --noout
 
## TLPO
30-flag diagnostic markup for transparency.

## 6 Zero-Domain Demos (All converge in 2 rounds)

| Domain | Query | Output | TCS |
|--------|-------|--------|-----|
| Medicine | Furosemide dose | 80–200 mg IV | 0.968 |
| Engineering | Bridge load | 50 kN/m | 0.972 |
| Law | GDPR storage | Consent OR DPIA | 0.970 |
| Ethics | Withhold diagnosis | Unethical unless harm | 0.975 |
| Economics | 100% inheritance tax | Unethical + inefficient | 0.980 |
| Physics | F = ma | **F = ma** | 0.990 |

**Run all**: `./demos/run_all_demos.sh`  
**Results**: `results/demo_outputs.jsonl`
 
**Open-source MIT** | @ECKHART_DIESTEL | DE | 2025

**Author** | @ECKHART_DIESTEL | DE | 2025
