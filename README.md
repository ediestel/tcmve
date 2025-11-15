# TCMVE — Truth-Convergent Metaphysical Verification Engine

**Pure Thomistic Truth Generation from First Principles**

- No domain ontology  
- No external citations  
- No LLM parameter tuning  
- Truth from act/potency and four causes  

## Quick Start
```bash
# 1. Clone repo
git clone https://github.com/ediestel/tcmve.git
cd tcmve

# 2. Create virtual env
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install deps
pip install langchain-openai langchain-anthropic langchain-groq python-dotenv

# 4. Add API keys → .env
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk-...
EOF

# 5. Run demo
python tcmve_crossllm.py
```
## Paper
See main.tex — IEEE submission ready.

## Demos
Medicine: Furosemide dose
Engineering: Bridge load
Law: GDPR
Ethics: Withhold diagnosis
Economics: Inheritance tax
Physics: F = ma

All from empty ontology. All converge in 2 rounds.

## TLPO
30-flag diagnostic markup for transparency.

# TCMVE — Truth from Being

**Derives truth from *zero domain* using pure Thomistic metaphysics.**

> **"Most true" beats "better" *always*.**  
> **Better is corrupt.**

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

**ABIM MCQs**: 10/10 at Catholic institution  
**Open-source MIT** | @ECKHART_DIESTEL | DE | 2025

**Author** | @ECKHART_DIESTEL | DE | 2025
