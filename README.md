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

## Author
@ECKHART_DIESTEL | DE | 2025
