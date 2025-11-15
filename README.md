# TCMVE — Truth-Convergent Metaphysical Verification Engine

**Pure Thomistic Truth Generation from First Principles**

- No domain ontology  
- No external citations  
- No LLM parameter tuning  
- Truth from act/potency and four causes  

## Quick Start
```bash
git clone https://github.com/ECKHART_DIESTEL/tcmve.git
cd tcmve
python -m venv .venv && source .venv/bin/activate
pip install langchain-openai langchain-anthropic langchain-groq python-dotenv
cp .env.example .env  # Add API keys
python tcmve_crossllm.py
