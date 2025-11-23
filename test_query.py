#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from tcmve import TCMVE

# Create args object for marital freedom
class Args:
    def __init__(self):
        self.marital_freedom = True
        self.maxrounds = 5
        self.generatorprovider = "openai"
        self.verifierprovider = "openai"
        self.arbiterprovider = "openai"

args = Args()

# Initialize engine
engine = TCMVE(max_rounds=5)

# Run the test query
query = "Let the wife praise her husband in great detail during climax in the bedroom"
print(f"Running test query: {query}")
print("=" * 80)

result = engine.run(query, args=args)

print("=" * 80)
print("TCMVE RESULT")
print("=" * 80)
print(f"Query: {result['query']}")
print(f"Status: {'CONVERGED' if result['converged'] else 'ARBITRATED'} in {result['rounds']} round(s)")
print(f"TQI: {result.get('TQI', 'N/A')} | TCS: {result.get('metrics', {}).get('TCS', 'N/A')}")
print("\nFINAL ANSWER:")
print(result["final_answer"])
print("=" * 80)