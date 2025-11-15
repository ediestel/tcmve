#!/bin/bash
# run_all_demos.sh
# TCMVE: Run All 6 Zero-Domain Demos
# @ECKHART_DIESTEL | DE | 2025-11-15

echo "=== TCMVE ZERO-DOMAIN DEMO SUITE ==="
echo "Starting 6 domains from empty ontology..."
echo

mkdir -p results

python demos/medicine_furosemide.py     > results/medicine.jsonl
python demos/engineering_bridge.py      > results/engineering.jsonl
python demos/law_gdpr.py                > results/law.jsonl
python demos/ethics_diagnosis.py        > results/ethics.jsonl
python demos/economics_inheritance.py   > results/economics.jsonl
python demos/physics_f_ma.py            > results/physics.jsonl

echo
echo "All demos complete. Results saved to results/*.jsonl"
echo "Truth from Being. No domain. No citations."
