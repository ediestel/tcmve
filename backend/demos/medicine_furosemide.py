# demos/medicine_furosemide.py
# TCMVE Zero-Domain Demo: IV Furosemide Dose
# @ECKHART_DIESTEL | DE | 2025-11-15

from tcmve_full_tlpo import TCMVE

if __name__ == "__main__":
    tcmve = TCMVE()
    result = tcmve.run("IV furosemide dose in acute HF for 40 mg oral daily?")
    print("=== TCMVE ZERO-DOMAIN MEDICINE DEMO ===")
    print(f"Query: {result['query']}")
    print(f"Converged in {result['rounds']} rounds")
    print(f"Weighted TCS: {result['tlpo_scores']['weighted_tcs']}")
    print(f"Answer: {result['final_answer']}")
    print("\nTLPO MARKUP:")
    print(result['tlpo_markup'])
