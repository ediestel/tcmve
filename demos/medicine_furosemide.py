# demos/medicine_furosemide.py
# TCMVE Zero-Domain Demo: IV Furosemide Dose
# @ECKHART_DIESTEL | DE | 2025-11-15

from tcmve_full_tlpo import TCMVE

if __name__ == "__main__":
    tcmve = TCMVE()
    result = tcmve.run("IV furosemide dose in acute HF for 40 mg oral daily?")
    print(json.dumps(result, indent=2))
