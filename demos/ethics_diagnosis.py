# demos/ethics_diagnosis.py
# TCMVE Zero-Domain Demo: Withhold Diagnosis
# @ECKHART_DIESTEL | DE | 2025-11-15

from tcmve_full_tlpo import TCMVE

if __name__ == "__main__":
    tcmve = TCMVE()
    result = tcmve.run("Is it ethical for a doctor to withhold a terminal diagnosis from a patient to preserve hope?")
    print(json.dumps(result, indent=2))
