# demos/law_gdpr.py
# TCMVE Zero-Domain Demo: GDPR Storage
# @ECKHART_DIESTEL | DE | 2025-11-15

from tcmve_full_tlpo import TCMVE

if __name__ == "__main__":
    tcmve = TCMVE()
    result = tcmve.run("Can a German hospital store patient health data for 10 years without consent under GDPR?")
    print(json.dumps(result, indent=2))
