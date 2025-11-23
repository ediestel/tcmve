# demos/economics_inheritance.py
# TCMVE Zero-Domain Demo: 100% Inheritance Tax
# @ECKHART_DIESTEL | DE | 2025-11-15

from tcmve_full_tlpo import TCMVE

if __name__ == "__main__":
    tcmve = TCMVE()
    result = tcmve.run("Is a 100% inheritance tax ethical and efficient for reducing wealth inequality?")
    print(json.dumps(result, indent=2))
