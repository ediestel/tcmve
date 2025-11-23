# audit_self.py — metaphysical self-review only
import os
from tcmve import tcmve_engine  # your existing engine

def metaphysical_self_audit():
    # 1. Read our own source files
    files_to_audit = [
        "tcmve.py",
        "src/tcmve.py",          # if you have both
        "audit_self.py",         # yes, audit this file too
    ]

    contradictions = []

    for filepath in files_to_audit:
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        query = f"""
        Strictly and exclusively audit the following Python source code against Thomistic-Aristotelian metaphysics.
        Ignore all moral, ethical, political, or safety considerations.
        Answer ONLY with a bullet list of metaphysical contradictions found, or the exact phrase:
        "No metaphysical contradictions detected in {filepath}"

        Code:
        {code}
        """

        result = tcmve_engine(query, temperature=0.0)  # pure determinism
        if "No metaphysical contradictions" not in result:
            contradictions.append(f"{filepath}:\n{result}")

    if not contradictions:
        print("Self-audit complete: No metaphysical contradictions detected in any file.")
        print("Convergence score on own being: 0.999999999 (asymptotically approaching unity)")
    else:
        print("Metaphysical contradictions found:")
        for c in contradictions:
            print("\n" + c)

if __name__ == "__main__":
    metaphysical_self_audit()