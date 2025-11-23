import pytest
from backend.tcmve import TCMVE

def test_tcmve_init():
    engine = TCMVE()
    assert engine.max_rounds == 5
    assert engine.nash_mode == "auto"

def test_calculate_v():
    # Test the V calculation logic indirectly via run
    engine = TCMVE()
    result = engine.run("What is 2+2?")
    assert "final_answer" in result
    assert "eIQ" in result