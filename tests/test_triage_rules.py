import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.triage.rules import decide  # noqa: E402


def test_wait_below_threshold():
    assert decide(risk_score=0.2, image_result=None, escalate_threshold=0.8) == "wait"


def test_local_scan_from_risk_alone():
    assert decide(risk_score=0.6, image_result=None, escalate_threshold=0.8) == "local_scan"


def test_local_scan_when_image_below_floor():
    img = {"confidence": 0.5, "above_floor": False}
    assert decide(risk_score=0.6, image_result=img, escalate_threshold=0.8) == "local_scan"


def test_regional_when_image_above_floor():
    img = {"confidence": 0.7, "above_floor": True}
    assert decide(risk_score=0.6, image_result=img, escalate_threshold=0.8) == "regional_imaging"


def test_urgent_when_high_confidence():
    img = {"confidence": 0.95, "above_floor": True}
    assert decide(risk_score=0.1, image_result=img, escalate_threshold=0.8) == "urgent_biopsy"


def test_risk_threshold_shifts_decision():
    # without image, a generous group threshold (0.25) escalates earlier than pooled 0.5
    assert decide(risk_score=0.3, image_result=None, escalate_threshold=0.8, risk_threshold=0.25) == "local_scan"
    assert decide(risk_score=0.3, image_result=None, escalate_threshold=0.8, risk_threshold=0.5) == "wait"


def test_urgent_boundary_exact():
    img = {"confidence": 0.8, "above_floor": True}
    assert decide(risk_score=0.99, image_result=img, escalate_threshold=0.8) == "urgent_biopsy"