import pathlib

import pytest

from app.scoring import WEIGHTS, normalise_record, score_lead

SAMPLE = {
    "Name": "Rahul Sharma", "Company Name": "ABC Technologies",
    "Website": "www.abctechnologies.com", "Job Title": "Founder", "Industry": "IT Services",
    "Company Location": "Bangalore", "Company Size": "50-100 Employees",
    "Lead Source": "Website", "Requirement": "Looking for CRM software",
    "Email": "rahul@example.com", "Phone": "+91 9876543210",
}

STRONG = {
    "Name": "A", "Company Name": "X", "Website": "https://x.example", "Job Title": "CEO",
    "Industry": "Finance", "Company Location": "Mumbai, India", "Company Size": "5000+ employees",
    "Lead Source": "Referral", "Requirement": "Budget approved; planning implementation next quarter.",
    "Email": "a@x.example", "Phone": "+1-202-555-0100",
}

WEAK = {
    "Name": "B", "Company Name": "Y", "Website": "", "Job Title": "Student", "Industry": "Education",
    "Company Location": "Other", "Company Size": "1-10 employees", "Lead Source": "Cold Outreach",
    "Requirement": "Looking for a free tool for a small personal project.", "Email": "", "Phone": "",
}


def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


def test_same_input_gives_same_output():
    assert score_lead(STRONG) == score_lead(STRONG)


def test_strong_lead_is_hot_and_weak_lead_is_cold():
    assert score_lead(STRONG)["classification"] == "Hot"
    assert score_lead(WEAK)["classification"] == "Cold"


def test_score_within_bounds():
    for lead in (STRONG, WEAK, SAMPLE, {}):
        s = score_lead(lead)["score"]
        assert 0 <= s <= 100


def test_missing_fields_are_handled_not_crashing():
    r = score_lead({})
    assert r["score"] >= 0
    assert r["confidence"]["level"] == "Low"
    assert "Job Title" in r["missing_fields"]


def test_key_field_missing_caps_confidence_at_medium():
    lead = dict(STRONG)
    lead.pop("Job Title")
    lead["Job Title"] = ""
    r = score_lead(lead)
    assert r["confidence"]["level"] in ("Medium", "Low")


def test_invalid_email_is_flagged():
    r = score_lead({**STRONG, "Email": "not-an-email"})
    assert any("Email" in w for w in r["data_warnings"])
    assert "Email" in r["missing_fields"]


def test_header_variants_are_normalised():
    raw = {"\ufeffname": "Z", " Job Title ": "CTO", "company size": "201-500 employees"}
    lead = normalise_record(raw)
    assert lead["Name"] == "Z" and lead["Job Title"] == "CTO"
    assert lead["Company Size"] == "201-500 employees"


def test_free_tool_requirement_routes_to_self_serve():
    r = score_lead(WEAK)
    assert "self-serve" in r["recommended_next_action"].lower()


def test_hot_lead_has_positive_and_explanation():
    r = score_lead(STRONG)
    assert r["positive_factors"]
    assert str(r["score"]) in r["scoring_explanation"]


def test_gate_forces_cold_for_student_even_with_strong_fields():
    r = score_lead({**STRONG, "Job Title": "Student"})
    assert r["classification"] == "Cold"
    assert r["gate_applied"] is True


def test_sample_dataset_runs_end_to_end():
    import csv
    path = pathlib.Path(__file__).resolve().parents[1] / "data" / "lead_scoring_sample_dataset_500.csv"
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    results = [score_lead(r) for r in rows]
    assert len(results) == 500
    assert all(0 <= r["score"] <= 100 for r in results)
