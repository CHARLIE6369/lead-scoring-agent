"""Deterministic, rule-based lead scoring.

The score is computed only from the rules in this file, so identical inputs
always produce identical scores. The optional LLM (see llm.py) only writes
explanations and next-step text from the factors computed here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration: weights and rules. Tune these; they are the methodology.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "job_title": 25,
    "company_size": 15,
    "industry": 15,
    "requirement": 20,
    "lead_source": 10,
    "data_completeness": 10,
    "location": 5,
}
assert sum(WEIGHTS.values()) == 100

# Neutral points used when a factor's input is missing (documented in README).
MISSING_POINTS = {
    "job_title": 8,
    "company_size": 7,
    "industry": 6,
    "requirement": 5,
    "lead_source": 4,
    "location": 2,
}

ALL_FIELDS = ["Name", "Company Name", "Website", "Job Title", "Industry",
              "Company Location", "Company Size", "Lead Source",
              "Requirement", "Email", "Phone"]
COMPLETENESS_FIELDS = ["Website", "Job Title", "Industry", "Company Location",
                       "Company Size", "Lead Source", "Requirement", "Email", "Phone"]
KEY_FIELDS = {"Job Title", "Requirement"}  # if missing, confidence is capped at Medium

# (regex on lower-cased title, points out of 25, label)
TITLE_RULES = [
    (r"\b(ceo|cto|cio|cfo|coo|chief|founder|co-founder)\b", 25, "C-level or founder decision-maker"),
    (r"\bvp\b|vice president", 22, "VP-level decision-maker"),
    (r"\bdirector\b", 20, "Director-level leader"),
    (r"\bhead of\b", 18, "Head-of leader"),
    (r"\bmanager\b", 14, "Manager-level influencer"),
    (r"\b(consultant|analyst|executive)\b", 10, "Individual contributor or advisor"),
    (r"\b(engineer|developer)\b", 8, "Technical practitioner with limited buying authority"),
    (r"\b(student|intern)\b", 2, "Student or intern with no buying authority"),
]

# (regex on lower-cased requirement, points out of 20, label). First match wins.
REQUIREMENT_RULES = [
    (r"free tool|personal project", 2, "Wants a free tool for a personal project"),
    (r"researching|exploring options|no immediate purchase", 6, "Early research with no purchase timeline"),
    (r"would like to understand|understand available features", 10, "Information gathering before deciding"),
    (r"budget approved|proof of concept|pricing proposal|decision expected|\bdemo\b", 20,
     "Explicit buying signal"),
    (r"replacing an existing vendor|enterprise-wide|100-person", 18,
     "Vendor replacement or enterprise-scale need"),
    (r"integrates with|segmentation|automate|manual|scale|ai-powered lead", 15,
     "Clear operational pain point"),
]

COMPANY_SIZE_POINTS = {  # out of 15
    "5000+ employees": 15,
    "1001-5000 employees": 14,
    "501-1000 employees": 12,
    "201-500 employees": 11,
    "51-200 employees": 9,
    "11-50 employees": 6,
    "1-10 employees": 3,
}

INDUSTRY_TIERS = {  # out of 15 for high, 10 medium, 5 low
    "Technology": 15, "IT Services": 15, "Finance": 15, "Cybersecurity": 15,
    "Analytics": 15, "Healthcare": 15, "Retail": 15,
    "Consulting": 10, "Logistics": 10, "Manufacturing": 10, "Automotive": 10,
    "Energy": 10, "Real Estate": 10, "Biotechnology": 10, "Media": 10,
    "Construction": 10, "Government": 10,
    "Education": 5, "Agriculture": 5,
}

LEAD_SOURCE_POINTS = {  # out of 10
    "Referral": 10, "Partner Referral": 10, "Website Contact Form": 9,
    "Webinar": 8, "Trade Show": 8, "LinkedIn": 7, "Organic Search": 7,
    "Email Campaign": 6, "Google Ads": 6, "Cold Outreach": 3,
}

PRIMARY_MARKETS = ("india", "singapore", "uae", "uk", "usa")   # 5 points
SECONDARY_MARKETS = ("canada", "australia")                    # 4 points

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?[0-9\-\s()]{7,20}$")

CLASS_HOT_MIN = 80    # top ~25% of the sample dataset
CLASS_WARM_MIN = 65   # 65-79 Warm; below 65 Cold

# Gate: leads with no buying intent or authority cannot be Hot or Warm,
# whatever their other factors score.
GATE_MAX_CLASS_IF = ("requirement", "job_title")  # factors that can trigger the gate
GATE_POINTS = 2                                    # points at or below this trigger the gate


@dataclass
class Factor:
    key: str
    label: str
    points: int
    max_points: int
    reason: str
    missing: bool = False

    @property
    def ratio(self) -> float:
        return self.points / self.max_points


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\ufeff", "").strip()
    return "" if text.lower() in {"", "nan", "none", "null", "n/a"} else text


_CANON = {f.lower(): f for f in ALL_FIELDS}


def normalise_record(raw: dict) -> dict:
    """Map header variants (case, spaces, BOM) to canonical field names."""
    out = {f: "" for f in ALL_FIELDS}
    for key, value in raw.items():
        if key is None:
            continue
        canon = _CANON.get(str(key).replace("\ufeff", "").strip().lower())
        if canon:
            out[canon] = _clean(value)
    return out


def _is_valid(field: str, value: str) -> bool:
    if not value:
        return False
    if field == "Email":
        return bool(EMAIL_RE.match(value))
    if field == "Phone":
        return bool(PHONE_RE.match(value))
    return True


# ---------------------------------------------------------------------------
# Factor scorers. Each returns a Factor; missing input gets a neutral score.
# ---------------------------------------------------------------------------
def _title_factor(title: str) -> Factor:
    mx = WEIGHTS["job_title"]
    if not title:
        return Factor("job_title", "Job title unknown", MISSING_POINTS["job_title"], mx,
                      "Job title missing; neutral score applied.", missing=True)
    t = title.lower()
    for pattern, pts, label in TITLE_RULES:
        if re.search(pattern, t):
            return Factor("job_title", label, pts, mx, f"'{title}' is a {label.lower()}.")
    return Factor("job_title", "Unrecognised role", 10, mx,
                  f"'{title}' did not match a title rule; default applied.")


def _size_factor(size: str) -> Factor:
    mx = WEIGHTS["company_size"]
    if not size:
        return Factor("company_size", "Company size unknown", MISSING_POINTS["company_size"], mx,
                      "Company size missing; neutral score applied.", missing=True)
    pts = COMPANY_SIZE_POINTS.get(size)
    if pts is None:
        return Factor("company_size", f"Company size {size}", MISSING_POINTS["company_size"], mx,
                      "Unrecognised size format; neutral score applied.")
    return Factor("company_size", f"Company size {size}", pts, mx, f"{size} band.")


def _industry_factor(industry: str) -> Factor:
    mx = WEIGHTS["industry"]
    if not industry:
        return Factor("industry", "Industry unknown", MISSING_POINTS["industry"], mx,
                      "Industry missing; neutral score applied.", missing=True)
    pts = INDUSTRY_TIERS.get(industry)
    if pts is None:
        return Factor("industry", f"{industry} (unclassified)", 6, mx,
                      "Industry not in fit list; neutral score applied.")
    tier = {15: "high fit", 10: "medium fit", 5: "low fit"}[pts]
    return Factor("industry", f"{industry} ({tier})", pts, mx, f"{industry} is a {tier} industry.")


def _requirement_factor(requirement: str) -> Factor:
    mx = WEIGHTS["requirement"]
    if not requirement:
        return Factor("requirement", "Requirement unknown", MISSING_POINTS["requirement"], mx,
                      "Requirement missing; neutral score applied.", missing=True)
    text = requirement.lower()
    for pattern, pts, label in REQUIREMENT_RULES:
        if re.search(pattern, text):
            return Factor("requirement", label, pts, mx, f"Requirement indicates: {label.lower()}.")
    return Factor("requirement", "Unclassified requirement", 8, mx,
                  "Requirement did not match an intent rule; default applied.")


def _source_factor(source: str) -> Factor:
    mx = WEIGHTS["lead_source"]
    if not source:
        return Factor("lead_source", "Lead source unknown", MISSING_POINTS["lead_source"], mx,
                      "Lead source missing; neutral score applied.", missing=True)
    pts = LEAD_SOURCE_POINTS.get(source, 4)
    return Factor("lead_source", f"Lead source: {source}", pts, mx, f"Source '{source}' scored by channel quality.")


def _location_factor(location: str) -> Factor:
    mx = WEIGHTS["location"]
    if not location:
        return Factor("location", "Location unknown", MISSING_POINTS["location"], mx,
                      "Location missing; neutral score applied.", missing=True)
    low = location.lower()
    if any(m in low for m in PRIMARY_MARKETS):
        return Factor("location", f"Primary target market: {location}", 5, mx, "Primary target market.")
    if any(m in low for m in SECONDARY_MARKETS):
        return Factor("location", f"Secondary market: {location}", 4, mx, "Secondary market.")
    return Factor("location", f"Other market: {location}", 3, mx, "Market outside target list.")


def _completeness_factor(lead: dict) -> tuple[Factor, list[str], list[str]]:
    mx = WEIGHTS["data_completeness"]
    missing, warnings, present = [], [], 0
    for field in COMPLETENESS_FIELDS:
        value = lead.get(field, "")
        if _is_valid(field, value):
            present += 1
        elif not value:
            missing.append(field)
        else:
            warnings.append(f"{field} '{value}' has an invalid format")
            missing.append(field)
    points = round(mx * present / len(COMPLETENESS_FIELDS))
    factor = Factor("data_completeness", f"Data completeness {present}/{len(COMPLETENESS_FIELDS)}",
                    points, mx, f"{present} of {len(COMPLETENESS_FIELDS)} key fields are present and valid.")
    return factor, missing, warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def score_lead(raw: dict) -> dict:
    lead = normalise_record(raw)
    completeness, missing, warnings = _completeness_factor(lead)
    factors = [
        _title_factor(lead["Job Title"]),
        _size_factor(lead["Company Size"]),
        _industry_factor(lead["Industry"]),
        _requirement_factor(lead["Requirement"]),
        _source_factor(lead["Lead Source"]),
        completeness,
        _location_factor(lead["Company Location"]),
    ]
    score = sum(f.points for f in factors)
    classification = "Hot" if score >= CLASS_HOT_MIN else "Warm" if score >= CLASS_WARM_MIN else "Cold"
    gated = [f for f in factors if f.key in GATE_MAX_CLASS_IF and f.points <= GATE_POINTS]
    if gated and classification != "Cold":
        classification = "Cold"

    ratio = (len(COMPLETENESS_FIELDS) - len(missing)) / len(COMPLETENESS_FIELDS)
    level = "High" if ratio >= 0.9 else "Medium" if ratio >= 0.7 else "Low"
    if level == "High" and KEY_FIELDS & set(missing):
        level = "Medium"
    conf_reason = (f"{round(ratio * 100)}% of key fields are present"
                   + (f"; missing key fields {sorted(KEY_FIELDS & set(missing))} cap confidence"
                      if KEY_FIELDS & set(missing) else "")
                   + ".")

    positives = [f.label for f in factors if f.ratio >= 0.7]
    negatives = [f.label for f in factors if f.ratio <= 0.4]

    result = {
        "name": lead["Name"],
        "company": lead["Company Name"],
        "score": score,
        "classification": classification,
        "confidence": {"level": level, "reason": conf_reason},
        "factors": [
            {"key": f.key, "label": f.label, "points": f.points, "max_points": f.max_points,
             "reason": f.reason, "missing": f.missing}
            for f in factors
        ],
        "positive_factors": positives,
        "negative_factors": negatives,
        "missing_fields": missing,
        "data_warnings": warnings,
        "llm_used": False,
    }
    result["gate_applied"] = bool(gated)
    result.update(_rule_based_text(result, factors))
    return result


def _rule_based_text(result: dict, factors: list[Factor]) -> dict:
    cls = result["classification"]
    quality = {
        "Hot": "High-quality lead: a senior, budget-relevant buyer with a clear need or buying signal.",
        "Warm": "Moderate-quality lead: some fit signals, but intent or authority needs confirmation.",
        "Cold": "Low-quality lead: weak fit or intent; not a priority for sales follow-up.",
    }[cls]

    pos = ", ".join(result["positive_factors"][:3]) or "none"
    neg = ", ".join(result["negative_factors"][:3]) or "none"
    explanation = (f"Score {result['score']}/100 ({cls}). Strongest signals: {pos}. "
                   f"Gaps: {neg}. Confidence {result['confidence']['level']}: "
                   f"{result['confidence']['reason']}")

    req = next(f for f in factors if f.key == "requirement")
    if req.points <= 2:
        action = "Route to self-serve or free resources; no sales follow-up."
    elif cls == "Hot":
        action = "Book a discovery or demo call within 48 hours and send a pricing proposal."
    elif cls == "Warm":
        action = ("Send a tailored proposal and confirm the decision timeline within a week."
                  if req.points >= 15 else
                  "Share a relevant case study and qualify budget, authority and timeline in 3-5 days.")
    else:
        action = "Add to a nurture sequence and re-score when new engagement arrives."
    if result["missing_fields"]:
        contact = [f for f in result["missing_fields"] if f in ("Email", "Phone")]
        if contact:
            action += f" Collect missing contact details first ({', '.join(contact)})."

    return {
        "quality_assessment": quality,
        "scoring_explanation": explanation,
        "recommended_next_action": action,
    }


FLAT_FIELDS = ["Name", "Company", "Score", "Classification", "Confidence",
               "Quality Assessment", "Positive Factors", "Negative Factors",
               "Scoring Explanation", "Recommended Next Action",
               "Missing Fields", "Data Warnings"]


def to_flat_row(r: dict) -> dict:
    return {
        "Name": r["name"], "Company": r["company"], "Score": r["score"],
        "Classification": r["classification"], "Confidence": r["confidence"]["level"],
        "Quality Assessment": r["quality_assessment"],
        "Positive Factors": "; ".join(r["positive_factors"]),
        "Negative Factors": "; ".join(r["negative_factors"]),
        "Scoring Explanation": r["scoring_explanation"],
        "Recommended Next Action": r["recommended_next_action"],
        "Missing Fields": "; ".join(r["missing_fields"]),
        "Data Warnings": "; ".join(r["data_warnings"]),
    }
