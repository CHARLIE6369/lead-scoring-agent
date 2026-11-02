# AI-Based Lead Scoring System

A FastAPI application that scores sales leads (0–100), classifies them Hot / Warm / Cold,
explains the score, rates confidence and recommends a next action.

The score is produced by a **deterministic, weighted rule engine**. An LLM (Groq, optional)
is used only to write the quality assessment, explanation and next action from the facts the
rules computed. It never generates or changes the number, so identical inputs always give
identical scores.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional: add GROQ_API_KEY
```

Load the key into your shell before starting, or export it directly:
`export GROQ_API_KEY=...` (never commit a real key).

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the UI. API docs are at http://127.0.0.1:8000/docs.

## Score the sample dataset

```bash
python scripts/score_dataset.py                       # rules only
python scripts/score_dataset.py data/lead_scoring_sample_dataset_500.csv outputs/scored_leads.csv --llm
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Web UI (single lead + CSV upload) |
| GET | `/api/health` | Status and whether the LLM key is set |
| POST | `/api/score?use_llm=true` | Score one lead (JSON body with the 11 input fields) |
| POST | `/api/score-csv?use_llm=false&output=json` | Score a CSV, returns summary and results |
| POST | `/api/score-csv?output=csv` | Score a CSV, returns a scored CSV download |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/score?use_llm=false \
  -H "Content-Type: application/json" \
  -d '{"Name":"Rahul Sharma","Company Name":"ABC Technologies","Job Title":"Founder",
       "Industry":"IT Services","Company Location":"Bangalore","Company Size":"51-200 employees",
       "Lead Source":"Website","Requirement":"Looking for CRM software",
       "Email":"rahul@example.com","Phone":"+91 9876543210","Website":"https://www.abctechnologies.com"}'
```

## Scoring methodology

Total 100 points across seven factors:

| Factor | Points | How it is scored |
|---|---|---|
| Job title / seniority | 25 | C-level or founder 25, VP 22, Director 20, Head of 18, Manager 14, Consultant/Analyst/Executive 10, Engineer 8, Student/Intern 2 |
| Requirement intent | 20 | Explicit buying signal (budget approved, PoC, pricing proposal, decision within 30 days, demo) 20; vendor replacement or enterprise-wide 18; clear pain point (manual process, automation, integration, segmentation) 15; wants to understand features 10; early research 6; free tool or personal project 2 |
| Company size | 15 | 5000+ 15, 1001–5000 14, 501–1000 12, 201–500 11, 51–200 9, 11–50 6, 1–10 3 |
| Industry fit | 15 | High fit 15 (Technology, IT Services, Finance, Cybersecurity, Analytics, Healthcare, Retail); medium 10 (Consulting, Logistics, Manufacturing, Automotive, Energy, Real Estate, Biotechnology, Media, Construction, Government); low 5 (Education, Agriculture) |
| Lead source | 10 | Referral / Partner Referral 10, Website Contact Form 9, Webinar / Trade Show 8, LinkedIn / Organic Search 7, Email Campaign / Google Ads 6, Cold Outreach 3 |
| Data completeness | 10 | Proportional to 9 key fields being present and valid (email and phone format checked) |
| Location | 5 | Primary markets (India, Singapore, UAE, UK, USA) 5; Canada / Australia 4; other 3 |

**Classification:** Hot ≥ 80, Warm 65–79, Cold < 65. These cutoffs were set against the score distribution of the
500-row sample (median 74, top quartile 80+). With the usual 70/40 cutoffs, 65% of that sample would be Hot.

**Gate rule:** a lead whose requirement is a free tool or personal project, or whose job title is student or intern,
is classified Cold regardless of its score, because it has no buying intent or authority.

**Missing data:** each factor has a neutral default (e.g. missing job title = 8/25, missing requirement = 5/20)
and the missing field is listed. Confidence is High when ≥ 90% of key fields are present, Medium at ≥ 70%,
otherwise Low. A missing Job Title or Requirement caps confidence at Medium.

**Positive / negative factors:** a factor is positive when it earns ≥ 70% of its points, negative when ≤ 40%.

All weights and rules live at the top of `app/scoring.py`, so they can be tuned without touching the logic.

## Handling the sample dataset

The supplied CSV (500 rows) has a UTF-8 BOM on the header, which the loader handles. It also contains blanks
in several columns and invalid or placeholder contact values, which are scored neutrally and flagged rather
than dropped. Company Size is stored as ranges (e.g. `51-200 employees`) and Requirement as free text, which the
intent rules classify by keyword.

## Tests

```bash
pytest -q
```

Tests cover: weights sum to 100, deterministic output, Hot and Cold classification, missing and invalid input,
header normalisation, the free-tool routing rule, and an end-to-end run over the 500-row dataset.

## Design notes

- **Why rules for the score:** the brief requires consistent, explainable scores. A rule engine gives the same
  input the same score, and every point can be traced to a factor.
- **Why the LLM is optional:** batch scoring of 500 rows with an LLM is slow and rate-limited, and the score does
  not depend on it. With no key set, the app returns rule-based text.
- **Limitations:** weights are a judgement call, not fitted to conversion data. The keyword intent rules are simple
  and would need extending for free-text requirements that use different wording. Industry and market tiers are
  assumptions to be adjusted to the sales team's ICP.
