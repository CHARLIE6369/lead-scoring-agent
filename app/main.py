from __future__ import annotations

import csv
import io
import pathlib
from collections import Counter

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from app.llm import enrich_with_llm, llm_enabled
from app.scoring import FLAT_FIELDS, normalise_record, score_lead, to_flat_row

BASE_DIR = pathlib.Path(__file__).resolve().parent

app = FastAPI(title="AI-Based Lead Scoring System", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm_available": llm_enabled()}


@app.post("/api/score")
def score_one(lead: dict, use_llm: bool = Query(True)) -> dict:
    result = score_lead(lead)
    return enrich_with_llm(result) if use_llm else result


def _summary(results: list[dict]) -> dict:
    counts = Counter(r["classification"] for r in results)
    total = len(results)
    return {
        "total": total,
        "hot": counts.get("Hot", 0),
        "warm": counts.get("Warm", 0),
        "cold": counts.get("Cold", 0),
        "average_score": round(sum(r["score"] for r in results) / total, 1) if total else 0,
        "with_missing_fields": sum(1 for r in results if r["missing_fields"]),
    }


@app.post("/api/score-csv")
async def score_csv(
    file: UploadFile = File(...),
    use_llm: bool = Query(False, description="LLM text per row; slower and rate-limited for large files"),
    output: str = Query("json", pattern="^(json|csv)$"),
):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    text = (await file.read()).decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise HTTPException(status_code=400, detail="The CSV has no data rows.")

    results = []
    for raw in rows:
        r = score_lead(raw)
        results.append(enrich_with_llm(r) if use_llm else r)

    if output == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=FLAT_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(to_flat_row(r))
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=scored_leads.csv"},
        )
    return {"summary": _summary(results), "results": results}
