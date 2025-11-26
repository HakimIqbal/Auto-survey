# Campus Food-Ordering Survey Generator

Python script that simulates responses to 25 MSU campus food-ordering questions, creates Google Form-like charts, and compiles an HTML dashboard. All narrative answers (free text, Q5 enrichment, executive summary) can be sourced from Groq Cloud for realistic, non-dummy prose.

## Features
- 50+ respondents (configurable with `--n`) with human-like randomness and unique answer sets.
- Auto-select question types (single, Likert, multi-select) and generates pie or clustered bar charts per question.
- Optional Groq Cloud integration for natural text answers and a professional executive summary card in the HTML report.
- Outputs:
  - `output/responses.csv`
  - `output/charts/Q*.png`
  - `output/report.html`

## Requirements
Install dependencies once:

```bash
python -m pip install -r requirements.txt
```

Set your Groq API key (preferred) in `.env`:

```bash
cp .env.example .env
echo "GROQ_API_KEY=sk-your-key" >> .env
```

## Usage
Generate the full dataset (Groq required by default):

```bash
python generate_google_form_like_summary.py --n 50
```

- `--seed 42` for reproducible randomness.
- `--allow-faker` if you need to bypass Groq (not recommended).

### Output
- CSV with respondent answers.
- 25 chart images mirroring Google Forms style.
- `report.html` with charts, tables, quotes, and an executive summary written by Groq.

## Notes
- The script retries Groq requests when encountering rate limits (HTTP 429). If limits persist, run again later.
- For custom question banks or styling tweaks, edit `generate_google_form_like_summary.py`.
