#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

import matplotlib
import numpy as np
import pandas as pd
import requests
from faker import Faker
from jinja2 import Template

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

QuestionType = Literal["likert5", "single", "multi", "numeric", "short_text", "long_text"]


class Question(TypedDict, total=False):
    id: int
    text: str
    type: QuestionType
    options: Optional[List[str]]
    weights: Optional[List[float]]
    multi_min: Optional[int]
    multi_max: Optional[int]
    range: Optional[Tuple[float, float]]


LIKERT = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]


def build_questions() -> List[Question]:
    data = [
        (1, "How often do you order food on campus during weekdays?", "single", "Never|Rarely|Sometimes|Often|Very often", "0.05,0.15,0.35,0.3,0.15", None, None),
        (2, "How satisfied are you with the current food-ordering process at MSU?", "likert5", "Very dissatisfied|Dissatisfied|Neutral|Satisfied|Very satisfied", "0.05,0.1,0.35,0.35,0.15", None, None),
        (3, "What are the main challenges you face when ordering food during peak hours?", "multi", "Long queues|App availability|Menu clarity|Payment issues|Delivery delays|Other", "0.3,0.15,0.15,0.1,0.2,0.1", 1, 3),
        (4, "How long do you usually wait for your food during lunch hours?", "single", "<10 min|10–20 min|20–30 min|30–45 min|>45 min", "0.12,0.28,0.3,0.2,0.1", None, None),
        (5, "Have you used online food delivery apps like GrabFood or Foodpanda before? If yes, how was your experience?", "single", "Never tried|Tried once, not great|Use occasionally, mixed|Use weekly, mostly good|Use often, love it", "0.1,0.18,0.27,0.25,0.2", None, None),
        (6, "Would you be interested in using a campus-based food-ordering and delivery app?", "single", "No|Maybe|Yes", "0.1,0.25,0.65", None, None),
        (7, "Which features do you consider most important for such an app (e.g., menu browsing, order tracking, secure payment)?", "multi", "Menu browsing|Order customization|Real-time tracking|Secure payments|Promotions/loyalty|Group orders", "0.2,0.15,0.2,0.2,0.15,0.1", 2, 4),
        (8, "What are your expectations for delivery speed and accuracy?", "likert5", "Very low expectations|Low expectations|Neutral|High expectations|Very high expectations", "0.05,0.1,0.2,0.4,0.25", None, None),
        (9, "How likely are you to recommend this kind of service to your friends on campus?", "likert5", "", "0.05,0.1,0.25,0.35,0.25", None, None),
        (10, "Would you prefer to pay digitally (e-wallets, FPX, debit/credit) or by cash?", "single", "Digital|Cash|Both", "0.55,0.15,0.3", None, None),
        (11, "Would you be interested in becoming a student delivery partner for extra income?", "single", "No|Maybe|Yes", "0.45,0.35,0.2", None, None),
        (12, "What factors would motivate you to become a delivery rider (e.g., flexible schedule, earnings, campus convenience)?", "multi", "Flexible hours|Earnings|Campus familiarity|Learning experience|Social aspect", "0.25,0.35,0.15,0.15,0.1", 1, 3),
        (13, "How important is GPS-based tracking and route optimization for delivery efficiency?", "likert5", "", "0.02,0.08,0.18,0.42,0.3", None, None),
        (14, "Do you think the app should have an automatic rider assignment feature based on rider location?", "single", "No|Not sure|Yes", "0.1,0.25,0.65", None, None),
        (15, "How do you feel about having a transparent order assignment system for fairness?", "likert5", "", "0.03,0.08,0.25,0.4,0.24", None, None),
        (16, "How important is it for food vendors to have a real-time dashboard for managing orders?", "likert5", "", "0.02,0.08,0.22,0.4,0.28", None, None),
        (17, "Would live menu updates (e.g., item availability and price changes) improve your ordering experience?", "likert5", "", "0.02,0.08,0.2,0.42,0.28", None, None),
        (18, "How useful would AI-based food recommendations be in helping you choose meals quickly?", "likert5", "", "0.05,0.15,0.3,0.32,0.18", None, None),
        (19, "How important is it for vendors to be able to send notifications about order status (e.g., ready, dispatched, delivered)?", "likert5", "", "0.02,0.05,0.18,0.45,0.3", None, None),
        (20, "Would you be more likely to order if vendors offered loyalty points, discounts, or special promotions?", "likert5", "", "0.03,0.07,0.2,0.4,0.3", None, None),
        (21, "How concerned are you about payment security when using a digital campus delivery app?", "likert5", "", "0.04,0.12,0.3,0.32,0.22", None, None),
        (22, "Do you feel comfortable sharing your location for order tracking purposes?", "likert5", "", "0.08,0.18,0.3,0.3,0.14", None, None),
        (23, "Would you trust other students to handle food delivery professionally and reliably?", "likert5", "", "0.1,0.18,0.32,0.28,0.12", None, None),
        (24, "How important is a feedback and rating system for maintaining good service quality?", "likert5", "", "0.02,0.05,0.15,0.45,0.33", None, None),
        (25, "What additional features or improvements would you like to see in the app to make it more successful and widely accepted within the MSU community?", "long_text", "", "", None, None),
    ]
    questions: List[Question] = []
    for idx, text, qtype, opt, wts, mmin, mmax in data:
        questions.append(
            Question(
                id=idx,
                text=text,
                type=qtype,  # type: ignore[arg-type]
                options=[p.strip() for p in opt.split("|") if p.strip()] if opt else None,
                weights=[float(x) for x in wts.split(",")] if wts else None,
                multi_min=mmin,
                multi_max=mmax,
                range=None,
            )
        )
    return questions


QUESTION_BANK = build_questions()
ENRICH_SINGLE_WITH_TEXT = {5}
TEXT_KEYWORDS: Dict[int, List[Tuple[str, Tuple[str, ...]]]] = {
    25: [
        ("Faster delivery & tracking", ("fast", "speed", "tracking", "delay")),
        ("More promos & loyalty", ("promo", "discount", "loyal", "points")),
        ("Vendor variety & menus", ("vendor", "menu", "variety", "options")),
        ("Better communication & support", ("support", "chat", "notify", "update")),
        ("Payment & security", ("payment", "secure", "security", "fpx")),
    ]
}
REPORT_TEMPLATE = Template(
    """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Survey Summary</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;background:#f5f5f5;margin:0;padding:24px;}
h1{text-align:center;color:#202124;}
.meta{text-align:center;color:#5f6368;margin-bottom:24px;}
.summary-card{max-width:900px;margin:0 auto 24px auto;background:#fff;border-radius:14px;padding:18px;box-shadow:0 1px 4px rgba(60,64,67,.3);}
.summary-card h2{margin-top:0;color:#202124;}
.summary-card p{line-height:1.5;color:#3c4043;white-space:pre-line;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;}
.card{background:#fff;border-radius:14px;padding:18px;box-shadow:0 1px 4px rgba(60,64,67,.3);}
table{width:100%;border-collapse:collapse;font-size:.9rem;margin-top:8px;}
th,td{padding:4px;border-bottom:1px solid #e0e0e0;text-align:left;}
img{width:100%;border-radius:10px;margin-top:12px;}
.quotes{font-size:.85rem;color:#555;margin-top:10px;}
.quotes p{margin:4px 0;}
</style></head>
<body>
<h1>Campus Food Ordering Survey</h1>
<div class="meta">{{respondent_count}} respondents · Generated {{generated_at}}</div>
{% if overview %}
<div class="summary-card">
  <h2>Executive Summary</h2>
  <p>{{ overview }}</p>
</div>
{% endif %}
<div class="grid">
{% for q in summaries -%}
  <div class="card">
    <h2>Q{{q.id}}. {{q.text}}</h2>
    <table>
      <thead><tr><th>Option</th><th>Count</th><th>%</th></tr></thead>
      <tbody>
      {% for row in q.table %}
        <tr><td>{{row.option}}</td><td>{{row.count}}</td><td>{{row.percent}}</td></tr>
      {% endfor %}
      </tbody>
    </table>
    <img src="{{q.chart}}" alt="Chart Q{{q.id}}">
    {% if q.quotes %}
    <div class="quotes">
      {% for quote in q.quotes %}<p>&ldquo;{{quote}}&rdquo;</p>{% endfor %}
    </div>
    {% endif %}
  </div>
{% endfor %}
</div>
</body></html>"""
)


def normalized(weights: Optional[Sequence[float]], length: int) -> Optional[np.ndarray]:
    if not weights:
        return None
    arr = np.array(weights, dtype=float)
    if arr.size != length:
        raise ValueError("weights mismatch")
    total = arr.sum()
    return arr / total if total else None


def groq_text_answer(
    question: str,
    context: Dict[str, object],
    api_key: str,
    timeout: float = 12.0,
    system_prompt: str = "You generate short, varied, natural survey answers from Malaysian university students. Max 2 sentences.",
    max_tokens: int = 64,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\nContext: {json.dumps(context)}\nWrite a concise, realistic answer."},
        ],
        "temperature": 0.9,
        "max_tokens": max_tokens,
    }
    backoff = 1.5
    attempts = 4
    for attempt in range(attempts):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429 and attempt < attempts - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError("Failed to obtain response from Groq after retries.")


class TextGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        faker: Faker,
        api_key: Optional[str],
        use_groq: bool,
        require_groq: bool,
    ):
        self.rng = rng
        self.faker = faker
        self.api_key = api_key if (api_key and use_groq) else None
        self.use_groq = bool(self.api_key)
        self.require_groq = require_groq
        if self.require_groq and not self.use_groq:
            raise RuntimeError(
                "GROQ_API_KEY must be set when --use-groq is provided."
            )

    def _fallback_sentence(self, min_words: int, max_words: int) -> str:
        words = int(self.rng.integers(min_words, max_words + 1))
        return self.faker.sentence(nb_words=words)

    def _groq_answer(self, question: str, context: Dict[str, object]) -> Optional[str]:
        if not self.use_groq:
            return None
        try:
            return groq_text_answer(question, context, self.api_key)  # type: ignore[arg-type]
        except Exception as exc:
            if self.require_groq:
                raise RuntimeError("Groq text generation failed.") from exc
            return None

    def short(self, question: str, context: Dict[str, object]) -> str:
        context = dict(context)
        context.setdefault("style", "short")
        result = self._groq_answer(question, context)
        if result:
            return result
        base = self._fallback_sentence(10, 22)
        extra = self._fallback_sentence(8, 16)
        return base if len(base.split()) > 8 else f"{base} {extra}"

    def long(self, question: str, context: Dict[str, object]) -> str:
        context = dict(context)
        context.setdefault("style", "long")
        result = self._groq_answer(question, context)
        if result:
            return result
        parts = [
            self._fallback_sentence(12, 22),
            self._fallback_sentence(10, 20),
        ]
        return " ".join(parts)


LONG_TEXT_CONTEXT_KEYS = [1, 3, 5, 7, 8, 17, 20, 21, 22, 23, 24]


class Sampler:
    def __init__(self, rng: np.random.Generator, text_gen: TextGenerator):
        self.rng = rng
        self.text_gen = text_gen

    def _long_context(self, row_context: Optional[Dict[str, object]]) -> Dict[str, object]:
        if not row_context:
            return {}
        highlights = {}
        for qid in LONG_TEXT_CONTEXT_KEYS:
            key = f"Q{qid}"
            if key in row_context:
                highlights[key] = row_context[key]
        return {"highlights": highlights} if highlights else {}

    def sample(self, question: Question, row_context: Optional[Dict[str, object]] = None) -> str:
        qtype = question["type"]
        if qtype == "single":
            options = question["options"] or []
            probs = normalized(question.get("weights"), len(options))
            choice = options[int(self.rng.choice(len(options), p=probs))]
            if question["id"] in ENRICH_SINGLE_WITH_TEXT:
                detail = self.text_gen.short(question["text"], {"choice": choice})
                return f"{choice} | {detail}"
            return choice
        if qtype == "multi":
            options = question["options"] or []
            probs = normalized(question.get("weights"), len(options))
            low = question.get("multi_min") or 1
            high = question.get("multi_max") or len(options)
            size = int(self.rng.integers(low, high + 1))
            picks = self.rng.choice(len(options), size=size, replace=False, p=probs)
            return "; ".join(sorted((options[i] for i in picks), key=options.index))
        if qtype == "likert5":
            options = question["options"] or LIKERT
            probs = normalized(question.get("weights"), len(options))
            return options[int(self.rng.choice(len(options), p=probs))]
        if qtype == "numeric":
            low, high = question.get("range", (0.0, 1.0))
            mode = (low + high) / 2
            return f"{self.rng.triangular(low, mode, high):.2f}"
        if qtype == "short_text":
            return self.text_gen.short(question["text"], {"question_id": question["id"]})
        if qtype == "long_text":
            base_context = {"question_id": question["id"]}
            base_context.update(self._long_context(row_context))
            return self.text_gen.long(question["text"], base_context)
        raise ValueError(qtype)


def summarize(series: pd.Series, question: Question, total: int) -> Tuple[pd.Series, pd.Series]:
    qtype = question["type"]
    if qtype == "multi":
        expanded: List[str] = []
        for raw in series.fillna(""):
            expanded.extend([part.strip() for part in raw.split(";") if part.strip()])
        counts = pd.Series(expanded).value_counts() if expanded else pd.Series(dtype=int)
        options = question.get("options") or []
        counts = counts.reindex(options, fill_value=0)
        percents = counts / total * 100
        return counts, percents
    if qtype in {"short_text", "long_text"}:
        def categorize(ans: str) -> str:
            lower = ans.lower()
            for label, keywords in TEXT_KEYWORDS.get(question["id"], []):
                if any(k in lower for k in keywords):
                    return label
            return "Other ideas"
        categories = series.fillna("").apply(categorize)
        counts = categories.value_counts()
        percents = counts / total * 100
        return counts, percents
    values = series.fillna("").apply(lambda val: val.split("|", 1)[0].strip() if " | " in val else val)
    counts = values.value_counts()
    options = question.get("options") or LIKERT
    counts = counts.reindex(options, fill_value=0)
    percents = counts / total * 100
    return counts, percents


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned[:60] or "chart"


class ChartMaker:
    def __init__(self, chart_dir: Path):
        self.chart_dir = chart_dir

    def _path(self, question: Question) -> Path:
        return self.chart_dir / f"Q{question['id']}_{slugify(question['text'])}.png"

    @staticmethod
    def _wrap(text: str, width: int) -> str:
        return textwrap.fill(text, width=width)

    def _title(self, question: Question) -> str:
        return self._wrap(f"Q{question['id']}: {question['text']}", 60)

    def pie(self, question: Question, counts: pd.Series) -> Path:
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        values = counts.values
        labels = counts.index.tolist()

        def autopct(pct: float) -> str:
            total = counts.sum()
            absolute = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n({absolute})"

        fig.suptitle(self._title(question), fontsize=10, y=0.98)
        wedges, texts, autotexts = ax.pie(
            values,
            labels=[self._wrap(label, 24) for label in labels],
            autopct=autopct,
            startangle=90,
            counterclock=False,
        )
        ax.axis("equal")
        for text in texts + autotexts:
            text.set_fontsize(8)
        plt.subplots_adjust(top=0.78, bottom=0.05)
        fig.tight_layout()
        path = self._path(question)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def bar(self, question: Question, counts: pd.Series, percents: pd.Series) -> Path:
        fig, ax = plt.subplots(figsize=(7.8, 5.0))
        x = np.arange(len(counts))
        bars = ax.bar(x, counts.values, width=0.7, color="#4C72B0", edgecolor="#1f3d63")
        ax.set_xticks(x)
        wrapped_labels = [self._wrap(label, 22) for label in counts.index.tolist()]
        ax.set_xticklabels(wrapped_labels, rotation=0, ha="center")
        ax.set_ylabel("Responses")
        fig.suptitle(self._title(question), fontsize=10, y=0.98)
        for bar, pct in zip(bars, percents.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.subplots_adjust(top=0.78, bottom=0.28)
        fig.tight_layout()
        path = self._path(question)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def hist(self, question: Question, series: pd.Series) -> Path:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.hist(series.astype(float), bins=6, color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Value")
        ax.set_ylabel("Responses")
        fig.suptitle(self._title(question), fontsize=10, y=0.98)
        plt.subplots_adjust(top=0.82)
        fig.tight_layout()
        path = self._path(question)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


def select_quotes(series: pd.Series, question: Question, rng: np.random.Generator) -> List[str]:
    if question["type"] != "long_text" and question["id"] not in ENRICH_SINGLE_WITH_TEXT:
        return []
    values = [v for v in series.dropna().tolist() if v]
    if not values:
        return []
    rng.shuffle(values)
    quotes: List[str] = []
    for answer in values[:3]:
        if question["id"] in ENRICH_SINGLE_WITH_TEXT and " | " in answer:
            quotes.append(answer.split("|", 1)[1].strip())
        else:
            quotes.append(answer)
    return quotes


def generate_responses(count: int, sampler: Sampler, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    seen: set = set()
    for rid in range(1, count + 1):
        row: Dict[str, object] = {"respondent_id": rid}
        for q in QUESTION_BANK:
            row[f"Q{q['id']}"] = sampler.sample(q, row)
        key = tuple(row[f"Q{q['id']}"] for q in QUESTION_BANK)
        bump = 0
        while key in seen and bump < 5:
            idx = int(rng.integers(0, len(QUESTION_BANK)))
            q = QUESTION_BANK[idx]
            row[f"Q{q['id']}"] = sampler.sample(q, row)
            key = tuple(row[f"Q{q['id']}"] for q in QUESTION_BANK)
            bump += 1
        if key in seen:
            idx = int(rng.integers(0, len(QUESTION_BANK)))
            q = QUESTION_BANK[idx]
            row[f"Q{q['id']}"] = sampler.sample(q, row)
            key = tuple(row[f"Q{q['id']}"] for q in QUESTION_BANK)
        seen.add(key)
        rows.append(row)
    return pd.DataFrame(rows)


def build_summaries(
    df: pd.DataFrame,
    chart_maker: ChartMaker,
    report_dir: Path,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, object]], List[str]]:
    total = len(df)
    summaries: List[Dict[str, object]] = []
    insights: List[str] = []
    for q in QUESTION_BANK:
        column = f"Q{q['id']}"
        counts, percents = summarize(df[column], q, total)
        qtype = q["type"]
        if qtype in {"single", "likert5"}:
            chart_path = chart_maker.pie(q, counts)
        elif qtype == "numeric":
            chart_path = chart_maker.hist(q, df[column])
        else:
            chart_path = chart_maker.bar(q, counts, percents)
        table = [
            {"option": str(opt), "count": int(counts.loc[opt]), "percent": f"{percents.loc[opt]:.1f}%"}
            for opt in counts.index
        ]
        summaries.append(
            {
                "id": q["id"],
                "text": q["text"],
                "table": table,
                "chart": os.path.relpath(chart_path, start=report_dir),
                "quotes": select_quotes(df[column], q, rng),
            }
        )
        if not counts.empty:
            top_option = counts.idxmax()
            share = percents.loc[top_option] if top_option in percents else 0.0
            insights.append(
                f"Q{q['id']}: {q['text']} — most common response '{top_option}' ({share:.1f}% of respondents)."
            )
    return summaries, insights


def generate_overall_summary(
    insights: List[str],
    respondent_count: int,
    api_key: Optional[str],
    require_groq: bool,
) -> str:
    context = {
        "respondent_count": respondent_count,
        "insights": insights[:12],
    }
    if api_key:
        prompt = (
            "Provide a concise, professional executive summary for decision makers. "
            "Highlight momentum, concerns, and priority recommendations."
        )
        return groq_text_answer(
            prompt,
            context,
            api_key,
            system_prompt="You are a strategy consultant summarizing survey data for university leadership. Be confident, neutral, and actionable.",
            max_tokens=220,
        )
    if require_groq:
        raise RuntimeError("GROQ_API_KEY is required to generate the executive summary.")
    fallback = "Key highlights: " + " ".join(insights[:3])
    return fallback.strip()


def render_report(
    path: Path,
    respondent_count: int,
    summaries: List[Dict[str, object]],
    overview: Optional[str],
) -> None:
    html = REPORT_TEMPLATE.render(
        respondent_count=respondent_count,
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        summaries=summaries,
        overview=overview,
    )
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Google Form-like summary.")
    parser.add_argument("--n", type=int, default=50, help="Number of respondents")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--use-groq",
        dest="use_groq",
        action="store_true",
        help="Require Groq Cloud for all narrative answers (default).",
    )
    parser.add_argument(
        "--allow-faker",
        dest="use_groq",
        action="store_false",
        help="Allow Faker fallback when Groq isn't available.",
    )
    parser.set_defaults(use_groq=True)
    return parser.parse_args()


def main() -> None:
    if load_dotenv:
        load_dotenv()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    try:
        faker = Faker("en_MY")
    except AttributeError:
        faker = Faker("en_US")
    groq_key = os.getenv("GROQ_API_KEY")
    use_groq = bool(groq_key)
    text_gen = TextGenerator(rng, faker, groq_key, use_groq, args.use_groq)
    sampler = Sampler(rng, text_gen)
    df = generate_responses(args.n, sampler, rng)
    output_dir = Path("output")
    chart_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "responses.csv"
    df.to_csv(csv_path, index=False)
    chart_maker = ChartMaker(chart_dir)
    summaries, insights = build_summaries(df, chart_maker, output_dir, rng)
    overview_raw = generate_overall_summary(
        insights,
        args.n,
        groq_key if groq_key else None,
        args.use_groq,
    )
    overview = (
        overview_raw.replace("**", "").replace("* ", "• ")
        if overview_raw
        else None
    )
    report_path = output_dir / "report.html"
    render_report(report_path, args.n, summaries, overview)
    print(f"Saved CSV -> {csv_path}")
    print(f"Saved charts -> {chart_dir}/*.png")
    print(f"Open report -> {report_path}")


if __name__ == "__main__":
    main()
