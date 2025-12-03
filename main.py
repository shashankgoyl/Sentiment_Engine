# Project: Customer Insight API (FastAPI)
# File: main.py
# A simple, easy-to-run implementation that:
# 1) Accepts a batch of text reviews via POST JSON {"reviews": [...]}.
# 2) Returns overall sentiment, top 3 key themes, and a one-sentence actionable feedback.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import math

# NLP libraries
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = FastAPI(title="Customer Insight API")

# Pydantic model for request
class ReviewsInput(BaseModel):
    reviews: List[str]

# Helper: simple text clean
def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Lazy-loaded transformers pipelines
_sentiment_pipeline = None
_text2text_pipeline = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        if not HAS_TRANSFORMERS:
            return None
        # fine-tuned sentiment model (small and fast)
        _sentiment_pipeline = pipeline("sentiment-analysis")
    return _sentiment_pipeline

def get_text2text_pipeline():
    global _text2text_pipeline
    if _text2text_pipeline is None:
        if not HAS_TRANSFORMERS:
            return None
        # t5 or flan-t5 small can be used - pipeline will pick a default if available
        _text2text_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
    return _text2text_pipeline

# Compute overall sentiment score and label
# We'll map POSITIVE->+1 NEGATIVE->-1 and average across reviews

def compute_overall_sentiment(reviews: List[str]) -> Dict[str, Any]:
    clean = [r for r in reviews if r and r.strip()]
    if not clean:
        return {"label": "Neutral", "score": 0.0}

    pipe = get_sentiment_pipeline()
    scores = []
    if pipe is not None:
        results = pipe(clean)
        for res in results:
            label = res.get("label", "NEUTRAL")
            # Using score*direction
            score = res.get("score", 0.0)
            if label.upper().startswith("POS"):
                scores.append(score)
            else:
                scores.append(-score)
    else:
        # fallback: rule-based tiny sentiment using keywords
        pos_kw = ["love","great","good","awesome","excellent","amazing","happy"]
        neg_kw = ["terrible","bad","slow","hate","awful","worst","poor"]
        for r in clean:
            s = 0.0
            low = r.lower()
            for kw in pos_kw:
                if kw in low:
                    s += 0.6
            for kw in neg_kw:
                if kw in low:
                    s -= 0.6
            scores.append(max(-1.0, min(1.0, s)))

    # average and normalize to [-1,1]
    if not scores:
        avg = 0.0
    else:
        avg = sum(scores) / len(scores)
        avg = max(-1.0, min(1.0, avg))

    # map to label
    if avg >= 0.25:
        label = "Positive"
    elif avg <= -0.25:
        label = "Negative"
    else:
        label = "Neutral"

    # Also return a confidence-like absolute value
    return {"label": label, "score": round(avg, 3)}

# Extract top N themes using TF-IDF over the whole corpus
# This returns short phrases/words that are most characteristic.

def extract_key_themes(reviews: List[str], top_n: int = 3) -> List[str]:
    clean_reviews = [clean_text(r) for r in reviews if r and r.strip()]
    if not clean_reviews:
        return []

    # Use unigrams + bigrams so we can pick simple phrases
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=2000)
    try:
        X = vectorizer.fit_transform(clean_reviews)
    except ValueError:
        return []
    feature_names = vectorizer.get_feature_names_out()

    # average tf-idf across documents
    import numpy as np
    avg_tfidf = X.mean(axis=0).A1

    # get top indices
    idxs = avg_tfidf.argsort()[::-1]

    themes = []
    for i in idxs:
        candidate = feature_names[i]
        # small filters to avoid nonsense
        if len(candidate) < 3:
            continue
        if candidate.isdigit():
            continue
        themes.append(candidate)
        if len(themes) >= top_n:
            break
    return themes

# Produce a single-sentence actionable feedback using a generation model if available

def generate_actionable_feedback(reviews: List[str]) -> str:
    pipe = get_text2text_pipeline()
    joined = " \n ".join([r.strip() for r in reviews if r and r.strip()])
    if not joined:
        return "No reviews provided."

    prompt = (
        "You are a product-focused assistant. Given these customer reviews, provide ONE concise sentence that summarizes what specifically needs improvement. "
        "Be direct and actionable.\n\nReviews:\n" + joined + "\n\nActionable sentence:"
    )

    if pipe is not None:
        try:
            # keep max_length small for one sentence
            out = pipe(prompt, max_length=64, do_sample=False)
            text = out[0]['generated_text'] if isinstance(out, list) else str(out)
            # clean up
            text = text.strip().replace('\n',' ')
            # if model echoes prompt, truncate after colon
            if 'Actionable sentence:' in text:
                text = text.split('Actionable sentence:')[-1].strip()
            # take first sentence
            m = re.split(r'[\.\!\?]\s+', text)
            if m:
                return m[0].strip() + ('.' if not m[0].strip().endswith('.') else '')
            return text
        except Exception:
            pass

    # fallback heuristic: look for keywords and craft a sentence
    low = joined.lower()
    if 'support' in low or 'service' in low:
        return 'Improve customer support responsiveness and ticket resolution time.'
    if 'ui' in low or 'interface' in low or 'slow' in low:
        return 'Optimize the user interface performance to reduce lag and improve responsiveness.'
    if 'price' in low or 'cost' in low:
        return 'Re-evaluate pricing or clearly communicate value to reduce price-related complaints.'
    # generic fallback
    return 'Address the most common complaints mentioned above (e.g., support, performance, or usability) to improve customer satisfaction.'

@app.post('/analyze')
async def analyze_reviews(payload: ReviewsInput):
    reviews = payload.reviews
    if not isinstance(reviews, list):
        raise HTTPException(status_code=400, detail='reviews must be a list of strings')

    overall = compute_overall_sentiment(reviews)
    themes = extract_key_themes(reviews, top_n=3)
    action = generate_actionable_feedback(reviews)

    response = {
        'overall_sentiment': overall,
        'key_themes': themes,
        'actionable_feedback': action
    }
    return response