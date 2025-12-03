Customer Insight API - Quick Setup (PyCharm-friendly)

1) Create project in PyCharm:
   - File -> New Project -> Pure Python -> choose the folder where main.py will live.
   - Create a virtual environment via PyCharm or use the terminal.

2) Add the code (main.py) to the project (this file).

3) Create requirements.txt with the listed packages.

4) Install packages via PyCharm's package manager or the terminal:
   pip install -r requirements.txt

5) Run the server:
   uvicorn main:app --reload --port 8000

6) Use the sample curl command to test.

Notes:
- If you don't want heavy ML dependencies, the code has a fallback heuristic for basic behavior.
- For local development, using CPU-only models is recommended (they are slower but simple). Use small models like 'google/flan-t5-small' to keep memory usage low.


# 1) Create a virtualenv: python -m venv .venv
# 2) Activate it:
#    - Windows: .venv\Scripts\activate
#    - macOS/Linux: source .venv/bin/activate
# 3) Install requirements: pip install -r requirements.txt
# 4) Start server: uvicorn main:app --reload --port 8000
#    Swagger UI (localhost:8000/docs)
# 5) Test (example):
#    curl -X POST "http://127.0.0.1:8000/analyze" -H "Content-Type: application/json" \
#      -d '{"reviews": ["I love the new feature, but the UI is too slow.", "Terrible support."]}'

# Example expected JSON response (fields may vary slightly depending on the models):
# {
#   "overall_sentiment": {"label": "Neutral", "score": 0.0},
#   "key_themes": ["ui", "support", "new feature"],
#   "actionable_feedback": "Improve customer support responsiveness and optimize UI performance to reduce lag."
# }