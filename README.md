# Career Counseling Agent (HW11)

Streamlit app that uses Gemini (e.g., `gemini-2.5-flash`), LangChain, and custom tools to help with career planning:

- Skills Gap Analyzer
- Resume Scorer (0-10 with fixes)
- Salary Estimator
- Interview Question Generator

## Quick start

```bash
cd /Users/spartan/Workspace/Data-236/HW11
python -m pip install -r requirements.txt
# or use the provided .env file
echo "GOOGLE_API_KEY=your_key_here" > .env
streamlit run app.py
```

Use the sidebar to pick the model and temperature, or clear the chat. Tool call traces can be toggled on/off.

## Example queries

- "Here are my skills (Python, SQL, Airflow). What are my gaps for a data engineer role in Seattle?"
- "Score this resume for a senior product manager and suggest improvements."
- "Estimate salary for a remote staff ML engineer with 8 years of experience."
- "Generate behavioral and technical questions for a mid-level backend engineer."

## Notes for submission

- Capture screenshots of the running app with at least one example for each tool.
- If the API key is missing/invalid you'll see an error in the assistant reply; set `GOOGLE_API_KEY` and rerun.
