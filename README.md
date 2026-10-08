# AI-Resume-Assistant
# 📄 ATS Resume Analyzer

Upload your resume (PDF, DOCX or TXT) and get:

- an **ATS score out of 100** with five category scores
- **prioritised improvements** (High / Medium / Low) with exact fixes
- **keywords found and missing**, optionally matched against a pasted job description
- **stronger rewrites** of weak bullet points
- instant **rule-based checks** (contact info, sections, bullet count, quantified results)
- a downloadable Markdown report

Built with [Streamlit](https://streamlit.io) and Google's Gemini Flash model.

> The score is an AI estimate of how ATS-friendly a resume is. It is not the output of a real ATS and may vary slightly between runs.

## How it works

1. The resume text is extracted with `pypdf` / `python-docx`. Image-only PDFs are sent to Gemini directly.
2. Simple regex checks run locally (email, phone, sections, bullets, numbers).
3. Gemini returns a structured JSON analysis.
4. The app validates the JSON and computes the overall score itself from weighted category scores:
   formatting 20%, keywords 25%, content impact 30%, structure 15%, readability 10%.

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-analyzer.git
cd ats-resume-analyzer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Get a free API key at <https://aistudio.google.com/apikey>, then either paste it into the sidebar or set it once:

```bash
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
```

You can also create `.streamlit/secrets.toml` (never commit this file):

```toml
GEMINI_API_KEY = "your-key"
# GEMINI_MODEL = "gemini-3.5-flash"   # optional
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (the repo must contain `app.py` and `requirements.txt` at the top level).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Choose your repository, branch `main`, and main file path `app.py`.
5. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
6. Click **Deploy**.

Tip: if you do not want visitors to use your quota, leave the secret empty. Each visitor then pastes their own key in the sidebar.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Secrets, env var, or sidebar | none |
| `GEMINI_MODEL` | Secrets, env var, or sidebar | `gemini-3.5-flash` |

If the chosen model name is not found, the app automatically retries with `gemini-2.5-flash`. Model names change over time; see <https://ai.google.dev/gemini-api/docs/models>.

## Limitations

- Maximum file size is 5 MB.
- Password-protected PDFs are not supported.
- Multi-column or graphic-heavy resumes may extract in a jumbled order, which is itself a sign they are hard for an ATS to read.
- Free-tier Gemini keys have rate limits. If you see a quota message, wait a minute.

## Project structure

```
ats-resume-analyzer/
├── app.py
├── requirements.txt
└── README.md
```

## Privacy

Resumes are processed in memory and sent to the Gemini API for analysis. The app does not save uploaded files.
