# ZentroMart VoiceOps — 2026 Capstone Submission

A support agent over the **synthetic** ZentroMart dataset, with an independent prompt-only baseline, policy-grounded retrieval, verified mock order lookups, multi-turn escalation handling, and Gnani-only voice input/output. The interactive microphone demo is built with Streamlit and the LLM client supports Groq's OpenAI-compatible API.

## Important: Dataset supplied separately

This submission deliberately **excludes the official capstone dataset**. Put the original unmodified `Capstone_Project_Dataset_FTTT_2026` folder inside `vendor/`, or set `VOICEOPS_DATASET_DIR` to its path. **Tests and all runtime operations require that dataset; running without it will fail.** Do not commit the generated `tickets_created.csv`. The dataset is synthetic; do not use the mock verification system with actual customer accounts.

## Install (Windows / VS Code)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with your own **private** Groq and Gnani keys. Never upload this file or publish it. The included `.env.example` has placeholders only. The app and voice scripts load `.env` explicitly. CLI scripts `run.py` / `scripts/evaluate.py` read environment variables; load them in your PowerShell session as needed:

```powershell
Get-Content .env | Where-Object { $_ -match '^\s*[^#\s][^=]*=' } | ForEach-Object {
    $name, $value = $_ -split '=', 2
    [Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim(), 'Process')
}
$env:PYTHONUTF8='1'
```

## Run

```powershell
python vendor/Capstone_Project_Dataset_FTTT_2026/validate_dataset.py
python -m pytest -q
python scripts/build_index.py
python run.py --baseline --question 'What is your return policy?'
python run.py --question 'What is the return window for electronics?'
python scripts/evaluate_stage3_offline.py
python scripts/evaluate.py
python -m streamlit run voiceops_interactive.py
```

`evaluate.py` makes live baseline and RAG calls, may cost quota, and may encounter HTTP 429. To avoid re-running it unnecessarily, see the preserved raw results in `reports/`.

## Interactive voice demo

On your local computer open `http://localhost:8501` after starting Streamlit. Record a question in English or Hindi (select the STT language); click **Send recording**. Review the transcript, the assistant's answer and play the generated Gnani TTS reply. Follow-up messages within the same browser session use the same `Agent` instance. The app intentionally disables ticket creation; a handover decision is not an actual connection to a human.

## Architecture

See `ARCHITECTURE.md` for the diagram and data/security boundaries.

## Verified results and limitations

- Synthetic dataset validation: PASS.
- Local automated tests: **23/23 passed** during independent packaging check.
- Offline multi-turn *routing decision* matches: **96/96** with RAG stub and tickets disabled. This is not live dialogue correctness.
- Latest archived baseline: **51 LIVE, 1 ERROR** of 52.
- Latest archived RAG: **52 LIVE** of 52.
- Exploratory mean gold-KB-section **Recall@5 = 64.89%** across 47 applicable questions, not answer accuracy or official score.
- User tested live Gnani STT and TTS and reported a functioning multi-turn Streamlit demo. No demo video was attached; this build cannot independently rerun those credentials.
- Full answer quality and production identity/security have **not** been established. The Groq baseline intentionally cannot access company policies or customer accounts.

See `reports/final_evaluation.pdf` for the evaluation report, `reports/final_verified_summary.json` for machine-readable metrics, and `demo/demo_script.md` for a demonstration outline.
