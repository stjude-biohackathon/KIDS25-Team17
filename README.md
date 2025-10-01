# KIDS25-Team17

Rename the `env.json` file to `.env` and provide the NCBI API key and the linked email.

If you already have a PubMed/NCBI account, you can request an API key from the account settings page https://account.ncbi.nlm.nih.gov/settings/.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run app.py
```
