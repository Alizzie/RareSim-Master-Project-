# RareSim Frontend

Vue 3 + FastAPI web interface for the RareSim rare disease similarity pipeline.

## Structure

```
raresim-frontend/
  src/
    App.vue                  ← root component, wires InputPanel + ResultsPanel
    main.js                  ← Vue entry point
    components/
      InputPanel.vue         ← left panel: HPO input / raw text + extraction / method selector
      ResultsPanel.vue       ← right panel: ranked disease results
    api/
      index.js               ← all frontend API calls (extract, diagnose)
      backend/
        main.py              ← FastAPI backend (place in your RareSim src/api/)
  public/
    index.html
  package.json
  vite.config.js             ← dev server proxies /api → localhost:8000
```

## Setup

### 1. Backend (FastAPI)

Copy `src/api/backend/main.py` into your RareSim repo at `src/api/main.py`.

Install FastAPI:
```bash
pip install fastapi uvicorn
```

Run the backend from your RareSim project root:
```bash
uvicorn raresim_api.main:app --reload --port 8000
```

Check it works:
```bash
curl http://localhost:8000/api/health
```

### 2. Frontend (Vue)

```bash
cd raresim-frontend
npm install
npm run dev
```

Open http://localhost:3000

The Vite dev server automatically proxies `/api/*` requests to `http://localhost:8000`.

## API Endpoints

### POST /api/extract
Extract HPO terms from raw clinical text.

```json
// request
{ "text": "patient with cerebellar ataxia and myopia", "method": "dictionary" }

// response
{ "terms": [{ "hpo_id": "HP:0001251", "label": "Cerebellar ataxia", "method": "...", "confidence": 1.0 }] }
```

### POST /api/diagnose
Run similarity and return ranked diseases.

```json
// request
{
  "mode": "hpo",
  "hpo_terms": ["HP:0001251", "HP:0000545"],
  "raw_text": null,
  "methods": ["semantic_resnik_bma", "transformer"],
  "top_k": 10
}

// response
{
  "results": [
    { "rank": 1, "disease_id": "ORPHA:95", "label": "Friedreich ataxia", "score": 0.891, "method_name": "semantic_resnik_bma", "shared_phenotype_labels": [...], "explanation": {...} }
  ],
  "meta": { "n_patient_terms": 2, "n_diseases": 9800, "methods_run": [...], "runtime_seconds": 4.2 }
}
```

## Input Modes

**HPO Terms mode** — paste HPO IDs directly (HP:0001251, HP:0000545).
The frontend parses them with regex and shows them as tags.

**Raw Text mode** — paste clinical description.
Optional: click Extract to run one of the 5 extraction methods and convert text → HPO IDs.
Raw text is also passed to transformer/LLM methods directly.