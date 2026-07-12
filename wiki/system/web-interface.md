# Web Interface

RareSim includes a browser-based interface for running patient diagnosis interactively. It consists of a Vue 3 frontend and a FastAPI backend.

## Running the Interface

**Terminal 1 — backend:**
```bash
RARESIM_ROOT=/path/to/RareSim uvicorn raresim_api.main:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
cd packages/raresim-frontend
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Input Panel

The left panel handles patient input and method selection.

### Phenotype Search
A searchbar at the top lets you find HPO terms by name (e.g. typing "ataxia" returns all matching phenotypes). Each result has two buttons:
- **+ Include** — adds the term to the patient's HPO term list
- **− Exclude** — marks the term as excluded; excluded terms are filtered out before diagnosis runs

Excluded terms are sent to the backend as `excluded_hpo_terms` and removed from the patient's HPO set before any similarity method runs.

### Input Modes
- **HPO Terms** — paste HPO IDs directly (e.g. `HP:0001251, HP:0000545`). Terms are parsed and displayed as tags.
- **Raw Text** — paste clinical notes. Use the Extract bar to map text to HPO terms via dictionary lookup, FastHPOCR, GPT-4o-mini, PhenoBrain, or BioNER.

### Similarity Methods
Select one or more methods to run. Available methods:

| Method | Badge | Notes |
|--------|-------|-------|
| Resnik BMA | IC | Semantic similarity using information content |
| Lin BMA | IC | Lin's normalized semantic similarity |
| JC BMA | IC | Jiang-Conrath semantic similarity |
| Jaccard | set | Set overlap: intersection / union |
| Dice | set | Set overlap: 2 × intersection / (A + B) |
| TF-IDF | txt | Term frequency-inverse document frequency |
| Transformer | emb | Sentence transformer embeddings |
| LLM | llm | GPT-based ranking |
| HPO2Vec+ | emb | Node2Vec embeddings on enriched HPO graph |
| Autoencoder | nn | Denoising autoencoder latent space similarity. |

### Top-K
Slider to control how many results are returned per method (5, 10, 15, or 20).

## Results Panel

The right panel shows diagnosis results after running.

### Method Filter
When multiple methods are selected, filter buttons appear above the results list —> one per method plus an "All" option. Clicking a method shows only that method's top-K results

### Result Cards
Each card shows:
- Rank, disease label, disease ID
- Method used (shown as a badge)
- Score with a visual bar
- Expandable detail section with shared phenotypes and top term matches

### Save Patient
After running a diagnosis, click **Save Patient** to save the session to disk. Choose the format before saving:
- **JSON** — saves HPO terms, raw text, methods used, and full results
- **Phenopacket** — saves HPO terms in GA4GH phenopacket format with results in metadata

Files are saved to `outputs/webapp/patient_profiles/` and can be retrieved via `GET /api/patients`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/extract` | Extract HPO terms from clinical text |
| POST | `/api/diagnose` | Run similarity diagnosis |
| GET | `/api/hpo/search?q=` | Search HPO terms by label |
| POST | `/api/patients/save` | Save patient session to disk |
| GET | `/api/patients` | List saved patient sessions |
| GET | `/api/patients/{filename}` | Load a saved patient session |
| GET | `/api/health` | Health check |