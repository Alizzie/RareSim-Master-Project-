/**
 * API layer — all calls to the FastAPI backend go through here.
 * Base URL proxied via vite.config.js to http://localhost:8000
 */

const BASE = '/api'

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

/**
 * Extract HPO terms from raw clinical text.
 *
 * POST /api/extract
 * body: { text: string, method: string }
 * returns: { terms: [{ hpo_id, label, method, confidence }] }
 */
export function extractTerms(text, method) {
  return post('/extract', { text, method })
}

/**
 * Run similarity diagnosis.
 *
 * POST /api/diagnose
 * body: {
 *   mode: 'hpo' | 'text',
 *   hpo_terms: string[],        // used when mode = 'hpo' or after extraction
 *   raw_text: string | null,    // used when mode = 'text' (for transformer/llm)
 *   methods: string[],
 *   top_k: number,
 * }
 * returns: {
 *   results: [{
 *     rank, disease_id, label, score, method_name,
 *     shared_phenotype_labels, explanation
 *   }],
 *   meta: { n_diseases, runtime_seconds, methods_run }
 * }
 */
export function diagnose(payload) {
  return post('/diagnose', payload)
}
