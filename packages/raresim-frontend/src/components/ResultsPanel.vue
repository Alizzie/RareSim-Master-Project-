<template>
  <main class="results-panel">

    <!-- ── Empty state ── -->
    <div v-if="state === 'idle'" class="empty-state">
      <div class="empty-icon">🧬</div>
      <div class="empty-title">No results yet</div>
      <div class="empty-sub">
        Enter HPO terms or clinical text, choose similarity methods,<br/>then run diagnosis.
      </div>
    </div>

    <!-- ── Loading state ── -->
    <div v-else-if="state === 'loading'" class="loading-state">
      <div class="loading-ring" />
      <div class="loading-title">Running diagnosis…</div>
      <div class="loading-sub">
        Running {{ runningMethods.length }} method{{ runningMethods.length !== 1 ? 's' : '' }}<span v-if="nDiseases > 0"> across {{ nDiseases }} diseases</span>
      </div>
      <div class="loading-methods">
        <span v-for="m in runningMethods" :key="m" class="loading-method">{{ m }}</span>
      </div>
    </div>

    <!-- ── Error state ── -->
    <div v-else-if="state === 'error'" class="error-state">
      <div class="error-icon">⚠️</div>
      <div class="error-title">Something went wrong</div>
      <div class="error-msg">{{ errorMessage }}</div>
      <button class="btn-retry" @click="$emit('retry')">Try again</button>
    </div>

    <!-- ── Results ── -->
    <div v-else-if="state === 'done'" class="results-view">

<!-- Header -->
<div class="results-header">
  <div class="results-title">Diagnosis Results</div>
  <div class="results-meta">
    <span>{{ results.length }} candidates</span>
    <span class="meta-sep">·</span>
    <span>{{ meta.methods_run?.length || 0 }} method{{ meta.methods_run?.length !== 1 ? 's' : '' }}</span>
    <span class="meta-sep">·</span>
    <span>{{ meta.n_patient_terms }} HPO terms</span>
    <span v-if="meta.runtime_seconds" class="meta-sep">·</span>
    <span v-if="meta.runtime_seconds">{{ meta.runtime_seconds.toFixed(1) }}s</span>
  </div>
  <div class="save-wrap">
    <select v-model="saveFormat" class="save-format-select">
      <option value="json">JSON</option>
      <option value="phenopacket">Phenopacket</option>
    </select>
    <button class="btn-save" :disabled="saving" @click="handleSave">
      <span v-if="saving">Saving…</span>
      <span v-else-if="saveStatus === 'saved'">✓ Saved</span>
      <span v-else-if="saveStatus === 'error'">Failed</span>
      <span v-else>Save Patient</span>
    </button>
  </div>
  <p v-if="saveFileName" class="save-filename">Saved as {{ saveFileName }}</p>
</div>

      <!-- Status chips -->
      <div class="status-bar">
        <div class="chip chip-green">
          <span class="chip-dot" />
          {{ results.length }} diseases ranked
        </div>
        <div class="chip chip-blue">
          <span class="chip-dot" />
          top score {{ topScore }}
        </div>
        <div v-for="m in meta.methods_run" :key="m" class="chip chip-gray">
          {{ methodLabel(m) }}
        </div>
      </div>


          <div v-if="methodsInResults.length > 1" class="method-filter-bar">
      <button
        :class="['method-filter-btn', { active: activeMethod === 'all' }]"
        @click="activeMethod = 'all'"
      >
        All
      </button>
      <button
        v-for="m in methodsInResults"
        :key="m"
        :class="['method-filter-btn', { active: activeMethod === m }]"
        @click="activeMethod = m"
      >
        {{ methodLabel(m) }}
      </button>
    </div>

      <!-- Result list -->
      <div class="results-list">
        <div
          v-for="(r, i) in filteredResults"
          :key="r.disease_id + r.method_name"
          :class="['result-card', { expanded: expandedIdx === i }]"
          @click="toggleExpand(i)"
        >
          <!-- Main row -->
          <div class="card-main">
            <div :class="['rank-badge', rankClass(r.rank)]">{{ r.rank }}</div>
            <div class="card-info">
              <div class="card-name">{{ r.label }}</div>
              <div class="card-sub">
                <span class="card-orpha">{{ r.disease_id }}</span>
                <span class="card-method">{{ methodLabel(r.method_name) }}</span>
              </div>
            </div>
            <div class="card-score-wrap">
              <div class="card-score">{{ r.score.toFixed(3) }}</div>
              <div class="score-bar-wrap">
                <div
                  class="score-bar"
                  :class="scoreBarClass(r.score)"
                  :style="{ width: scoreWidth(r.score) + '%' }"
                />
              </div>
            </div>
            <div class="expand-icon">{{ expandedIdx === i ? '▲' : '▼' }}</div>
          </div>

          <!-- Expanded detail -->
          <div v-if="expandedIdx === i" class="card-detail" @click.stop>
            <div class="detail-section">
              <div class="detail-label">Disease ID</div>
              <div class="detail-value mono">{{ r.disease_id }}</div>
            </div>

            <!-- Plain-language summary -->
            <div v-if="r.explanation?.summary" class="detail-section">
              <div class="detail-label">Summary</div>
              <div class="detail-value">{{ r.explanation.summary }}</div>
            </div>

            <!-- Coverage stats -->
            <div v-if="r.explanation?.coverage" class="detail-section">
              <div class="detail-label">Coverage</div>
              <div class="coverage-grid">
                <div class="coverage-item">
                  <span class="coverage-value">{{ pct(r.explanation.coverage.patient_hpo_coverage) }}</span>
                  <span class="coverage-key">patient terms matched</span>
                </div>
                <div class="coverage-item">
                  <span class="coverage-value">{{ pct(r.explanation.coverage.disease_hpo_coverage) }}</span>
                  <span class="coverage-key">disease terms matched</span>
                </div>
                <div class="coverage-item">
                  <span class="coverage-value">{{ r.explanation.coverage.n_matched_terms }}</span>
                  <span class="coverage-key">terms matched</span>
                </div>
                <div class="coverage-item">
                  <span class="coverage-value">{{ r.explanation.coverage.n_unmatched_patient_terms }}</span>
                  <span class="coverage-key">unmatched (patient)</span>
                </div>
              </div>
            </div>

            <!-- Matched terms -->
            <div v-if="r.explanation?.matched_terms?.length" class="detail-section">
              <div class="detail-label">Matched phenotypes ({{ r.explanation.matched_terms.length }})</div>
              <div class="term-list">
                <div
                  v-for="t in r.explanation.matched_terms.slice(0, 8)"
                  :key="t.id"
                  class="term-row term-row-match"
                >
                  <span class="term-label">{{ t.label }}</span>
                  <span class="term-id">{{ t.id }}</span>
                  <span v-if="t.ic !== undefined" class="term-ic">IC {{ t.ic.toFixed(2) }}</span>
                </div>
              </div>
            </div>

            <!-- Unmatched patient terms -->
            <div v-if="r.explanation?.unmatched_patient_terms?.length" class="detail-section">
              <div class="detail-label">Unmatched patient phenotypes ({{ r.explanation.unmatched_patient_terms.length }})</div>
              <div class="term-list">
                <div
                  v-for="t in r.explanation.unmatched_patient_terms.slice(0, 8)"
                  :key="t.id"
                  class="term-row term-row-unmatch"
                >
                  <span class="term-label">{{ t.label }}</span>
                  <span class="term-id">{{ t.id }}</span>
                  <span v-if="t.ic !== undefined" class="term-ic">IC {{ t.ic.toFixed(2) }}</span>
                </div>
              </div>
            </div>

            <!-- Method-specific details (structured where recognized, JSON fallback otherwise) -->
            <div v-if="r.explanation?.method_specific" class="detail-section">
              <div class="detail-label">Method details</div>

              <!-- Semantic BMA shape -->
              <template v-if="r.explanation.method_specific.bma_directions">
                <div class="ms-block">
                  <div class="ms-row">
                    <span class="ms-key">Variant</span>
                    <span class="ms-val">{{ r.explanation.method_specific.bma_variant }}</span>
                  </div>
                  <div class="ms-row">
                    <span class="ms-key">Patient → disease avg</span>
                    <span class="ms-val mono">{{ r.explanation.method_specific.bma_directions.patient_to_disease_avg?.toFixed(3) }}</span>
                  </div>
                  <div class="ms-row">
                    <span class="ms-key">Disease → patient avg</span>
                    <span class="ms-val mono">{{ r.explanation.method_specific.bma_directions.disease_to_patient_avg?.toFixed(3) }}</span>
                  </div>
                  <div class="ms-row">
                    <span class="ms-key">Asymmetry</span>
                    <span class="ms-val mono">{{ r.explanation.method_specific.bma_directions.asymmetry?.toFixed(3) }} ({{ formatAsymmetry(r.explanation.method_specific.bma_directions.asymmetry_interpretation) }})</span>
                  </div>
                </div>

                <div v-if="r.explanation.method_specific.semantic_clusters?.length" class="ms-subsection">
                  <div class="ms-subtitle">Semantic clusters</div>
                  <div
                    v-for="(c, ci) in r.explanation.method_specific.semantic_clusters"
                    :key="ci"
                    class="cluster-row"
                  >
                    <span class="cluster-mica">{{ c.mica_label }}</span>
                    <span class="cluster-meta">{{ c.n_patient_terms }} patient term{{ c.n_patient_terms !== 1 ? 's' : '' }} · avg {{ c.cluster_avg_score?.toFixed(3) }}</span>
                  </div>
                </div>

                <div v-if="r.explanation.method_specific.weak_patient_matches?.length" class="ms-subsection">
                  <div class="ms-subtitle">Weakest patient matches</div>
                  <div
                    v-for="w in r.explanation.method_specific.weak_patient_matches.slice(0, 5)"
                    :key="w.id"
                    class="term-row term-row-weak"
                  >
                    <span class="term-label">{{ w.label }} → {{ w.best_match_label }}</span>
                    <span class="term-ic">{{ w.best_score?.toFixed(3) }}</span>
                  </div>
                </div>

                <div v-if="r.explanation.method_specific.ic_filter_impact" class="ms-subsection">
                  <div class="ms-subtitle">
                    IC filter — removed {{ r.explanation.method_specific.ic_filter_impact.n_removed }} of
                    {{ r.explanation.method_specific.ic_filter_impact.terms_before_filter }} terms
                  </div>
                  <div class="tags-wrap-sm">
                    <span
                      v-for="t in r.explanation.method_specific.ic_filter_impact.removed_terms"
                      :key="t.id"
                      class="ic-removed-tag"
                    >{{ t.label }}</span>
                  </div>
                </div>
              </template>

              <!-- Set-based / formula shape -->
              <template v-else-if="r.explanation.method_specific.formula_components">
                <div class="ms-block">
                  <div class="ms-row">
                    <span class="ms-key">Formula</span>
                    <span class="ms-val">{{ r.explanation.method_specific.formula_components.formula }}</span>
                  </div>
                  <div class="ms-row">
                    <span class="ms-key">Intersection / union</span>
                    <span class="ms-val mono">{{ r.explanation.method_specific.formula_components.intersection_size }} / {{ r.explanation.method_specific.formula_components.union_size }}</span>
                  </div>
                  <div v-if="r.explanation.method_specific.ic_weighted_match_score !== undefined" class="ms-row">
                    <span class="ms-key">IC-weighted match score</span>
                    <span class="ms-val mono">{{ r.explanation.method_specific.ic_weighted_match_score?.toFixed(3) }}</span>
                  </div>
                </div>
                <div v-if="r.explanation.method_specific.top_ic_matched_terms?.length" class="ms-subsection">
                  <div class="ms-subtitle">Top IC-weighted matches</div>
                  <div
                    v-for="t in r.explanation.method_specific.top_ic_matched_terms"
                    :key="t.id"
                    class="term-row term-row-match"
                  >
                    <span class="term-id">{{ t.id }}</span>
                    <span class="term-ic">IC {{ t.ic?.toFixed(2) }}</span>
                  </div>
                </div>
              </template>

             <!-- HPO2Vec+ shape -->
              <template v-else-if="r.explanation.method_specific.embedding_method === 'hpo2vec_random_walk'">
                <div class="ms-block">
                  <div class="ms-row">
                    <span class="ms-key">Embedding method</span>
                    <span class="ms-val">{{ r.explanation.method_specific.embedding_method }}</span>
                  </div>
                  <div class="ms-row">
                    <span class="ms-key">Aggregation</span>
                    <span class="ms-val">{{ r.explanation.method_specific.aggregation }}</span>
                  </div>
                </div>
                <div v-if="r.explanation.method_specific.score_note" class="ms-subsection">
                  <div class="ms-subtitle">Score note</div>
                  <div class="detail-value small">{{ r.explanation.method_specific.score_note }}</div>
                </div>
                <div v-if="r.explanation.method_specific.interpretation_note" class="ms-subsection">
                  <div class="ms-subtitle">Interpretation</div>
                  <div class="detail-value small">{{ r.explanation.method_specific.interpretation_note }}</div>
                </div>
              </template>

              <!-- Transformer shape -->
              <template v-else-if="r.explanation.method_specific.embedding_method && r.explanation.method_specific.model_name">
                <div class="ms-block">
                  <div class="ms-row">
                    <span class="ms-key">Model</span>
                    <span class="ms-val">{{ r.explanation.method_specific.model_name }}</span>
                  </div>
                  <div class="ms-row">
                    <span class="ms-key">Pooling</span>
                    <span class="ms-val">{{ r.explanation.method_specific.pooling || 'n/a' }}</span>
                  </div>
                </div>
                <div v-if="r.explanation.method_specific.score_note" class="ms-subsection">
                  <div class="ms-subtitle">Score note</div>
                  <div class="detail-value small">{{ r.explanation.method_specific.score_note }}</div>
                </div>
                <div v-if="r.explanation.method_specific.interpretation_note" class="ms-subsection">
                  <div class="ms-subtitle">Interpretation</div>
                  <div class="detail-value small">{{ r.explanation.method_specific.interpretation_note }}</div>
                </div>
              </template>

              <!-- Autoencoder shape -->
              <template v-else-if="r.explanation.method_specific.embedding_method === 'denoising_autoencoder_latent'">
                <div class="ms-block">
                  <div class="ms-row">
                    <span class="ms-key">Aggregation</span>
                    <span class="ms-val">{{ r.explanation.method_specific.aggregation }}</span>
                  </div>
                </div>
                <div v-if="r.explanation.method_specific.score_note" class="ms-subsection">
                  <div class="ms-subtitle">Score note</div>
                  <div class="detail-value small">{{ r.explanation.method_specific.score_note }}</div>
                </div>
                <div v-if="r.explanation.method_specific.interpretation_note" class="ms-subsection">
                  <div class="ms-subtitle">Interpretation</div>
                  <div class="detail-value small">{{ r.explanation.method_specific.interpretation_note }}</div>
                </div>
              </template>

                            <!-- TF-IDF shape -->
              <template v-else-if="r.explanation.method_specific.tfidf_mode">
                <div class="ms-block">
                  <div class="ms-row">
                    <span class="ms-key">Mode</span>
                    <span class="ms-val">{{ methodLabel(r.explanation.method_specific.tfidf_mode) }}</span>
                  </div>
                  <div v-if="r.explanation.method_specific.ic_weighted_match_score !== undefined" class="ms-row">
                    <span class="ms-key">IC-weighted match score</span>
                    <span class="ms-val mono">{{ r.explanation.method_specific.ic_weighted_match_score?.toFixed(3) }}</span>
                  </div>
                  <div v-if="r.explanation.method_specific.idf_weighted_score !== undefined" class="ms-row">
                    <span class="ms-key">IDF-weighted score</span>
                    <span class="ms-val mono">{{ r.explanation.method_specific.idf_weighted_score?.toFixed(3) }}</span>
                  </div>
                  <div v-if="r.explanation.method_specific.vector_norms" class="ms-row">
                    <span class="ms-key">Cosine (check)</span>
                    <span class="ms-val mono">{{ r.explanation.method_specific.vector_norms.score_check?.toFixed(4) }}</span>
                  </div>
                </div>

                <!-- Contributing HPO terms -->
                <div v-if="r.explanation.method_specific.contributing_hpo_terms?.length" class="ms-subsection">
                  <div class="ms-subtitle">Contributing terms</div>
                  <div
                    v-for="t in r.explanation.method_specific.contributing_hpo_terms"
                    :key="t.hpo_id"
                    class="term-row term-row-match"
                  >
                    <span class="term-label">{{ t.hpo_label }}</span>
                    <span class="term-id">{{ t.hpo_id }}</span>
                    <span class="term-ic">IC {{ t.ic?.toFixed(2) }}</span>
                  </div>
                </div>

                <!-- Low IDF matches -->
                <div v-if="r.explanation.method_specific.low_idf_matches?.length" class="ms-subsection">
                  <div class="ms-subtitle">Low-IDF matches (noisy)</div>
                  <div class="tags-wrap-sm">
                    <span
                      v-for="t in r.explanation.method_specific.low_idf_matches"
                      :key="t.id"
                      class="ic-removed-tag"
                    >{{ t.label }}</span>
                  </div>
                </div>

                <!-- IC filter impact -->
                <div v-if="r.explanation.method_specific.ic_filter_impact" class="ms-subsection">
                  <div class="ms-subtitle">
                    IC filter — removed {{ r.explanation.method_specific.ic_filter_impact.n_removed }} of
                    {{ r.explanation.method_specific.ic_filter_impact.terms_before_filter }} terms
                  </div>
                  <div class="tags-wrap-sm">
                    <span
                      v-for="t in r.explanation.method_specific.ic_filter_impact.removed_terms"
                      :key="t.id"
                      class="ic-removed-tag"
                    >{{ t.label }}</span>
                  </div>
                </div>
              </template>

              <!-- LLM shape -->
            <template v-else-if="r.explanation.method_specific.llm_method">
              <div class="ms-block">
                <div class="ms-row">
                  <span class="ms-key">Model</span>
                  <span class="ms-val">{{ r.explanation.method_specific.model_name }}</span>
                </div>
                <div v-if="r.explanation.method_specific.confidence" class="ms-row">
                  <span class="ms-key">Confidence</span>
                  <span class="ms-val">{{ r.explanation.method_specific.confidence }}</span>
                </div>
              </div>
              <div v-if="r.explanation.method_specific.score_note" class="ms-subsection">
                <div class="ms-subtitle">Score note</div>
                <div class="detail-value small">{{ r.explanation.method_specific.score_note }}</div>
              </div>
              <div v-if="r.explanation.method_specific.llm_response_preview" class="ms-subsection">
                <div class="ms-subtitle">LLM response</div>
                <div class="detail-value small">{{ r.explanation.method_specific.llm_response_preview }}</div>
              </div>
            </template>

              <!-- Fallback: unrecognized shape -->
              <div v-else class="detail-value small mono-block">{{ JSON.stringify(r.explanation.method_specific, null, 2) }}</div>
            </div>

            <!-- Fallback for older / simpler explanation shapes -->
            <div v-if="r.shared_phenotype_labels?.length" class="detail-section">
              <div class="detail-label">Shared phenotypes</div>
              <div class="shared-terms">
                <span
                  v-for="s in r.shared_phenotype_labels"
                  :key="s"
                  class="shared-term"
                >{{ s }}</span>
              </div>
            </div>
            <div v-if="r.explanation?.top_patient_to_disease_matches?.length" class="detail-section">
              <div class="detail-label">Top term matches (patient → disease)</div>
              <div class="match-list">
                <div
                  v-for="m in r.explanation.top_patient_to_disease_matches.slice(0, 3)"
                  :key="m.patient_term"
                  class="match-row"
                >
                  <span class="match-term">{{ m.patient_term }}</span>
                  <span class="match-arrow">→</span>
                  <span class="match-term">{{ m.disease_term }}</span>
                  <span class="match-score">{{ m.score.toFixed(3) }}</span>
                </div>
              </div>
            </div>
            <div v-if="r.explanation?.score_note" class="detail-section">
              <div class="detail-label">Score note</div>
              <div class="detail-value small">{{ r.explanation.score_note }}</div>
            </div>
          </div>
        </div>
      </div>
      <!-- Method comparison (collapsible) -->
      <div v-if="comparison" class="comparison-section">
        <button class="comparison-toggle" @click="showComparison = !showComparison">
          <span class="chevron" :class="{ open: showComparison }">▸</span>
          Method comparison
          <span class="count">{{ comparison.methods.length }} methods</span>
        </button>

        <MethodComparison
          v-if="showComparison"
          :comparison="comparison"
          :case-id="meta?.case_id || ''"
          :input-hpo="inputHpo"
        />
      </div>
    </div>

  </main>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import MethodComparison from './MethodComparison.vue'
import { savePatient } from '../api/index.js'

const props = defineProps({
  state:          { type: String,  default: 'idle' },  // idle | loading | done | error
  results:        { type: Array,   default: () => [] },
  meta:           { type: Object,  default: () => ({}) },
  comparison: { type: Object, default: null },
  inputHpo: { type: Array, default: () => [] },
  errorMessage:   { type: String,  default: '' },
  runningMethods: { type: Array,   default: () => [] },
  nDiseases:      { type: Number,  default: 0 },
})

defineEmits(['retry'])

watch(() => props.results, () => {
  activeMethod.value = 'all'
})

const expandedIdx = ref(null)
const showComparison = ref(false)
const activeMethod = ref('all')
const saving = ref(false)
const saveStatus = ref('')
const saveFormat = ref('json')
const saveFileName = ref('')

const methodsInResults = computed(() => {
  const seen = new Set()
  props.results.forEach(r => seen.add(r.method_name))
  return [...seen]
})

const filteredResults = computed(() => {
  if (activeMethod.value === 'all') return props.results
  return props.results.filter(r => r.method_name === activeMethod.value)
})

const topScore = computed(() => {
  if (!props.results.length) return '—'
  return Math.max(...props.results.map(r => r.score)).toFixed(3)
})

const maxScore = computed(() =>
  props.results.length ? Math.max(...props.results.map(r => r.score)) : 1
)

function toggleExpand(i) {
  expandedIdx.value = expandedIdx.value === i ? null : i
}

async function handleSave() {
    saving.value = true
    saveStatus.value = ''
    try {
        const res = await savePatient({
            patient_id: 'patient_' + Date.now(),
            hpo_terms: props.inputHpo,
            raw_text: '',
            results: filteredResults.value,
            methods: activeMethod.value === 'all'
                ? (props.meta.methods_run || [])
                : [activeMethod.value],
            format: saveFormat.value,
        })
        saveStatus.value = 'saved'
        saveFileName.value = res.filename || ''
        setTimeout(() => { saveStatus.value = ''; saveFileName.value = '' }, 5000)
    } catch (e) {
        saveStatus.value = 'error'
    } finally {
        saving.value = false
    }
}

function formatAsymmetry(value) {
  const map = {
    disease_better_covered: 'disease better covered',
    patient_better_covered: 'patient better covered',
    balanced: 'balanced',
  }
  return map[value] || (value || '').replace(/_/g, ' ')
}

function pct(value) {
  if (value === undefined || value === null) return '—'
  return Math.round(value * 100) + '%'
}

function rankClass(rank) {
  if (rank === 1) return 'rank-gold'
  if (rank === 2) return 'rank-silver'
  if (rank === 3) return 'rank-bronze'
  return 'rank-other'
}

function scoreWidth(score) {
  return Math.round((score / maxScore.value) * 100)
}

function scoreBarClass(score) {
  const pct = score / maxScore.value
  if (pct > 0.8) return 'bar-high'
  if (pct > 0.6) return 'bar-mid'
  return 'bar-low'
}

const METHOD_LABELS = {
  semantic_resnik_bma:        'Resnik BMA',
  semantic_lin_bma:           'Lin BMA',
  semantic_jiang_conrath_bma: 'JC BMA',
  set_jaccard:                'Jaccard',
  set_dice:                   'Dice',
  set_cosine:                 'Cosine',
  set_overlap:                'Overlap',
  tfidf_hpo:        'TF-IDF (HPO)',
  tfidf_text:       'TF-IDF (Text)',
  tfidf_hybrid:     'TF-IDF (Hybrid)',
  tfidf_hpo_labels: 'TF-IDF (Labels)',
  transformer:                'Transformer',
  llm:                        'LLM',
  hpo2vec_plus:               'HPO2Vec+',
  denoising_autoencoder:      'Autoencoder',
}
function methodLabel(id) {
  return METHOD_LABELS[id] || id
}
</script>

<style scoped>
.results-panel {
  flex: 1;
  overflow-y: auto;
  padding: 28px 32px;
  background: var(--bg);
}

.btn-save {
  padding: 6px 14px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  font-family: var(--sans);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all .15s;
  white-space: nowrap;
}
.btn-save:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-light);
}
.btn-save:disabled { opacity: .5; cursor: not-allowed; }

/* ── Empty ── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 420px;
  gap: 10px;
  text-align: center;
}
.empty-icon { font-size: 36px; margin-bottom: 4px; }
.empty-title { font-size: 16px; font-weight: 500; color: var(--text); }
.empty-sub { font-size: 13px; color: var(--text-tertiary); line-height: 1.7; max-width: 300px; }

/* ── Loading ── */
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 420px;
  gap: 12px;
  text-align: center;
}
.loading-ring {
  width: 44px;
  height: 44px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-title { font-size: 16px; font-weight: 500; color: var(--text); }
.loading-sub { font-size: 13px; color: var(--text-tertiary); }
.loading-methods { display: flex; gap: 6px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
.loading-method {
  font-size: 11px;
  font-family: var(--mono);
  background: var(--accent-light);
  color: var(--accent);
  padding: 2px 8px;
  border-radius: 99px;
}

/* ── Error ── */
.error-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 420px;
  gap: 10px;
  text-align: center;
}
.error-icon { font-size: 32px; }
.error-title { font-size: 16px; font-weight: 500; color: var(--red); }
.error-msg { font-size: 13px; color: var(--text-secondary); max-width: 340px; line-height: 1.6; }
.btn-retry {
  margin-top: 8px;
  padding: 8px 20px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  font-family: var(--sans);
  font-size: 13px;
  cursor: pointer;
  transition: all .15s;
}
.btn-retry:hover { border-color: var(--accent); color: var(--accent); }

/* ── Results header ── */
.results-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 16px;
  flex-wrap: wrap;
  gap: 8px;
}
.results-title { font-size: 20px; font-weight: 600; letter-spacing: -0.03em; color: var(--text); }
.results-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-tertiary);
  font-family: var(--mono);
}
.meta-sep { opacity: .4; }

/* ── Status chips ── */
.status-bar { display: flex; gap: 6px; margin-bottom: 20px; flex-wrap: wrap; }
.chip {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 99px;
  border: 1px solid var(--border);
}
.chip-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}
.chip-green { background: var(--green-light); color: var(--green); border-color: #BBDFD2; }
.chip-green .chip-dot { background: var(--green); }
.chip-blue { background: var(--accent-light); color: var(--accent); border-color: #C8D8F9; }
.chip-blue .chip-dot { background: var(--accent); }
.chip-gray { background: var(--surface); color: var(--text-secondary); font-family: var(--mono); font-size: 11px; }

/* ── Cards ── */
.results-list { display: flex; flex-direction: column; gap: 8px; }
.result-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  cursor: pointer;
  transition: box-shadow .15s;
}
.result-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,.08); }
.result-card.expanded { border-color: var(--accent); }

.card-main {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 13px 16px;
}
.rank-badge {
  width: 28px;
  height: 28px;
  border-radius: 7px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 500;
  flex-shrink: 0;
}
.rank-gold   { background: #FEF3C7; color: #92400E; }
.rank-silver { background: #F3F4F6; color: #374151; }
.rank-bronze { background: #FEF0E7; color: #7C3C0E; }
.rank-other  { background: var(--tag-bg); color: var(--text-secondary); }

.card-info { flex: 1; min-width: 0; }
.card-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.card-sub {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 2px;
}
.card-orpha {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-tertiary);
}
.card-method {
  font-size: 11px;
  background: var(--tag-bg);
  color: var(--text-secondary);
  padding: 1px 7px;
  border-radius: 99px;
}

.card-score-wrap {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  flex-shrink: 0;
}
.card-score {
  font-family: var(--mono);
  font-size: 15px;
  font-weight: 500;
  color: var(--text);
}
.score-bar-wrap {
  width: 72px;
  height: 4px;
  background: var(--tag-bg);
  border-radius: 2px;
  overflow: hidden;
}
.score-bar { height: 100%; border-radius: 2px; transition: width .4s ease; }
.bar-high { background: var(--green); }
.bar-mid  { background: var(--accent); }
.bar-low  { background: var(--border-strong); }

.expand-icon {
  font-size: 10px;
  color: var(--text-tertiary);
  flex-shrink: 0;
  margin-left: 4px;
}

/* ── Card detail ── */
.card-detail {
  padding: 14px 16px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 12px;
  cursor: default;
}
.detail-section {}
.detail-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-bottom: 6px;
}
.detail-value { font-size: 13px; color: var(--text); line-height: 1.5; }
.detail-value.mono { font-family: var(--mono); font-size: 12px; }
.detail-value.small { font-size: 12px; color: var(--text-secondary); }

.shared-terms { display: flex; flex-wrap: wrap; gap: 5px; }
.shared-term {
  font-size: 12px;
  padding: 3px 9px;
  background: var(--green-light);
  color: var(--green);
  border-radius: 99px;
  font-weight: 500;
}

.match-list { display: flex; flex-direction: column; gap: 4px; }
.match-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}
.match-term { color: var(--text); }
.match-arrow { color: var(--text-tertiary); }
.match-score {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-secondary);
  margin-left: auto;
}

/* ── Coverage grid ── */
.coverage-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.coverage-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 10px;
  background: var(--tag-bg);
  border-radius: var(--radius);
}
.coverage-value {
  font-family: var(--mono);
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
}
.coverage-key {
  font-size: 10px;
  color: var(--text-tertiary);
  line-height: 1.3;
}

/* ── Term lists (matched / unmatched) ── */
.term-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.term-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  padding: 5px 9px;
  border-radius: 6px;
}
.term-row-match { background: var(--green-light); }
.term-row-unmatch { background: var(--red-light); }
.term-label {
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--text);
}
.term-id {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-tertiary);
  flex-shrink: 0;
}
.term-ic {
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 600;
  color: var(--text-secondary);
  flex-shrink: 0;
}
.mono-block {
  font-family: var(--mono);
  font-size: 11px;
  background: var(--tag-bg);
  padding: 8px 10px;
  border-radius: 6px;
  white-space: pre-wrap;
  word-break: break-word;
}

/* ── Method-specific structured blocks ── */
.ms-block {
  display: flex;
  flex-direction: column;
  gap: 4px;
  background: var(--tag-bg);
  border-radius: 6px;
  padding: 8px 10px;
}
.ms-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  font-size: 12px;
}
.ms-key { color: var(--text-tertiary); }
.ms-val { color: var(--text); text-align: right; }
.ms-val.mono { font-family: var(--mono); font-size: 11px; }

.ms-subsection { margin-top: 10px; }
.ms-subtitle {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  margin-bottom: 6px;
}
.cluster-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 12px;
  padding: 5px 9px;
  border-radius: 6px;
  background: var(--accent-light);
  margin-bottom: 4px;
}
.cluster-mica { color: var(--text); font-weight: 500; }
.cluster-meta { color: var(--text-tertiary); font-size: 11px; flex-shrink: 0; }

.term-row-weak {
  background: var(--tag-bg);
  display: flex;
  justify-content: space-between;
}

.tags-wrap-sm { display: flex; flex-wrap: wrap; gap: 4px; }
.ic-removed-tag {
  font-size: 11px;
  padding: 2px 8px;
  background: var(--tag-bg);
  color: var(--text-tertiary);
  border-radius: 99px;
}
.comparison-section { margin-top: 20px; }
.comparison-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 12px 14px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  cursor: pointer;
  text-align: left;
}
.comparison-toggle:hover { border-color: var(--border-strong); }
.comparison-toggle .chevron {
  display: inline-block;
  color: var(--text-secondary);
  transition: transform .15s;
}
.comparison-toggle .chevron.open { transform: rotate(90deg); }
.comparison-toggle .count { margin-left: auto; color: var(--text-tertiary); font-size: 11px; }
.comparison-section .mc { margin-top: 12px; }

.method-filter-bar {
  display: flex;
  gap: 6px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.method-filter-btn {
  font-size: 12px;
  font-weight: 500;
  padding: 5px 12px;
  border-radius: 99px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all .15s;
  font-family: var(--sans);
}
.method-filter-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
}
.method-filter-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: white;
}
.save-filename {
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--mono);
  margin-top: 4px;
  text-align: right;
}

</style>