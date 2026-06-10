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
        Running {{ runningMethods.length }} method{{ runningMethods.length !== 1 ? 's' : '' }}
        across {{ nDiseases }} diseases
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

      <!-- Result list -->
      <div class="results-list">
        <div
          v-for="(r, i) in results"
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
    </div>

  </main>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  state:          { type: String,  default: 'idle' },  // idle | loading | done | error
  results:        { type: Array,   default: () => [] },
  meta:           { type: Object,  default: () => ({}) },
  errorMessage:   { type: String,  default: '' },
  runningMethods: { type: Array,   default: () => [] },
  nDiseases:      { type: Number,  default: 0 },
})

defineEmits(['retry'])

const expandedIdx = ref(null)

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
  tfidf:                      'TF-IDF',
  transformer:                'Transformer',
  llm:                        'LLM',
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
</style>
