<template>
  <aside class="input-panel">

    <!-- ── Header ── -->
    <div class="panel-header">
      <span class="panel-label">Patient Input</span>

      <!-- Mode tabs -->
      <div class="mode-tabs">
        <button
          :class="['mode-tab', { active: mode === 'hpo' }]"
          @click="mode = 'hpo'"
        >
          <IconTerms />
          HPO Terms
        </button>
        <button
          :class="['mode-tab', { active: mode === 'text' }]"
          @click="mode = 'text'"
        >
          <IconText />
          Raw Text
        </button>
      </div>
    </div>

         <!-- ══ PHENOTYPE SEARCH ══ -->
<div class="search-section">
  <div class="section-label">Phenotype search</div>
  <div class="search-wrap">
    <input
      v-model="searchQuery"
      class="search-input"
      placeholder="Search phenotypes e.g. ataxia…"
      @input="onSearch"
    />
  </div>
  <div v-if="searchResults.length" class="search-results">
    <div
      v-for="r in searchResults"
      :key="r.hpo_id"
      class="search-result-row"
    >
      <div class="search-result-info">
        <span class="search-result-label">{{ r.label }}</span>
        <span class="search-result-id">{{ r.hpo_id }}</span>
      </div>
      <div class="search-result-actions">
        <button class="btn-include" @click="includeTerm(r)">+ Include</button>
        <button class="btn-exclude" @click="excludeTerm(r)">− Exclude</button>
      </div>
    </div>
  </div>
  <div v-if="excludedTerms.length" class="terms-section">
    <div class="section-label">Excluded terms ({{ excludedTerms.length }})</div>
    <div class="tags-wrap">
      <span
        v-for="t in excludedTerms"
        :key="t.hpo_id"
        class="tag tag-excluded"
      >
        {{ t.label }}
        <span class="tag-id">{{ t.hpo_id }}</span>
        <button class="tag-remove" @click="removeExcluded(t.hpo_id)">×</button>
      </span>
    </div>
  </div>
</div>

    <!-- ══ HPO MODE ══ -->
    <div v-if="mode === 'hpo'" class="mode-body">
      <p class="input-hint">
        Enter HPO term IDs separated by commas or new lines.<br/>
        Example: <code>HP:0001251, HP:0000545</code>
      </p>
      <textarea
        v-model="hpoRaw"
        class="hpo-textarea"
        placeholder="HP:0001251, HP:0000545, HP:0001263…"
        @input="parseHpoInput"
        spellcheck="false"
      />

      <!-- Parsed tags -->
      <div v-if="parsedTerms.length" class="terms-section">
        <div class="section-label">Parsed ({{ parsedTerms.length }})</div>
        <div class="tags-wrap">
          <span
            v-for="t in parsedTerms"
            :key="t"
            class="tag tag-hpo"
          >
            {{ t }}
            <button class="tag-remove" @click="removeTerm(t)">×</button>
          </span>
        </div>
      </div>

      <p v-if="hpoError" class="input-error">{{ hpoError }}</p>
    </div>

    <!-- ══ TEXT MODE ══ -->
    <div v-if="mode === 'text'" class="mode-body">
      <textarea
        v-model="rawText"
        class="text-textarea"
        placeholder="Paste clinical description here…&#10;&#10;e.g. 12-year-old with progressive cerebellar ataxia, mild intellectual disability, and bilateral myopia since age 8."
      />

      <!-- Extract bar -->
      <div class="extract-bar">
        <span class="extract-label">Extract HPO terms</span>
        <select v-model="extractMethod" class="extract-select">
          <option value="dictionary">Dictionary</option>
          <option value="fast_hpo_cr">FastHPOCR</option>
          <option value="chatgpt">GPT-4o-mini</option>
          <option value="phenobrain_api">PhenoBrain</option>
          <option value="biomedical_ner">BioNER</option>
        </select>
        <button
          class="btn-extract"
          :disabled="!rawText.trim() || extracting"
          @click="runExtraction"
        >
          {{ extracting ? 'Extracting…' : 'Extract' }}
        </button>
      </div>

      <!-- Extraction error -->
      <p v-if="extractError" class="input-error">{{ extractError }}</p>

      <!-- Extracted terms preview -->
      <div v-if="extractedTerms.length" class="terms-section">
        <div class="section-label">
          Extracted terms ({{ extractedTerms.length }})
          <span class="method-used">via {{ lastExtractMethod }}</span>
        </div>
        <div class="tags-wrap">
          <span
            v-for="t in extractedTerms"
            :key="t.hpo_id"
            class="tag tag-extracted"
            :title="`${t.hpo_id} · confidence: ${t.confidence}`"
          >
            {{ t.label }}
            <span class="tag-id">{{ t.hpo_id }}</span>
          </span>
        </div>
        <p class="extract-note">
          These terms will be used as HPO input for semantic / set-based methods.
          Raw text will be used directly for transformer / LLM.
        </p>
      </div>
    </div>

    <!-- ══ METHOD SELECTOR ══ -->
    <div class="methods-section">
      <div class="section-label">Similarity methods</div>
      <div class="methods-grid">
        <label
          v-for="m in availableMethods"
          :key="m.id"
          :class="['method-item', { checked: selectedMethods.has(m.id) }]"
        >
          <input
            type="checkbox"
            :value="m.id"
            :checked="selectedMethods.has(m.id)"
            @change="toggleMethod(m.id)"
          />
          <div class="check-box">
            <span v-if="selectedMethods.has(m.id)" class="check-icon">✓</span>
          </div>
          <span class="method-label">{{ m.label }}</span>
          <span class="method-badge">{{ m.badge }}</span>
          <span v-if="m.note && selectedMethods.has(m.id)" class="method-note">{{ m.note }}</span>
        </label>
      </div>
    </div>

    <!-- ══ TOP-K ══ -->
    <div class="topk-section">
      <div class="section-label">Top-K results</div>
      <div class="topk-row">
        <input
          v-model.number="topK"
          type="range"
          min="5"
          max="20"
          step="5"
          class="topk-slider"
        />
        <span class="topk-value">{{ topK }}</span>
      </div>
    </div>

    <!-- ══ RUN BUTTON ══ -->
    <div class="run-section">
      <button
        class="btn-run"
        :disabled="!canRun || running"
        @click="$emit('run', buildPayload())"
      >
        <span v-if="running" class="spinner" />
        <IconPlay v-else />
        {{ running ? 'Running…' : 'Run Diagnosis' }}
      </button>
      <p v-if="!canRun && !running" class="run-hint">
        {{ runHint }}
      </p>
    </div>

  </aside>
</template>

<script setup>
import { ref, computed, reactive } from 'vue'
import { extractTerms } from '../api/index.js'
import { searchHpo } from '../api/index.js'

// ── icons (inline SVG components) ─────────────────────────────────────────
const IconTerms = {
  template: `<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="3"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3"/>
  </svg>`
}
const IconText = {
  template: `<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
    <path d="M4 6h16M4 12h16M4 18h10"/>
  </svg>`
}
const IconPlay = {
  template: `<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24">
    <polygon points="5 3 19 12 5 21 5 3"/>
  </svg>`
}

// ── props / emits ──────────────────────────────────────────────────────────
const emit = defineEmits(['run'])
const props = defineProps({
  running: { type: Boolean, default: false }
})

// ── state ──────────────────────────────────────────────────────────────────
const mode          = ref('hpo')
const hpoRaw        = ref('')
const parsedTerms   = ref([])
const hpoError      = ref('')
const rawText       = ref('')
const extractMethod = ref('dictionary')
const extractedTerms = ref([])
const lastExtractMethod = ref('')
const extracting    = ref(false)
const extractError  = ref('')
const topK          = ref(10)
const searchQuery = ref('')
const searchResults = ref([])
const excludedTerms = ref([])
let searchTimeout = null

const selectedMethods = reactive(new Set(['semantic_resnik_bma', 'transformer']))

const availableMethods = [
  { id: 'semantic_resnik_bma',         label: 'Resnik BMA',   badge: 'IC'  },
  { id: 'semantic_lin_bma',            label: 'Lin BMA',      badge: 'IC'  },
  { id: 'semantic_jiang_conrath_bma',  label: 'JC BMA',       badge: 'IC'  },
  { id: 'set_jaccard',                 label: 'Jaccard',      badge: 'set' },
  { id: 'set_dice',                    label: 'Dice',         badge: 'set' },
  { id: 'tfidf',                       label: 'TF-IDF',       badge: 'txt' },
  { id: 'transformer',                 label: 'Transformer',  badge: 'emb' },
  { id: 'llm',                         label: 'LLM',          badge: 'llm' },
  { id: 'hpo2vec_plus',                label: 'HPO2Vec+',     badge: 'emb' },
  { id: 'denoising_autoencoder', label: 'Autoencoder', badge: 'nn', note: 'Works best with 10+ HPO terms' },
]

// ── computed ───────────────────────────────────────────────────────────────
const hasHpoInput = computed(() =>
  mode.value === 'hpo'
    ? parsedTerms.value.length > 0
    : rawText.value.trim().length > 0
)

const canRun = computed(() =>
  hasHpoInput.value && selectedMethods.size > 0
)

const runHint = computed(() => {
  if (!hasHpoInput.value) return mode.value === 'hpo'
    ? 'Enter at least one HPO term ID'
    : 'Enter clinical text'
  if (selectedMethods.size === 0) return 'Select at least one method'
  return ''
})

// ── HPO parsing ────────────────────────────────────────────────────────────
const HPO_PATTERN = /HP:\d{7}/g

function parseHpoInput() {
  hpoError.value = ''
  const raw = hpoRaw.value
  if (!raw.trim()) { parsedTerms.value = []; return }

  const found = [...new Set(raw.match(HPO_PATTERN) || [])]
  if (!found.length) {
    hpoError.value = 'No valid HPO IDs found. Format: HP:0001251'
  }
  parsedTerms.value = found
}

function removeTerm(id) {
  parsedTerms.value = parsedTerms.value.filter(t => t !== id)
  // also remove from textarea
  hpoRaw.value = parsedTerms.value.join(', ')
}

function onSearch() {
  clearTimeout(searchTimeout)
  if (searchQuery.value.length < 2) { searchResults.value = []; return }
  searchTimeout = setTimeout(async () => {
    const data = await searchHpo(searchQuery.value)
    searchResults.value = data.terms
  }, 300)
}

function includeTerm(term) {
  if (!parsedTerms.value.includes(term.hpo_id)) {
    parsedTerms.value.push(term.hpo_id)
    hpoRaw.value = parsedTerms.value.join(', ')
  }
  searchResults.value = []
  searchQuery.value = ''
}

function excludeTerm(term) {
  if (!excludedTerms.value.find(t => t.hpo_id === term.hpo_id)) {
    excludedTerms.value.push(term)
  }
  searchResults.value = []
  searchQuery.value = ''
}

function removeExcluded(id) {
  excludedTerms.value = excludedTerms.value.filter(t => t.hpo_id !== id)
}

// ── Extraction ─────────────────────────────────────────────────────────────
async function runExtraction() {
  extracting.value = true
  extractError.value = ''
  try {
    const data = await extractTerms(rawText.value, extractMethod.value)
    extractedTerms.value = data.terms
    lastExtractMethod.value = extractMethod.value
  } catch (e) {
    extractError.value = `Extraction failed: ${e.message}`
  } finally {
    extracting.value = false
  }
}

// ── Methods ────────────────────────────────────────────────────────────────
function toggleMethod(id) {
  if (selectedMethods.has(id)) selectedMethods.delete(id)
  else selectedMethods.add(id)
}

// ── Build payload ──────────────────────────────────────────────────────────
function buildPayload() {
  const hpoTerms = mode.value === 'hpo'
    ? parsedTerms.value
    : extractedTerms.value.map(t => t.hpo_id)

  return {
    mode: mode.value,
    hpo_terms: hpoTerms,
    excluded_hpo_terms: excludedTerms.value.map(t => t.hpo_id),
    raw_text: mode.value === 'text' ? rawText.value : null,
    methods: [...selectedMethods],
    top_k: topK.value,
  }
}
</script>

<style scoped>
.input-panel {
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  width: 400px;
  flex-shrink: 0;
  overflow-y: auto;
}

/* ── Header ── */
.panel-header {
  padding: 20px 20px 0;
  flex-shrink: 0;
}
.panel-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: .07em;
  margin-bottom: 12px;
}

/* ── Mode tabs ── */
.mode-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  background: var(--tag-bg);
  border-radius: var(--radius);
  padding: 3px;
  margin-bottom: 16px;
}
.mode-tab {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 7px 12px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--text-secondary);
  font-family: var(--sans);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all .15s;
}
.mode-tab.active {
  background: var(--surface);
  color: var(--text);
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
}

/* ── Mode body ── */
.mode-body {
  padding: 0 20px 16px;
  flex-shrink: 0;
}
.input-hint {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.6;
  margin-bottom: 10px;
}
.input-hint code {
  font-family: var(--mono);
  background: var(--tag-bg);
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 11px;
}
.hpo-textarea,
.text-textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-family: var(--mono);
  font-size: 12px;
  line-height: 1.7;
  background: var(--bg);
  color: var(--text);
  resize: vertical;
  outline: none;
  transition: border .15s;
}
.hpo-textarea { min-height: 90px; }
.text-textarea { min-height: 160px; font-family: var(--sans); font-size: 13px; }
.hpo-textarea:focus,
.text-textarea:focus {
  border-color: var(--accent);
  background: var(--surface);
}
.hpo-textarea::placeholder,
.text-textarea::placeholder { color: var(--text-tertiary); }

/* ── Tags ── */
.terms-section { margin-top: 12px; }
.section-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.method-used {
  font-size: 10px;
  background: var(--tag-bg);
  padding: 1px 6px;
  border-radius: 99px;
  text-transform: none;
  letter-spacing: 0;
  font-weight: 400;
  color: var(--text-secondary);
}
.tags-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 500;
  padding: 3px 8px 3px 9px;
  border-radius: 99px;
}
.tag-hpo {
  background: var(--accent-light);
  color: var(--accent);
  border: 1px solid #C8D8F9;
  font-family: var(--mono);
  font-size: 11px;
}
.tag-extracted {
  background: var(--green-light);
  color: var(--green);
  border: 1px solid #BBDFD2;
  flex-direction: column;
  align-items: flex-start;
  gap: 1px;
  padding: 4px 10px;
  border-radius: 8px;
}
.tag-id {
  font-family: var(--mono);
  font-size: 10px;
  opacity: .7;
}
.tag-remove {
  background: none;
  border: none;
  cursor: pointer;
  color: inherit;
  opacity: .6;
  font-size: 14px;
  line-height: 1;
  padding: 0;
  transition: opacity .1s;
}
.tag-remove:hover { opacity: 1; }
.input-error {
  margin-top: 6px;
  font-size: 12px;
  color: var(--red);
}
.extract-note {
  margin-top: 8px;
  font-size: 11px;
  color: var(--text-tertiary);
  line-height: 1.5;
}

/* ── Extract bar ── */
.extract-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--tag-bg);
  border-radius: var(--radius);
  padding: 8px 12px;
  margin-top: 10px;
  flex-wrap: wrap;
}
.extract-label {
  font-size: 12px;
  color: var(--text-secondary);
  flex: 1;
  min-width: 100px;
}
.extract-select {
  font-family: var(--sans);
  font-size: 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px 8px;
  background: var(--surface);
  color: var(--text);
  outline: none;
  cursor: pointer;
}
.btn-extract {
  font-family: var(--sans);
  font-size: 12px;
  font-weight: 500;
  padding: 5px 12px;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  transition: all .15s;
  white-space: nowrap;
}
.btn-extract:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
  background: var(--accent-light);
}
.btn-extract:disabled { opacity: .4; cursor: not-allowed; }

/* ── Methods ── */
.methods-section {
  padding: 14px 20px;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}
.methods-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 5px;
  margin-top: 8px;
}
.method-item {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 7px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: all .15s;
  user-select: none;
}
.method-item input { display: none; }
.method-item:hover { border-color: var(--accent); background: var(--accent-light); }
.method-item.checked { border-color: var(--accent); background: var(--accent-light); }
.check-box {
  width: 14px;
  height: 14px;
  border-radius: 3px;
  border: 1.5px solid var(--border-strong);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: all .15s;
}
.method-item.checked .check-box {
  background: var(--accent);
  border-color: var(--accent);
}
.check-icon { color: white; font-size: 10px; font-weight: 700; line-height: 1; }
.method-label { font-size: 12px; font-weight: 500; color: var(--text); flex: 1; }
.method-badge {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-tertiary);
}

/* ── Top-K ── */
.topk-section {
  padding: 12px 20px;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}
.topk-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 8px;
}
.topk-slider {
  flex: 1;
  accent-color: var(--accent);
  cursor: pointer;
}
.topk-value {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  min-width: 20px;
  text-align: right;
}

/* ── Run ── */
.run-section {
  padding: 14px 20px 20px;
  flex-shrink: 0;
}
.btn-run {
  width: 100%;
  padding: 11px 16px;
  border-radius: var(--radius);
  border: none;
  background: var(--accent);
  color: white;
  font-family: var(--sans);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  letter-spacing: -0.01em;
  transition: all .15s;
}
.btn-run:hover:not(:disabled) {
  background: var(--accent-hover);
  transform: translateY(-1px);
  box-shadow: 0 4px 14px rgba(45,91,227,.3);
}
.btn-run:disabled {
  background: var(--border-strong);
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}
.run-hint {
  margin-top: 8px;
  font-size: 11px;
  color: var(--text-tertiary);
  text-align: center;
}
.spinner {
  width: 15px;
  height: 15px;
  border: 2px solid rgba(255,255,255,.35);
  border-top-color: white;
  border-radius: 50%;
  animation: spin .7s linear infinite;
  flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Phenotype search ── */
.search-section {
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.search-wrap {
  margin-top: 8px;
  position: relative;
}
.search-input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-family: var(--sans);
  font-size: 13px;
  background: var(--bg);
  color: var(--text);
  outline: none;
  transition: border .15s;
}
.search-input:focus {
  border-color: var(--accent);
  background: var(--surface);
}
.search-results {
  margin-top: 6px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  max-height: 220px;
  overflow-y: auto;
}
.search-result-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  gap: 8px;
}
.search-result-row:last-child { border-bottom: none; }
.search-result-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.search-result-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.search-result-id {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-tertiary);
}
.search-result-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}
.btn-include, .btn-exclude {
  font-size: 11px;
  font-weight: 500;
  padding: 3px 8px;
  border-radius: 99px;
  border: 1px solid;
  cursor: pointer;
  transition: all .15s;
  white-space: nowrap;
}
.btn-include {
  background: var(--green-light);
  color: var(--green);
  border-color: #BBDFD2;
}
.btn-include:hover { background: var(--green); color: white; }
.btn-exclude {
  background: var(--red-light);
  color: var(--red);
  border-color: #F0C0BB;
}
.btn-exclude:hover { background: var(--red); color: white; }
.tag-excluded {
  background: var(--red-light);
  color: var(--red);
  border: 1px solid #F0C0BB;
  flex-direction: column;
  align-items: flex-start;
  gap: 1px;
  padding: 4px 10px;
  border-radius: 8px;
  display: inline-flex;
}
.method-note {
  font-size: 10px;
  color: var(--text-tertiary);
  width: 100%;
  margin-top: 2px;
}
</style>

