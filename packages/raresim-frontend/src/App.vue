<template>
  <div class="app-root">

    <!-- ── Header ── -->
    <header class="app-header">
      <div class="logo">
        <div class="logo-dot" />
        RareSim
      </div>
      <div class="header-right">
        <span class="version">v0.1</span>
      </div>
    </header>

    <!-- ── Body ── -->
    <div class="app-body">
      <InputPanel
        :running="diagState === 'loading'"
        @run="handleRun"
      />
      <ResultsPanel
        :state="diagState"
        :results="results"
        :meta="meta"
        :error-message="errorMessage"
        :running-methods="lastPayload?.methods || []"
        :n-diseases="meta.n_diseases || 0"
        @retry="handleRetry"
      />
    </div>

  </div>
</template>

<script setup>
import { ref } from 'vue'
import InputPanel from './components/InputPanel.vue'
import ResultsPanel from './components/ResultsPanel.vue'
import { diagnose } from './api/index.js'

const diagState    = ref('idle')   // idle | loading | done | error
const results      = ref([])
const meta         = ref({})
const errorMessage = ref('')
const lastPayload  = ref(null)

async function handleRun(payload) {
  lastPayload.value = payload
  diagState.value   = 'loading'
  results.value     = []
  errorMessage.value = ''

  try {
    const data = await diagnose(payload)
    results.value = data.results
    meta.value    = data.meta || {}
    diagState.value = 'done'
  } catch (e) {
    errorMessage.value = e.message
    diagState.value    = 'error'
  }
}

function handleRetry() {
  if (lastPayload.value) handleRun(lastPayload.value)
}
</script>

<style>
/* ── Global CSS variables ── */
:root {
  --bg:            #F4F3EF;
  --surface:       #FFFFFF;
  --border:        #E2E0D8;
  --border-strong: #C8C5BA;
  --text:          #1A1916;
  --text-secondary:#6B6860;
  --text-tertiary: #9E9C96;
  --accent:        #2D5BE3;
  --accent-light:  #EEF2FD;
  --accent-hover:  #1E47CC;
  --green:         #1A7F5A;
  --green-light:   #EBF7F2;
  --red:           #C0392B;
  --red-light:     #FDF0EF;
  --tag-bg:        #EEECEA;
  --mono:          'DM Mono', monospace;
  --sans:          'DM Sans', sans-serif;
  --radius:        8px;
  --radius-lg:     12px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
}

/* ── App layout ── */
.app-root {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

/* ── Header ── */
.app-header {
  height: 52px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  flex-shrink: 0;
  position: relative;
  z-index: 10;
}
.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--mono);
  font-size: 15px;
  font-weight: 500;
  letter-spacing: -0.02em;
  color: var(--text);
}
.logo-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent);
}
.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}
.version {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-tertiary);
  background: var(--tag-bg);
  padding: 2px 8px;
  border-radius: 99px;
}

/* ── Body ── */
.app-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}
</style>
