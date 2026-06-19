<script setup>
/*
  MethodComparison.vue — single-case, per-method comparison.

  Renders the `comparison` block returned by POST /api/diagnose:
    { k, methods, short_names, categories, by_method, consensus, agreement, top_candidate }

  No ground truth is assumed — every view speaks to agreement/consensus only.
  Consensus is re-fused client-side from `by_method` so toggling a method off
  recomputes instantly (the RRF formula below mirrors core.method_comparison).
*/
import { ref, computed } from 'vue'

const props = defineProps({
  comparison: { type: Object, required: true },
  caseId: { type: String, default: '' },
  inputHpo: { type: Array, default: () => [] },   // [{id,label}]
})

const RRF_K = 60
const view = ref('consensus')
const selected = ref(null)
const disabled = ref(new Set())

const allMethods = computed(() => props.comparison.methods)
const activeMethods = computed(() => allMethods.value.filter(m => !disabled.value.has(m)))
const sname = (m) => props.comparison.short_names[m] || m
const cat = (m) => props.comparison.categories[m] || ''

function toggle(m) {
  const next = new Set(disabled.value)
  next.has(m) ? next.delete(m) : next.add(m)
  if (next.size < allMethods.value.length) disabled.value = next   // keep >=1 active
}
function pick(id) { selected.value = selected.value === id ? null : id }

/* client-side RRF over active methods (instant re-fuse on toggle) */
const consensus = computed(() => {
  const bm = props.comparison.by_method
  const fused = {}, support = {}, ranks = {}, labels = {}
  for (const m of activeMethods.value) {
    for (const it of bm[m] || []) {
      fused[it.disease_id] = (fused[it.disease_id] || 0) + 1 / (RRF_K + it.rank)
      ;(support[it.disease_id] ||= new Set()).add(m)
      ;(ranks[it.disease_id] ||= {})[m] = it.rank
      labels[it.disease_id] ??= it.label
    }
  }
  return Object.keys(fused).map(id => {
    const rk = Object.values(ranks[id])
    return {
      disease_id: id, label: labels[id], rrf_score: fused[id],
      method_ranks: ranks[id], n_methods: support[id].size,
      best_rank: Math.min(...rk), worst_rank: Math.max(...rk),
    }
  }).sort((a, b) => b.rrf_score - a.rrf_score || a.best_rank - b.best_rank)
})

/* stable colour per disease, by consensus order */
const PALETTE = ['#3a52d6','#0f9d8c','#d6603a','#8a4fd6','#c0392b','#2d8a4e','#b8860b',
                 '#1f7a99','#a63a78','#5a6b7a','#7a5a3a','#3a7ad6','#9c4f2a','#2a9c7a','#6a3a9c','#999']
const colorOf = computed(() => {
  const map = {}
  consensus.value.forEach((d, i) => { map[d.disease_id] = PALETTE[i % PALETTE.length] })
  return map
})

const maxRank = computed(() => props.comparison.k || 10)
function heat(rank) {
  const t = Math.min(1, (rank - 1) / Math.max(1, maxRank.value - 1))
  return `hsl(231 ${62 - t * 30}% ${34 + t * 40}%)`
}
function heatText(rank) {
  const t = Math.min(1, (rank - 1) / Math.max(1, maxRank.value - 1))
  return (34 + t * 40) > 62 ? '#1a2030' : '#fff'
}

/* agreement: filter the precomputed Jaccard matrix to active methods */
const agree = computed(() => {
  const a = props.comparison.agreement
  const keep = a.methods_ordered.map((m, i) => ({ m, i })).filter(x => !disabled.value.has(x.m))
  const idx = keep.map(x => x.i)
  return {
    methods: keep.map(x => x.m),
    matrix: idx.map(i => idx.map(j => a.matrix[i][j])),
  }
})
function simColor(v) { return `hsl(174 ${30 + v * 45}% ${88 - v * 52}%)` }
function simText(v) { return v > 0.45 ? '#fff' : '#1a2030' }

const totalRrf = (d) => d.rrf_score.toFixed(4)
</script>

<template>
  <div class="mc">
    <header class="mc-head">
      <div class="mc-title">Method comparison <span>· {{ activeMethods.length }}/{{ allMethods.length }} methods</span></div>
      <span v-if="caseId" class="mc-case">{{ caseId }}</span>
    </header>

    <div v-if="inputHpo.length" class="mc-hpo">
      <span v-for="h in inputHpo" :key="h.id" class="hpo"><b>{{ h.label }}</b> {{ h.id }}</span>
    </div>

    <div class="mc-methods">
      <button v-for="m in allMethods" :key="m" class="mtog" :class="{ off: disabled.has(m) }"
              :aria-pressed="!disabled.has(m)" @click="toggle(m)">
        <span class="dot" :style="{ background: disabled.has(m) ? '#bbb' : '#3a52d6' }"></span>
        {{ sname(m) }}<span class="cat">{{ cat(m) }}</span>
      </button>
    </div>

    <div class="mc-tabs" role="tablist">
      <button v-for="t in ['consensus','side','grid','agree']" :key="t" class="tab"
              :aria-selected="view === t" @click="view = t">
        {{ { consensus:'Consensus', side:'Side-by-side', grid:'Grid', agree:'Agreement' }[t] }}
      </button>
    </div>

    <div class="mc-selbar">
      <template v-if="selected">
        <span class="sw" :style="{ background: colorOf[selected] }"></span>
        Tracking <b>{{ consensus.find(d => d.disease_id === selected)?.label || selected }}</b>
        <button class="clear" @click="pick(selected)">clear</button>
      </template>
      <template v-else>Click any disease to track it across every view.</template>
    </div>

    <!-- CONSENSUS -->
    <section v-if="view === 'consensus'">
      <p class="sub">Methods fused by Reciprocal Rank Fusion. The meter shows each method's rank (darker = higher); a full dark row means broad agreement.</p>
      <div v-for="(d, i) in consensus" :key="d.disease_id" class="crow" :class="{ sel: selected === d.disease_id }" @click="pick(d.disease_id)">
        <span class="crank">{{ i + 1 }}</span>
        <span class="cname"><span class="sw" :style="{ background: colorOf[d.disease_id] }"></span>
          {{ d.label }} <span class="orpha">{{ d.disease_id }}</span></span>
        <span class="support">
          <span v-for="m in activeMethods" :key="m" class="seg"
                :style="{ background: d.method_ranks[m] ? heat(d.method_ranks[m]) : '#f1f3f6' }"
                :title="sname(m) + ': ' + (d.method_ranks[m] ? 'rank ' + d.method_ranks[m] : 'absent')"></span>
          <span class="cnt">{{ d.n_methods }}/{{ activeMethods.length }}</span>
        </span>
      </div>
    </section>

    <!-- SIDE BY SIDE -->
    <section v-else-if="view === 'side'">
      <p class="sub">Each column is one method's ranked list; a disease keeps its colour everywhere.</p>
      <div class="cols">
        <div v-for="m in activeMethods" :key="m" class="col">
          <h3>{{ sname(m) }}</h3>
          <div v-for="it in comparison.by_method[m]" :key="it.disease_id" class="cell"
               :class="{ sel: selected === it.disease_id }" :title="'score ' + it.score" @click="pick(it.disease_id)">
            <span class="r">{{ it.rank }}</span>
            <span class="sw sm" :style="{ background: colorOf[it.disease_id] || '#ccc' }"></span>
            <span class="nm">{{ it.label }}</span>
          </div>
        </div>
      </div>
    </section>

    <!-- GRID -->
    <section v-else-if="view === 'grid'">
      <p class="sub">Cell = rank (darker = higher); blank = method didn't surface it. Rows sorted by consensus.</p>
      <div class="gridwrap">
        <table class="grid">
          <thead><tr><th class="dz">disease</th><th v-for="m in activeMethods" :key="m"><div class="rot">{{ sname(m) }}</div></th></tr></thead>
          <tbody>
            <tr v-for="d in consensus" :key="d.disease_id" :class="{ sel: selected === d.disease_id }">
              <td><span class="grow" @click="pick(d.disease_id)"><span class="sw" :style="{ background: colorOf[d.disease_id] }"></span>{{ d.label }}</span></td>
              <td v-for="m in activeMethods" :key="m">
                <div v-if="d.method_ranks[m]" class="gcell" :style="{ background: heat(d.method_ranks[m]), color: heatText(d.method_ranks[m]) }"
                     :title="sname(m) + ' · rank ' + d.method_ranks[m]" @click="pick(d.disease_id)">{{ d.method_ranks[m] }}</div>
                <div v-else class="gcell empty">·</div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- AGREEMENT -->
    <section v-else>
      <p class="sub">Pairwise top-{{ comparison.k }} overlap (Jaccard). Darker = the two methods rank alike — same metric as the offline figure.</p>
      <div class="gridwrap">
        <table class="rbo">
          <thead><tr><th></th><th v-for="m in agree.methods" :key="m" class="top"><div>{{ sname(m) }}</div></th></tr></thead>
          <tbody>
            <tr v-for="(m, i) in agree.methods" :key="m">
              <th>{{ sname(m) }}</th>
              <td v-for="(m2, j) in agree.methods" :key="m2">
                <div class="rcell" :class="{ diag: i === j }"
                     :style="i === j ? {} : { background: simColor(agree.matrix[i][j]), color: simText(agree.matrix[i][j]) }"
                     :title="sname(m) + ' ↔ ' + sname(m2) + ': ' + agree.matrix[i][j].toFixed(2)">
                  {{ i === j ? '–' : agree.matrix[i][j].toFixed(2) }}
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.mc { font-family: ui-sans-serif, system-ui, sans-serif; color: #161a20; font-size: 14px; }
.mc-head { display: flex; justify-content: space-between; align-items: baseline; }
.mc-title { font-weight: 650; }
.mc-title span { color: #6b7280; font-weight: 400; }
.mc-case { font-family: ui-monospace, monospace; font-size: 12px; color: #6b7280; }
.mc-hpo { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
.hpo { font-family: ui-monospace, monospace; font-size: 11.5px; background: #e9ecfb; color: #2438a8; padding: 3px 8px; border-radius: 6px; }
.mc-methods { display: flex; gap: 7px; flex-wrap: wrap; margin: 12px 0; }
.mtog { font-family: ui-monospace, monospace; font-size: 11px; padding: 4px 9px; border-radius: 999px; border: 1px solid #dce1e9; background: #fff; cursor: pointer; display: flex; align-items: center; gap: 6px; }
.mtog.off { opacity: .42; text-decoration: line-through; }
.mtog .dot { width: 7px; height: 7px; border-radius: 50%; }
.mtog .cat { color: #8b919d; }
.mc-tabs { display: flex; gap: 4px; background: #fff; border: 1px solid #dce1e9; border-radius: 10px; padding: 5px; width: fit-content; }
.tab { font-size: 13px; font-weight: 550; padding: 7px 14px; border-radius: 7px; border: none; background: none; cursor: pointer; color: #6b7280; }
.tab[aria-selected="true"] { background: #3a52d6; color: #fff; }
.sub { font-size: 12.5px; color: #6b7280; margin: 12px 0 14px; }
.mc-selbar { display: flex; align-items: center; gap: 10px; font-size: 12.5px; color: #6b7280; min-height: 22px; margin-top: 12px; }
.mc-selbar b { color: #161a20; }
.clear { font-size: 12px; border: 1px solid #dce1e9; background: #fff; border-radius: 6px; padding: 2px 9px; cursor: pointer; color: #6b7280; }
.sw { width: 10px; height: 10px; border-radius: 3px; display: inline-block; flex: none; }
.sw.sm { width: 8px; height: 8px; }
.crow { display: grid; grid-template-columns: 30px 1fr auto; gap: 12px; align-items: center; padding: 9px 8px; border-radius: 8px; cursor: pointer; border: 1px solid transparent; }
.crow:hover { background: #fafbfc; }
.crow.sel { background: #e9ecfb; border-color: #c9d0f6; }
.crank { font-family: ui-monospace, monospace; font-size: 13px; color: #6b7280; text-align: right; }
.cname { font-weight: 550; display: flex; align-items: center; gap: 9px; min-width: 0; }
.orpha { font-family: ui-monospace, monospace; font-size: 11px; color: #8b919d; font-weight: 400; }
.support { display: flex; gap: 2px; align-items: center; }
.seg { width: 11px; height: 9px; border-radius: 2px; }
.cnt { font-family: ui-monospace, monospace; font-size: 11px; color: #6b7280; margin-left: 7px; }
.cols { display: grid; grid-auto-flow: column; grid-auto-columns: 150px; gap: 9px; overflow-x: auto; padding-bottom: 6px; }
.col h3 { font-family: ui-monospace, monospace; font-size: 10.5px; margin: 0 0 9px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cell { display: flex; align-items: center; gap: 6px; padding: 5px 6px; border-radius: 6px; margin-bottom: 4px; cursor: pointer; border: 1px solid transparent; font-size: 12px; }
.cell:hover { background: #fafbfc; }
.cell.sel { outline: 2px solid #f5a623; outline-offset: -1px; background: #fffaf0; }
.cell .r { font-family: ui-monospace, monospace; font-size: 11px; color: #6b7280; width: 13px; }
.cell .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.gridwrap { overflow-x: auto; }
.grid { border-collapse: collapse; font-size: 12px; }
.grid th { font-weight: 500; color: #6b7280; padding: 4px 5px; font-family: ui-monospace, monospace; font-size: 10px; vertical-align: bottom; height: 92px; }
.grid th.dz { text-align: left; height: auto; vertical-align: middle; }
.rot { writing-mode: vertical-rl; transform: rotate(180deg); margin: 0 auto; }
.grid td { padding: 0; text-align: center; }
.gcell { margin: 2px; height: 28px; min-width: 30px; border-radius: 5px; display: flex; align-items: center; justify-content: center; font-family: ui-monospace, monospace; font-size: 11.5px; font-weight: 600; cursor: pointer; }
.gcell.empty { background: #f1f3f6; color: #8b919d; font-weight: 400; }
.grow { display: inline-flex; align-items: center; gap: 8px; padding: 4px 10px 4px 4px; cursor: pointer; white-space: nowrap; font-weight: 500; }
tr.sel .grow { color: #2438a8; font-weight: 650; }
tr.sel .gcell:not(.empty) { outline: 2px solid #f5a623; outline-offset: -2px; }
.rbo { border-collapse: separate; border-spacing: 2px; font-size: 11px; }
.rbo th { font-family: ui-monospace, monospace; font-size: 9.5px; color: #6b7280; font-weight: 500; padding: 2px; text-align: right; white-space: nowrap; }
.rbo th.top { text-align: center; height: 84px; vertical-align: bottom; }
.rbo th.top div { writing-mode: vertical-rl; transform: rotate(180deg); margin: 0 auto; }
.rcell { width: 40px; height: 30px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-family: ui-monospace, monospace; font-weight: 600; }
.rcell.diag { background: #eef1f4; color: #8b919d; }
</style>
