import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { testingApi } from '../../api/client'
import { useScreenRecorder } from '../../hooks/useScreenRecorder'
import {
  Zap, Play, Square, RefreshCw, Video, Download, Trophy, Clock,
  Cpu, Monitor, Save, Check, Circle, ImageDown,
} from 'lucide-react'

// ── Types ────────────────────────────────────────────────────────────────

interface SuitePrompt {
  id: string
  label: string
  prompt: string
  max_tokens: number
}

interface PromptResult {
  prompt_id: string
  label: string
  tokens_generated: number
  ttft_ms: number
  gen_time_ms: number
  tokens_per_second: number
  error: string | null
}

interface ModelResult {
  path: string
  name: string
  status: string
  error: string | null
  size_gb: number
  architecture: string
  quantization: string
  parameters: string
  n_ctx: number | null
  device: string
  load_ms: number
  prompts: PromptResult[]
  prompt_count: number
  avg_tps: number
  avg_ttft_ms: number
  total_tokens: number
  gen_time_ms: number
}

interface SuiteStatus {
  running: boolean
  run_id?: string
  device?: string
  status?: string
  duration_ms?: number
  error?: string | null
  current_model?: string
  current_prompt?: string
  progress?: number
  models?: ModelResult[]
}

interface CandidateModel {
  path: string
  filename: string
  size_gb: number
  size_human: string
  architecture: string
  quantization: string
  family: string
}

interface PastRun {
  run_id: string
  status: string
  device: string
  started_at: number
  duration_ms: number
  model_count: number
  best_tps: number
}

const BAR_COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#a78bfa', '#f87171', '#22d3ee', '#fb923c']

function downloadUrl(url: string, filename: string) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
}

function statusColor(status: string) {
  if (status === 'ok') return 'text-emerald-400'
  if (status === 'error') return 'text-red-400'
  if (status === 'cancelled') return 'text-amber-400'
  if (status === 'loading' || status === 'running') return 'text-accent'
  return 'text-text-muted'
}

// ── Component ────────────────────────────────────────────────────────────

export function TestingStudioPanel() {
  const [prompts, setPrompts] = useState<SuitePrompt[]>([])
  const [candidates, setCandidates] = useState<CandidateModel[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [promptIds, setPromptIds] = useState<string[]>([])
  const [device, setDevice] = useState<'cpu' | 'gpu'>('cpu')
  const [maxTokens, setMaxTokens] = useState('')
  const [status, setStatus] = useState<SuiteStatus | null>(null)
  const [pastRuns, setPastRuns] = useState<PastRun[]>([])
  const [busy, setBusy] = useState(false)
  const [loadingModels, setLoadingModels] = useState(true)
  const [notice, setNotice] = useState<string | null>(null)
  const [showHud, setShowHud] = useState(true)
  const recorder = useScreenRecorder()
  const pollRef = useRef<number | null>(null)

  // ── Data loading ───────────────────────────────────────────────────────

  const loadPrompts = useCallback(async () => {
    try {
      const p = await testingApi.prompts()
      setPrompts(p)
      setPromptIds(p.map((x: SuitePrompt) => x.id))
    } catch { /* ok */ }
  }, [])

  const loadModels = useCallback(async () => {
    setLoadingModels(true)
    try {
      const data = await testingApi.models()
      setCandidates(data.models || [])
    } catch { /* ok */ }
    setLoadingModels(false)
  }, [])

  const loadPastRuns = useCallback(async () => {
    try {
      setPastRuns(await testingApi.runs())
    } catch { /* ok */ }
  }, [])

  const refreshStatus = useCallback(async () => {
    try {
      const s = await testingApi.status()
      setStatus(s)
      return s as SuiteStatus
    } catch {
      return null
    }
  }, [])

  useEffect(() => {
    void loadPrompts()
    void loadModels()
    void loadPastRuns()
    void refreshStatus()
  }, [loadPrompts, loadModels, loadPastRuns, refreshStatus])

  // Poll while a run is live.
  useEffect(() => {
    if (!status?.running) {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current)
        pollRef.current = null
      }
      return
    }
    if (pollRef.current === null) {
      pollRef.current = window.setInterval(() => { void refreshStatus() }, 700)
    }
    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current)
        pollRef.current = null
      }
    }
  }, [status?.running, refreshStatus])

  // ── Actions ────────────────────────────────────────────────────────────

  const toggleModel = (path: string) => {
    setSelected((prev) => (prev.includes(path) ? prev.filter((p) => p !== path) : [...prev, path]))
  }

  const startRun = async () => {
    if (selected.length === 0) {
      setNotice('Select at least one model first.')
      return
    }
    setBusy(true)
    setNotice(null)
    try {
      const mt = maxTokens.trim() ? Number(maxTokens) : null
      await testingApi.run({
        paths: selected,
        device,
        prompt_ids: promptIds,
        max_tokens: mt && mt > 0 ? mt : null,
        n_ctx: null,
      })
      await refreshStatus()
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Could not start the test run.')
    }
    setBusy(false)
  }

  const stopRun = async () => {
    try {
      await testingApi.cancel()
    } catch { /* ok */ }
  }

  const models = useMemo(() => status?.models ?? [], [status])
  const ranked = useMemo(
    () => models.filter((m) => m.prompts && m.prompts.length > 0).sort((a, b) => b.avg_tps - a.avg_tps),
    [models],
  )
  const maxTps = useMemo(() => Math.max(1, ...ranked.map((m) => m.avg_tps)), [ranked])
  const fastest = ranked[0]
  const finished = !!status && !status.running && (status.status === 'done' || status.status === 'cancelled')

  const progressPct = Math.round((status?.progress ?? 0) * 100)
  const recSeconds = Math.floor(recorder.durationMs / 1000)

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="h-full flex flex-col overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border">
        <h3 className="text-sm font-semibold text-text flex items-center gap-2">
          <Zap size={14} className="text-accent" />
          Model Testing Studio
        </h3>
        <div className="flex items-center gap-1">
          <button
            onClick={() => { void loadModels(); void loadPastRuns() }}
            title="Refresh models"
            className="p-1.5 text-text-muted hover:text-text rounded hover:bg-elevated transition-colors"
          >
            <RefreshCw size={13} className={loadingModels ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {notice && (
        <div className="mx-3 mt-2 text-[11px] text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded px-2 py-1.5">
          {notice}
        </div>
      )}

      {/* Run controls */}
      <div className="px-3 py-2 space-y-2 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-text-muted uppercase tracking-wide">Run on</span>
          <div className="flex gap-1 bg-elevated rounded-lg p-0.5">
            <button
              onClick={() => setDevice('cpu')}
              disabled={status?.running}
              className={`flex items-center gap-1 text-[11px] px-2 py-1 rounded-md transition-colors ${device === 'cpu' ? 'bg-accent/15 text-accent font-medium' : 'text-text-muted hover:text-text'}`}
            >
              <Cpu size={11} /> CPU
            </button>
            <button
              onClick={() => setDevice('gpu')}
              disabled={status?.running}
              className={`flex items-center gap-1 text-[11px] px-2 py-1 rounded-md transition-colors ${device === 'gpu' ? 'bg-accent/15 text-accent font-medium' : 'text-text-muted hover:text-text'}`}
            >
              <Monitor size={11} /> GPU
            </button>
          </div>
          <input
            value={maxTokens}
            onChange={(e) => setMaxTokens(e.target.value.replace(/[^0-9]/g, ''))}
            placeholder="max tokens"
            disabled={status?.running}
            className="ml-auto w-20 bg-elevated border border-border rounded px-2 py-1 text-[11px] text-text placeholder:text-text-muted focus:outline-none focus:border-accent/50"
          />
        </div>

        {/* Prompt selection */}
        {prompts.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {prompts.map((p) => {
              const on = promptIds.includes(p.id)
              return (
                <button
                  key={p.id}
                  onClick={() => setPromptIds((prev) => (on ? prev.filter((x) => x !== p.id) : [...prev, p.id]))}
                  disabled={status?.running}
                  title={p.prompt}
                  className={`text-[10px] px-2 py-0.5 rounded-full border transition-colors ${
                    on ? 'border-accent/50 text-accent bg-accent/10' : 'border-border text-text-muted'
                  }`}
                >
                  {p.label}
                </button>
              )
            })}
          </div>
        )}

        <div className="flex items-center gap-2">
          {!status?.running ? (
            <button
              onClick={startRun}
              disabled={busy || selected.length === 0}
              className="flex-1 flex items-center justify-center gap-1.5 text-[11px] font-medium py-2 rounded-lg bg-accent/15 text-accent hover:bg-accent/25 disabled:opacity-40 transition-colors"
            >
              {busy ? <RefreshCw size={13} className="animate-spin" /> : <Play size={13} />}
              Run test ({selected.length})
            </button>
          ) : (
            <button
              onClick={stopRun}
              className="flex-1 flex items-center justify-center gap-1.5 text-[11px] font-medium py-2 rounded-lg bg-red-500/15 text-red-400 hover:bg-red-500/25 transition-colors"
            >
              <Square size={13} /> Stop
            </button>
          )}
        </div>

        {/* Progress */}
        {status?.running && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[10px] text-text-muted">
              <span className="truncate">
                {status.current_model || 'starting…'}
                {status.current_prompt ? ` · ${status.current_prompt}` : ''}
              </span>
              <span>{progressPct}%</span>
            </div>
            <div className="h-1.5 bg-bg rounded-full overflow-hidden">
              <div className="h-full bg-accent rounded-full transition-all duration-300" style={{ width: `${progressPct}%` }} />
            </div>
          </div>
        )}
      </div>

      {/* Recording */}
      <div className="px-3 py-2 border-b border-border space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-text-muted uppercase tracking-wide flex items-center gap-1">
            <Video size={11} /> Screen recording
          </span>
          {recorder.recording && (
            <span className="text-[10px] text-red-400 flex items-center gap-1">
              <Circle size={8} className="fill-red-500 text-red-500 animate-pulse" />
              REC {String(Math.floor(recSeconds / 60)).padStart(2, '0')}:{String(recSeconds % 60).padStart(2, '0')}
            </span>
          )}
        </div>

        {!recorder.supported ? (
          <div className="text-[10px] text-text-muted">Not supported in this environment.</div>
        ) : (
          <>
            <div className="flex gap-1.5">
              {!recorder.recording ? (
                <button
                  onClick={() => void recorder.start()}
                  className="flex-1 flex items-center justify-center gap-1.5 text-[11px] py-1.5 rounded-lg bg-elevated border border-border text-text hover:border-accent/40 transition-colors"
                >
                  <Video size={12} /> Record screen
                </button>
              ) : (
                <button
                  onClick={recorder.stop}
                  className="flex-1 flex items-center justify-center gap-1.5 text-[11px] py-1.5 rounded-lg bg-red-500/15 text-red-400 transition-colors"
                >
                  <Square size={12} /> Stop recording
                </button>
              )}
            </div>

            <label className="flex items-center gap-2 text-[10px] text-text-muted cursor-pointer">
              <input
                type="checkbox"
                checked={showHud}
                onChange={(e) => setShowHud(e.target.checked)}
                className="accent-accent"
              />
              Show live stats overlay in the recording
            </label>

            {recorder.error && <div className="text-[10px] text-amber-400">{recorder.error}</div>}

            {recorder.blob && (
              <div className="flex gap-1.5">
                <button
                  onClick={() => downloadUrl(recorder.videoUrl!, `gguf-speed-test-${status?.run_id || Date.now()}.webm`)}
                  className="flex-1 flex items-center justify-center gap-1.5 text-[11px] py-1.5 rounded-lg bg-elevated border border-border text-text hover:border-accent/40 transition-colors"
                >
                  <Download size={12} /> Download .webm
                </button>
                <button
                  onClick={() => void recorder.save(`gguf-speed-test-${status?.run_id || Date.now()}.webm`)}
                  title="Save to disk (Electron)"
                  className="flex items-center justify-center gap-1.5 text-[11px] px-2.5 py-1.5 rounded-lg bg-elevated border border-border text-text hover:border-accent/40 transition-colors"
                >
                  <Save size={12} />
                </button>
              </div>
            )}
            <div className="text-[9px] text-text-muted">
              Tip: choose “Entire screen” or this window when the picker opens so the speed test is visible.
            </div>
          </>
        )}
      </div>

      {/* Live leaderboard */}
      <div className="px-3 py-2 border-b border-border space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-text-muted uppercase tracking-wide flex items-center gap-1">
            <Trophy size={11} /> Live leaderboard
          </span>
          {fastest && <span className="text-[10px] text-emerald-400">{fastest.name}: {fastest.avg_tps.toFixed(1)} tok/s</span>}
        </div>

        {ranked.length === 0 ? (
          <div className="text-[10px] text-text-muted py-3 text-center">
            No measurements yet. Select models and press Run.
          </div>
        ) : (
          <div className="space-y-2">
            {ranked.map((m, i) => (
              <div key={m.path} className="space-y-1">
                <div className="flex items-center gap-2 text-[10px]">
                  <span className="text-text-muted w-4">{i + 1}.</span>
                  <span className="text-text flex-1 truncate">{m.name}</span>
                  <span className="text-text-sec">{m.avg_tps.toFixed(1)} tok/s</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-elevated rounded-full h-3 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${(m.avg_tps / maxTps) * 100}%`, backgroundColor: BAR_COLORS[i % BAR_COLORS.length], minWidth: m.avg_tps > 0 ? '4px' : '0' }}
                    />
                  </div>
                  <span className="text-[9px] text-text-muted w-16 text-right">{m.avg_ttft_ms.toFixed(0)}ms TTFT</span>
                </div>
                <div className="flex gap-3 text-[9px] text-text-muted pl-6">
                  <span className="flex items-center gap-0.5"><Clock size={9} /> load {m.load_ms.toFixed(0)}ms</span>
                  <span>{m.total_tokens} tokens</span>
                  {m.quantization && <span>{m.quantization}</span>}
                  {m.n_ctx && <span>ctx {m.n_ctx}</span>}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Per-model status incl. failures */}
        {models.filter((m) => m.status !== 'ok').length > 0 && (
          <div className="space-y-1 pt-1">
            {models.filter((m) => m.status !== 'ok').map((m) => (
              <div key={m.path} className="flex items-center gap-2 text-[10px]">
                <span className={statusColor(m.status)}>{m.status}</span>
                <span className="text-text-muted truncate flex-1">{m.name}</span>
                {m.error && <span className="text-red-400/80 truncate max-w-[45%]" title={m.error}>{m.error}</span>}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Export + history */}
      {finished && status?.run_id && (
        <div className="px-3 py-2 border-b border-border space-y-2">
          <span className="text-[10px] text-text-muted uppercase tracking-wide flex items-center gap-1">
            <ImageDown size={11} /> Share this run
          </span>
          <div className="flex gap-1.5">
            {(['json', 'markdown', 'csv'] as const).map((fmt) => (
              <button
                key={fmt}
                onClick={() => downloadUrl(testingApi.exportUrl(status.run_id!, fmt), `gguf-speed-test-${status.run_id}.${fmt === 'markdown' ? 'md' : fmt}`)}
                className="flex-1 text-[10px] py-1.5 rounded-lg bg-elevated border border-border text-text hover:border-accent/40 transition-colors"
              >
                {fmt.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Model picker */}
      <div className="px-3 py-2 border-b border-border space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-text-muted uppercase tracking-wide">Models on disk ({candidates.length})</span>
          <div className="flex gap-2">
            <button onClick={() => setSelected(candidates.map((c) => c.path))} className="text-[10px] text-accent hover:underline">All</button>
            <button onClick={() => setSelected([])} className="text-[10px] text-text-muted hover:underline">None</button>
          </div>
        </div>
        {loadingModels ? (
          <div className="text-[10px] text-text-muted py-3 text-center">Scanning for GGUF models…</div>
        ) : candidates.length === 0 ? (
          <div className="text-[10px] text-text-muted py-3 text-center">
            No .gguf files found. Drop models into your models folder, then refresh.
          </div>
        ) : (
          <div className="space-y-1">
            {candidates.map((c) => {
              const on = selected.includes(c.path)
              return (
                <label key={c.path} className="flex items-center gap-2 cursor-pointer group">
                  <span className={`w-3.5 h-3.5 rounded border flex items-center justify-center ${on ? 'bg-accent/20 border-accent/50' : 'border-border'}`}>
                    {on && <Check size={10} className="text-accent" />}
                  </span>
                  <input type="checkbox" className="hidden" checked={on} onChange={() => toggleModel(c.path)} disabled={status?.running} />
                  <span className="text-[11px] text-text truncate flex-1 group-hover:text-accent transition-colors" title={c.path}>
                    {c.filename}
                  </span>
                  <span className="text-[9px] text-text-muted">{c.size_human}</span>
                </label>
              )
            })}
          </div>
        )}
      </div>

      {/* Past runs */}
      {pastRuns.length > 0 && (
        <div className="px-3 py-2 space-y-1.5">
          <span className="text-[10px] text-text-muted uppercase tracking-wide">Past runs</span>
          {pastRuns.slice(0, 8).map((r) => (
            <button
              key={r.run_id}
              onClick={() => downloadUrl(testingApi.exportUrl(r.run_id, 'json'), `gguf-speed-test-${r.run_id}.json`)}
              className="w-full text-left bg-elevated border border-border rounded-lg px-2.5 py-1.5 hover:border-accent/30 transition-colors"
            >
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-text truncate">{r.model_count} models · {r.device.toUpperCase()}</span>
                <span className="text-accent">{r.best_tps.toFixed(1)} tok/s</span>
              </div>
              <div className="text-[9px] text-text-muted">
                {r.started_at ? new Date(r.started_at * 1000).toLocaleString() : ''} · {r.status}
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Recording HUD — rendered inside the window so it is captured on video */}
      {recorder.recording && showHud && (
        <div className="pointer-events-none fixed bottom-4 right-4 z-50 bg-black/80 backdrop-blur border border-white/15 rounded-xl px-4 py-3 text-white shadow-2xl min-w-[220px]">
          <div className="flex items-center gap-2 mb-2">
            <Circle size={9} className="fill-red-500 text-red-500 animate-pulse" />
            <span className="text-[11px] font-semibold tracking-wide">
              REC {String(Math.floor(recSeconds / 60)).padStart(2, '0')}:{String(recSeconds % 60).padStart(2, '0')}
            </span>
            <span className="text-[10px] text-white/60 ml-auto">{(status?.device || device).toUpperCase()}</span>
          </div>
          {status?.running && (
            <div className="text-[10px] text-white/70 mb-2 truncate">
              {status.current_model}{status.current_prompt ? ` · ${status.current_prompt}` : ''}
            </div>
          )}
          {ranked.slice(0, 4).map((m, i) => (
            <div key={m.path} className="flex items-center gap-2 text-[11px]">
              <span className="text-white/50 w-3">{i + 1}</span>
              <span className="flex-1 truncate">{m.name}</span>
              <span className="font-mono tabular-nums">{m.avg_tps.toFixed(1)}</span>
            </div>
          ))}
          {ranked.length === 0 && <div className="text-[10px] text-white/60">waiting for results…</div>}
          <div className="text-[9px] text-white/40 mt-2">tokens/sec on {status?.device || device}</div>
        </div>
      )}
    </div>
  )
}
