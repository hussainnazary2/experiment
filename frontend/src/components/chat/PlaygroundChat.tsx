import { useCallback, useEffect, useRef, useState } from 'react'
import { playgroundApi } from '../../api/client'
import { MarkdownRenderer } from './MarkdownRenderer'
import { useScreenRecorder } from '../../hooks/useScreenRecorder'
import { useLiveHardware } from '../../hooks/useLiveHardware'
import { useUIStore } from '../../stores/uiStore'
import { useModelStore } from '../../stores/modelStore'
import { Send, Square, Trash2, RefreshCw, Cpu, Zap, Timer, Hash, Gauge, Video, Download, Circle, Globe, FolderOpen } from 'lucide-react'

/**
 * A plain streaming chatbot - no agent, no tools, no plans.
 *
 * Everything a viewer needs is on screen: the reply streams in token by
 * token, big readouts show tokens/sec / tokens / TTFT / elapsed, and a
 * one-click REC button captures the window with a live stats overlay
 * burned into the frame (model, tok/s, RAM/CPU) - ready for Instagram
 * Reels / YouTube Shorts.
 */

interface Candidate {
  path: string
  filename: string
  size_human: string
  quantization: string
  architecture: string
}

interface Msg {
  role: 'user' | 'assistant'
  content: string
}

interface Stats {
  tokens_generated: number
  elapsed_ms: number
  ttft_ms: number
  tokens_per_second: number
  total_tokens_per_second?: number
}

function StatCard({ icon, label, value, unit, accent }: {
  icon: React.ReactNode
  label: string
  value: string
  unit?: string
  accent?: boolean
}) {
  return (
    <div className={`flex-1 min-w-0 rounded-lg border px-3 py-2 ${accent ? 'border-accent/40 bg-accent/10' : 'border-border bg-elevated'}`}>
      <div className={`flex items-center gap-1 text-[10px] uppercase tracking-wide ${accent ? 'text-accent' : 'text-text-muted'}`}>
        {icon}
        <span className="truncate">{label}</span>
      </div>
      <div className="flex items-baseline gap-1 mt-0.5">
        <span className={`font-mono tabular-nums text-2xl leading-none font-semibold ${accent ? 'text-accent' : 'text-text'}`}>
          {value}
        </span>
        {unit && <span className="text-[10px] text-text-muted">{unit}</span>}
      </div>
    </div>
  )
}

export function PlaygroundChat() {
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [selected, setSelected] = useState('')
  const [loaded, setLoaded] = useState<any>(null)
  const [loadingModel, setLoadingModel] = useState(false)
  const [loadingList, setLoadingList] = useState(true)
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [generating, setGenerating] = useState(false)
  const [stats, setStats] = useState<Stats | null>(null)
  const [maxTokens, setMaxTokens] = useState('512')
  const [error, setError] = useState<string | null>(null)
  const [showHud, setShowHud] = useState(true)
  // Device for the NEXT load: CPU by default (100% local, no GPU needed -
  // also the honest baseline for "how fast is this model on MY machine"
  // videos). Switching reloads the current model. Remembered across
  // restarts (localStorage + backend device_preference.json).
  const [useGpu, setUseGpu] = useState(() => {
    try {
      const v = localStorage.getItem('ggufloader_use_gpu')
      return v === null ? false : v === '1' // default: CPU
    } catch { return false }
  })
  // Same folder memory the agent-side picker uses, so both tabs see the
  // same models: read from localStorage on mount, browse to change.
  const [modelFolder, setModelFolder] = useState('')

  const recorder = useScreenRecorder()
  const hw = useLiveHardware(recorder.recording || generating, 1000)
  const { setRightPanelTab, toggleRightPanel, rightPanelOpen } = useUIStore()

  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const openModelBrowser = () => {
    setRightPanelTab('hfbrowser')
    if (!rightPanelOpen) toggleRightPanel()
  }

  const loadList = useCallback(async (directory?: string) => {
    setLoadingList(true)
    try {
      const data = await playgroundApi.models(directory)
      setCandidates(data.models || [])
    } catch { /* ok */ }
    setLoadingList(false)
  }, [])

  const refreshStatus = useCallback(async () => {
    try {
      const s = await playgroundApi.status()
      setLoaded(s?.loaded ? s : null)
    } catch { /* ok */ }
  }, [])

  useEffect(() => {
    // Prefer the folder the user last used in the agent-side picker - the
    // default models dir alone made the dropdown look empty ("no models")
    // whenever GGUFs lived elsewhere, even though agent-mode loading worked.
    let dir: string | undefined
    try {
      const saved = localStorage.getItem('ggufloader_model_folder')
      if (saved) {
        dir = saved
        setModelFolder(saved)
      }
    } catch { /* ignore */ }
    void loadList(dir)
    void refreshStatus()
  }, [loadList, refreshStatus])

  // Poll the loaded-model status so the composer ACTIVATES as soon as a
  // model is ready - whenever that happens: the startup auto-load (a 12B
  // GGUF takes tens of seconds to load), a load from the agent tab, or a
  // load right here. The one-shot check on mount used to leave the input
  // disabled forever when the auto-load finished AFTER this tab opened,
  // which looked like "model loaded but I can't type".
  useEffect(() => {
    if (loadingModel) return // don't fight the swap currently in progress
    const t = setInterval(() => { void refreshStatus() }, 2000)
    return () => clearInterval(t)
  }, [refreshStatus, loadingModel])

  // Keep the newest tokens in view while streaming.
  useEffect(() => {
    const el = scrollRef.current
    if (el && generating) el.scrollTop = el.scrollHeight
  }, [messages, generating])

  const browseFolder = async () => {
    let path: string | null | undefined
    if ((window as any).electronAPI?.openFolderDialog) {
      path = await (window as any).electronAPI.openFolderDialog()
    } else {
      path = prompt('Enter the path to your models folder:')
    }
    if (path) {
      setModelFolder(path)
      try { localStorage.setItem('ggufloader_model_folder', path) } catch {}
      void loadList(path)
    }
  }

  const doLoad = async (pathArg?: string, gpuArg?: boolean) => {
    const path = pathArg ?? selected
    if (!path) return
    setLoadingModel(true)
    setError(null)
    try {
      const info = await playgroundApi.load(path, gpuArg ?? useGpu)
      setLoaded(info)
      // Sync the shared model store so the header chip / agent tab agree.
      void useModelStore.getState().refreshInfo()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load model')
      // A failed load still EVICTED whatever was loaded before (the backend
      // unloads the old model first). Re-sync this tab and the header so the
      // UI doesn't keep showing a model that is no longer in memory.
      void refreshStatus()
      void useModelStore.getState().refreshInfo()
    }
    setLoadingModel(false)
  }

  // Switch CPU/GPU: remembers the choice and reloads the CURRENT model on
  // the new device (when one is loaded) so the switch applies immediately,
  // and the backend persists it for the next startup auto-load.
  const switchDevice = (gpu: boolean) => {
    if (gpu === useGpu || generating || loadingModel) return
    setUseGpu(gpu)
    try { localStorage.setItem('ggufloader_use_gpu', gpu ? '1' : '0') } catch {}
    const path = selected || loaded?.path
    if (path) void doLoad(path, gpu)
  }

  const send = async () => {
    const text = input.trim()
    if (!text || generating) return
    if (!loaded) {
      setError('Load a model first.')
      return
    }

    const history = messages.map((m) => ({ role: m.role, content: m.content }))
    setMessages((prev) => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    setInput('')
    setError(null)
    setStats(null)
    setGenerating(true)

    const ac = new AbortController()
    abortRef.current = ac
    const mt = Number(maxTokens) || 512
    let acc = ''

    try {
      // No hardcoded temperature: the backend applies the loaded model's
      // family sampling (e.g. LFM2.5: temp 0.2 / top_k 80 / repeat 1.05)
      // so every model answers with ITS recommended settings, plus proper
      // stop tokens - what fixed the "<s>hi</s>" echo loops.
      for await (const ev of playgroundApi.stream(
        { message: text, history, max_tokens: mt },
        ac.signal,
      )) {
        if (ev.type === 'token') {
          acc += ev.data.token
          setMessages((prev) => {
            const next = prev.slice()
            next[next.length - 1] = { role: 'assistant', content: acc }
            return next
          })
        } else if (ev.type === 'stats') {
          setStats(ev.data as Stats)
        } else if (ev.type === 'done') {
          if (typeof ev.data.content === 'string') acc = ev.data.content
          setMessages((prev) => {
            const next = prev.slice()
            next[next.length - 1] = { role: 'assistant', content: acc }
            return next
          })
          setStats(ev.data as Stats)
        } else if (ev.type === 'error') {
          setError(ev.data.message || 'Generation failed')
        }
      }
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError')) {
        setError(e instanceof Error ? e.message : 'Generation failed')
      }
    } finally {
      setGenerating(false)
      abortRef.current = null
    }
  }

  const stop = () => abortRef.current?.abort()

  const clearChat = () => {
    stop()
    setMessages([])
    setStats(null)
    setError(null)
  }

  const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`)
  const recSeconds = Math.floor(recorder.durationMs / 1000)
  const recTime = `${String(Math.floor(recSeconds / 60)).padStart(2, '0')}:${String(recSeconds % 60).padStart(2, '0')}`

  return (
    <div className="h-full flex flex-col bg-bg">
      {/* Model bar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border bg-surface">
        <Cpu size={14} className="text-text-muted flex-shrink-0" />
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          disabled={generating || loadingModel}
          className="flex-1 min-w-0 bg-elevated border border-border rounded-lg px-2 py-1.5 text-xs text-text focus:outline-none focus:border-accent/50"
        >
          <option value="">
            {loadingList
              ? 'Scanning for GGUF models…'
              : candidates.length
                ? 'Select a model…'
                : modelFolder
                  ? 'No GGUF models in this folder'
                  : 'No GGUF models found'}
          </option>
          {candidates.map((c) => (
            <option key={c.path} value={c.path}>
              {c.filename} ({c.size_human})
            </option>
          ))}
        </select>
        <button
          onClick={() => { void doLoad() }}
          disabled={!selected || loadingModel || generating}
          className="flex items-center gap-1.5 text-[11px] font-medium px-3 py-1.5 rounded-lg bg-accent/15 text-accent hover:bg-accent/25 disabled:opacity-40 transition-colors"
        >
          {loadingModel ? <RefreshCw size={12} className="animate-spin" /> : <Zap size={12} />}
          Load
        </button>
        {/* CPU / GPU device toggle - which device the model runs on.
            CPU is the default; switching reloads the current model. */}
        <div
          title="Device the model runs on. CPU = fully on CPU (no GPU needed) · GPU = offload to your graphics card. Switching reloads the current model, and the choice is remembered for the next startup."
          className="flex items-center rounded-lg border border-border overflow-hidden flex-shrink-0"
        >
          <button
            onClick={() => switchDevice(false)}
            disabled={generating || loadingModel}
            className={`px-2 py-1.5 text-[10px] font-semibold tracking-wide transition-colors ${
              !useGpu ? 'bg-accent/25 text-accent' : 'text-text-muted hover:text-text'
            }`}
          >
            CPU
          </button>
          <button
            onClick={() => switchDevice(true)}
            disabled={generating || loadingModel}
            className={`px-2 py-1.5 text-[10px] font-semibold tracking-wide border-l border-border transition-colors ${
              useGpu ? 'bg-accent/25 text-accent' : 'text-text-muted hover:text-text'
            }`}
          >
            GPU
          </button>
        </div>
        {/* Browse to any models folder - same memory the agent picker uses */}
        <button
          onClick={() => { void browseFolder() }}
          title={modelFolder ? `Models folder: ${modelFolder}` : 'Choose your models folder'}
          className="p-1.5 text-text-muted hover:text-text rounded-lg hover:bg-elevated transition-colors"
        >
          <FolderOpen size={13} />
        </button>
        <button
          onClick={() => { void loadList(modelFolder || undefined); void refreshStatus() }}
          title="Rescan models"
          className="p-1.5 text-text-muted hover:text-text rounded-lg hover:bg-elevated transition-colors"
        >
          <RefreshCw size={13} className={loadingList ? 'animate-spin' : ''} />
        </button>
        {/* One-click screen recording with stats overlay */}
        {recorder.supported && (
          <button
            onClick={() => { void recorder.start() }}
            disabled={recorder.recording}
            title="Record this window with live stats overlay — for Instagram / YouTube"
            className="flex items-center gap-1.5 text-[11px] font-medium px-3 py-1.5 rounded-lg bg-red-500/15 text-red-400 hover:bg-red-500/25 disabled:opacity-40 transition-colors"
          >
            <Video size={12} /> REC
          </button>
        )}
        {recorder.recording && (
          <button
            onClick={recorder.stop}
            title="Stop recording"
            className="flex items-center gap-1.5 text-[11px] font-medium px-3 py-1.5 rounded-lg bg-red-500/25 text-red-400 hover:bg-red-500/35 transition-colors"
          >
            <Square size={12} /> Stop REC
          </button>
        )}
        <label className="flex items-center gap-1 text-[10px] text-text-muted cursor-pointer select-none" title="Show the live stats overlay inside the recording">
          <input
            type="checkbox"
            checked={showHud}
            onChange={(e) => setShowHud(e.target.checked)}
            className="accent-accent"
          />
          HUD
        </label>
      </div>

      {/* Model chip */}
      {loaded && (
        <div className="flex items-center gap-2 px-3 py-1.5 text-[10px] text-text-muted border-b border-border">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
          <span className="text-text truncate">{loaded.filename}</span>
          {loaded.quantization && <span className="px-1.5 py-0.5 rounded bg-elevated border border-border">{loaded.quantization}</span>}
          {loaded.n_ctx && <span>ctx {loaded.n_ctx}</span>}
          <span className="ml-auto">{loaded.use_gpu ? 'GPU' : 'CPU'}</span>
        </div>
      )}
      {loadingModel && (
        <div className="flex items-center gap-2 px-3 py-1.5 text-[10px] text-text-muted border-b border-border">
          <RefreshCw size={10} className="animate-spin" />
          Loading model on {useGpu ? 'GPU' : 'CPU'}…
        </div>
      )}

      {/* Live stats - the whole point: a viewer must see the speed */}
      <div className="flex gap-2 px-3 py-2 border-b border-border bg-surface/60">
        <StatCard
          icon={<Gauge size={11} />}
          label="Tokens / sec"
          value={stats ? stats.tokens_per_second.toFixed(1) : '—'}
          accent
        />
        <StatCard
          icon={<Hash size={11} />}
          label="Tokens generated"
          value={stats ? String(stats.tokens_generated) : '0'}
        />
        <StatCard
          icon={<Timer size={11} />}
          label="First token"
          value={stats ? fmtMs(stats.ttft_ms) : '—'}
        />
        <StatCard
          icon={<Timer size={11} />}
          label="Elapsed"
          value={stats ? fmtMs(stats.elapsed_ms) : '—'}
        />
      </div>

      {/* Finished recording -> download the video */}
      {recorder.blob && !recorder.recording && (
        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border bg-emerald-500/5">
          <span className="text-[10px] text-emerald-400 flex items-center gap-1">
            <Circle size={8} className="fill-emerald-400 text-emerald-400" /> Recording ready ({recTime})
          </span>
          <button
            onClick={() => recorder.download()}
            className="ml-auto flex items-center gap-1.5 text-[11px] font-medium px-3 py-1 rounded-lg bg-elevated border border-border text-text-sec hover:border-accent/40 transition-colors"
          >
            <Download size={11} /> Save .webm
          </button>
          <button
            onClick={recorder.reset}
            className="text-[10px] text-text-muted hover:text-text px-1"
            title="Discard recording"
          >
            ✕
          </button>
        </div>
      )}

      {/* Conversation */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && !error && (
          <div className="h-full flex flex-col items-center justify-center text-center text-text-muted">
            <Zap size={26} className="mb-3 opacity-30" />
            <p className="text-sm">Load a model, then type a message.</p>
            <p className="text-xs mt-1 opacity-70">Tokens stream here with live tokens/sec and a token counter.</p>
            {candidates.length === 0 && !loadingList && (
              <button
                onClick={openModelBrowser}
                className="mt-4 flex items-center gap-2 text-xs font-medium px-4 py-2 rounded-lg bg-accent/15 text-accent hover:bg-accent/25 transition-colors"
              >
                <Globe size={13} /> Browse HuggingFace models
              </button>
            )}
          </div>
        )}

        {messages.map((m, i) => {
          const isLast = i === messages.length - 1
          const streamingThis = generating && isLast && m.role === 'assistant'
          return (
            <div key={i} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
              <div
                className={
                  m.role === 'user'
                    ? 'max-w-[80%] rounded-2xl rounded-br-sm bg-accent/15 border border-accent/25 px-4 py-2 text-sm text-text whitespace-pre-wrap break-words'
                    : 'max-w-[85%] rounded-2xl rounded-bl-sm bg-elevated border border-border px-4 py-2 text-sm text-text'
                }
              >
                {m.role === 'assistant' ? (
                  m.content
                    ? <MarkdownRenderer content={m.content} />
                    : <span className="text-text-muted">{streamingThis ? '…' : ''}</span>
                ) : (
                  m.content
                )}
                {streamingThis && m.content && (
                  <span className="inline-block w-1.5 h-3.5 ml-0.5 align-text-bottom bg-accent animate-pulse" />
                )}
              </div>
            </div>
          )
        })}

        {error && (
          <div className="text-[11px] text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2">
            {error}
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="border-t border-border bg-surface px-3 py-2">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send()
              }
            }}
            rows={2}
            placeholder={loaded ? 'Type a message…  (Enter to send)' : 'Load a model to start chatting'}
            disabled={!loaded}
            className="flex-1 min-w-0 resize-none bg-elevated border border-border rounded-lg px-3 py-2 text-sm text-text placeholder:text-text-muted focus:outline-none focus:border-accent/50 disabled:opacity-50"
          />
          {generating ? (
            <button
              onClick={stop}
              className="flex items-center gap-1.5 text-xs font-medium px-3 py-2 rounded-lg bg-red-500/15 text-red-400 hover:bg-red-500/25 transition-colors"
            >
              <Square size={13} /> Stop
            </button>
          ) : (
            <button
              onClick={() => void send()}
              disabled={!loaded || !input.trim()}
              className="flex items-center gap-1.5 text-xs font-medium px-3 py-2 rounded-lg bg-accent/15 text-accent hover:bg-accent/25 disabled:opacity-40 transition-colors"
            >
              <Send size={13} /> Send
            </button>
          )}
        </div>

        <div className="flex items-center gap-3 mt-2 text-[10px] text-text-muted">
          <label className="flex items-center gap-1.5">
            Max tokens
            <input
              value={maxTokens}
              onChange={(e) => setMaxTokens(e.target.value.replace(/[^0-9]/g, ''))}
              disabled={generating}
              className="w-16 bg-elevated border border-border rounded px-1.5 py-0.5 text-text focus:outline-none focus:border-accent/50"
            />
          </label>
          <button onClick={clearChat} className="flex items-center gap-1 hover:text-text transition-colors">
            <Trash2 size={11} /> Clear
          </button>
          <button onClick={openModelBrowser} className="ml-auto flex items-center gap-1 hover:text-text transition-colors">
            <Globe size={11} /> Models
          </button>
        </div>
      </div>

      {/* REC HUD — rendered inside the window so it is captured in the video */}
      {recorder.recording && showHud && (
        <div className="pointer-events-none fixed bottom-4 right-4 z-50 bg-black/80 backdrop-blur border border-white/15 rounded-xl px-4 py-3 text-white shadow-2xl min-w-[240px]">
          <div className="flex items-center gap-2 mb-2">
            <Circle size={9} className="fill-red-500 text-red-500 animate-pulse" />
            <span className="text-[11px] font-semibold tracking-wide">REC {recTime}</span>
            <span className="text-[10px] text-white/60 ml-auto">
              {loaded?.use_gpu ? 'GPU' : 'CPU'} · 100% local
            </span>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-white/70 mb-2">
            <Cpu size={10} />
            <span className="truncate">{loaded?.filename || 'no model'}</span>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
            <div className="flex justify-between">
              <span className="text-white/50">tok/s</span>
              <span className="font-mono tabular-nums text-accent">{stats ? stats.tokens_per_second.toFixed(1) : '—'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/50">tokens</span>
              <span className="font-mono tabular-nums">{stats ? stats.tokens_generated : 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/50">TTFT</span>
              <span className="font-mono tabular-nums">{stats ? fmtMs(stats.ttft_ms) : '—'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/50">RAM</span>
              <span className="font-mono tabular-nums">
                {hw?.ram_total_gb != null ? `${(hw.ram_used_gb ?? 0).toFixed(1)}/${hw.ram_total_gb.toFixed(0)} GB` : '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/50">CPU</span>
              <span className="font-mono tabular-nums">{hw?.cpu_pct != null ? `${hw.cpu_pct.toFixed(0)}%` : '—'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/50">GPU</span>
              <span className="font-mono tabular-nums">
                {hw?.gpu ? `${hw.gpu.utilization_pct.toFixed(0)}%` : '—'}
              </span>
            </div>
          </div>
          <div className="text-[9px] text-white/40 mt-2">GGUF Loader · no cloud, no API keys</div>
        </div>
      )}
    </div>
  )
}
