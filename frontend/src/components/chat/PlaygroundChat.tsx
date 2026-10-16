import { useCallback, useEffect, useRef, useState } from 'react'
import { playgroundApi } from '../../api/client'
import { MarkdownRenderer } from './MarkdownRenderer'
import { Send, Square, Trash2, RefreshCw, Cpu, Zap, Timer, Hash, Gauge } from 'lucide-react'

/**
 * A plain streaming chatbot - no agent, no tools, no plans.
 *
 * Everything a viewer needs is on screen: the reply streams in token by
 * token, and big readouts show tokens/sec, how many tokens were generated,
 * time-to-first-token and elapsed time.
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

  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const loadList = useCallback(async () => {
    setLoadingList(true)
    try {
      const data = await playgroundApi.models()
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
    void loadList()
    void refreshStatus()
  }, [loadList, refreshStatus])

  // Keep the newest tokens in view while streaming.
  useEffect(() => {
    const el = scrollRef.current
    if (el && generating) el.scrollTop = el.scrollHeight
  }, [messages, generating])

  const doLoad = async () => {
    if (!selected) return
    setLoadingModel(true)
    setError(null)
    try {
      const info = await playgroundApi.load(selected)
      setLoaded(info)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load model')
    }
    setLoadingModel(false)
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
      for await (const ev of playgroundApi.stream(
        { message: text, history, max_tokens: mt, temperature: 0.2 },
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
            {loadingList ? 'Scanning for GGUF models…' : candidates.length ? 'Select a model…' : 'No GGUF models found'}
          </option>
          {candidates.map((c) => (
            <option key={c.path} value={c.path}>
              {c.filename} ({c.size_human})
            </option>
          ))}
        </select>
        <button
          onClick={doLoad}
          disabled={!selected || loadingModel || generating}
          className="flex items-center gap-1.5 text-[11px] font-medium px-3 py-1.5 rounded-lg bg-accent/15 text-accent hover:bg-accent/25 disabled:opacity-40 transition-colors"
        >
          {loadingModel ? <RefreshCw size={12} className="animate-spin" /> : <Zap size={12} />}
          Load
        </button>
        <button
          onClick={() => { void loadList(); void refreshStatus() }}
          title="Rescan models"
          className="p-1.5 text-text-muted hover:text-text rounded-lg hover:bg-elevated transition-colors"
        >
          <RefreshCw size={13} className={loadingList ? 'animate-spin' : ''} />
        </button>
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

      {/* Conversation */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && !error && (
          <div className="h-full flex flex-col items-center justify-center text-center text-text-muted">
            <Zap size={26} className="mb-3 opacity-30" />
            <p className="text-sm">Load a model, then type a message.</p>
            <p className="text-xs mt-1 opacity-70">Tokens stream here with live tokens/sec and a token counter.</p>
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
        </div>
      </div>
    </div>
  )
}
