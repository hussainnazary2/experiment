import { useCallback, useEffect, useRef, useState } from 'react'
import { playgroundApi } from '../../api/client'
import { MarkdownRenderer } from './MarkdownRenderer'
import { ModelPicker } from '../model/ModelPicker'
import { Send, Square, Trash2, Zap, Gauge, Hash, Cpu } from 'lucide-react'

/**
 * A plain streaming chatbot - no agent, no tools, no plans.
 *
 * Everything a viewer needs is on screen: the reply streams in token by
 * token, and two oversized readouts sit directly above the prompt box -
 * decode speed (tok/s) and the total token count for the turn. Model
 * selection uses the same picker as agent mode.
 */

interface Msg {
  role: 'user' | 'assistant'
  content: string
}

interface Stats {
  prompt_tokens?: number
  tokens_generated: number
  total_tokens?: number
  elapsed_ms: number
  ttft_ms: number
  tokens_per_second: number
  total_tokens_per_second?: number
}

/** One large readout in the strip above the prompt box. */
function StatChip({ icon, label, value, accent, title }: {
  icon: React.ReactNode
  label: string
  value: string
  accent?: boolean
  title?: string
}) {
  return (
    <div
      title={title}
      className={`flex items-baseline gap-2 rounded-xl border px-4 py-2.5 ${
        accent ? 'border-accent/40 bg-accent/10' : 'border-border bg-elevated'
      }`}
    >
      <span className={`self-center ${accent ? 'text-accent' : 'text-text-muted'}`}>{icon}</span>
      <span className={`font-mono tabular-nums text-2xl font-bold leading-none ${accent ? 'text-accent' : 'text-text'}`}>
        {value}
      </span>
      <span className="text-[10px] font-semibold uppercase tracking-wider text-text-muted leading-none">{label}</span>
    </div>
  )
}

export function PlaygroundChat() {
  const [loaded, setLoaded] = useState<any>(null)
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [generating, setGenerating] = useState(false)
  const [stats, setStats] = useState<Stats | null>(null)
  const [maxTokens, setMaxTokens] = useState('512')
  const [error, setError] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const refreshStatus = useCallback(async () => {
    try {
      const s = await playgroundApi.status()
      setLoaded(s?.loaded ? s : null)
    } catch { /* ok */ }
  }, [])

  useEffect(() => {
    void refreshStatus()
  }, [refreshStatus])

  // Keep the newest tokens in view while streaming.
  useEffect(() => {
    const el = scrollRef.current
    if (el && generating) el.scrollTop = el.scrollHeight
  }, [messages, generating])

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
          const final = ev.data as Stats
          setStats(final)
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

  return (
    <div className="h-full flex flex-col bg-bg">
      {/* Conversation */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && !error && (
          <div className="h-full flex flex-col items-center justify-center text-center text-text-muted">
            <Zap size={26} className="mb-3 opacity-30" />
            <p className="text-sm">
              {loaded ? 'Type a message to start chatting.' : 'Pick a model below, then type a message.'}
            </p>
            <p className="text-xs mt-1 opacity-70">
              Tokens stream here with live tokens/sec and a token counter above the prompt box.
            </p>
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

      {/* Composer: token readouts sit directly above the prompt box. */}
      <div className="border-t border-border bg-surface px-3 py-2">
        <div className="flex flex-wrap items-center gap-2 mb-2">
          <StatChip
            icon={<Gauge size={18} />}
            label="tok/s"
            value={stats ? stats.tokens_per_second.toFixed(1) : '—'}
            accent
            title="Decode speed (tokens per second since the first token)"
          />
          <StatChip
            icon={<Hash size={18} />}
            label="total tokens"
            value={stats ? String(stats.total_tokens ?? stats.tokens_generated) : '0'}
            title="Prompt + generated tokens for this turn"
          />
        </div>

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
            placeholder={loaded ? 'Type a message…  (Enter to send)' : 'Pick a model to start chatting'}
            disabled={!loaded}
            className="flex-1 min-w-0 resize-none bg-elevated border border-border rounded-lg px-3 py-2 text-sm text-text placeholder:text-text-muted focus:outline-none focus:border-accent/50 disabled:opacity-50"
          />
          {/* Same picker as agent mode: browse a folder, click a model to load it. */}
          <ModelPicker
            align="top"
            className="mb-1"
            load={(path) => playgroundApi.load(path, false)}
            onLoaded={() => { void refreshStatus() }}
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
          {loaded && (
            <span className="flex items-center gap-1.5 min-w-0">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
              <span className="text-text truncate max-w-[220px]">{loaded.filename}</span>
              {loaded.quantization && (
                <span className="px-1.5 py-0.5 rounded bg-elevated border border-border flex-shrink-0">
                  {loaded.quantization}
                </span>
              )}
              {loaded.n_ctx && <span className="flex-shrink-0">ctx {loaded.n_ctx}</span>}
              <span className="flex items-center gap-0.5 flex-shrink-0">
                <Cpu size={9} /> {loaded.use_gpu ? 'GPU' : 'CPU'}
              </span>
            </span>
          )}
          <label className="flex items-center gap-1.5 ml-auto">
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
