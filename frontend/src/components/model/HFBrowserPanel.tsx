import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { hfApi, modelApi } from '../../api/client'
import type { HFModel, HFDownloadStatus } from '../../api/client'
import {
  Globe, Search, Download, Heart, ChevronDown, ChevronRight, RefreshCw,
  X, HardDrive, Check, Zap, Package, ArrowDownToLine,
} from 'lucide-react'

/**
 * HuggingFace model browser.
 *
 * Search GGUF repos (empty query = trending), expand a repo to see its
 * .gguf files with real sizes, download with live progress + speed, then
 * load the model straight into the app. Downloaded files land in the
 * models folder where every picker (Playground, Testing Studio, Agent)
 * finds them.
 */

const SORTS = [
  { key: 'likes', label: 'Likes' },
  { key: 'downloads', label: 'Downloads' },
  { key: 'lastModified', label: 'Recent' },
] as const

function fmtGB(bytes: number): string {
  if (!bytes) return '—'
  const gb = bytes / 2 ** 30
  if (gb >= 100) return `${gb.toFixed(0)} GB`
  if (gb >= 10) return `${gb.toFixed(1)} GB`
  return `${gb.toFixed(2)} GB`
}

function fmtMBps(bps: number): string {
  if (!bps) return '—'
  return `${(bps / 2 ** 20).toFixed(1)} MB/s`
}

function fmtCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

/** Quant tier parsed from a GGUF filename: badge + quality/speed hint. */
function quantOf(filename: string): { label: string; hint: string; tone: string } | null {
  const f = filename.toLowerCase()
  const table: [RegExp, string, string, string][] = [
    [/iq1_s|q1_0|iq2|q2_k/, 'Q1-Q2', 'smallest · fastest · lowest quality', 'text-red-400 border-red-500/30 bg-red-500/10'],
    [/q3_k|iq3/, 'Q3', 'small · fast · decent', 'text-amber-400 border-amber-500/30 bg-amber-500/10'],
    [/q4_k_m|q4_k_s|q4_0|q4_1|iq4/, 'Q4', 'balanced (recommended)', 'text-accent border-accent/30 bg-accent/10'],
    [/q5_k|q5_0|q5_1/, 'Q5', 'high quality · slower', 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10'],
    [/q6_k/, 'Q6', 'near-lossless · slow', 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10'],
    [/q8_0|bf16|f16/, 'Q8/F16', 'lossless · largest', 'text-blue-400 border-blue-500/30 bg-blue-500/10'],
  ]
  for (const [re, label, hint, tone] of table) {
    if (re.test(f)) return { label, hint, tone }
  }
  return null
}

interface FileSize {
  size: number
  loading: boolean
}

export function HFBrowserPanel() {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<(typeof SORTS)[number]['key']>('likes')
  const [models, setModels] = useState<HFModel[]>([])
  const [searching, setSearching] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [sizes, setSizes] = useState<Record<string, Record<string, FileSize>>>({})
  const [dl, setDl] = useState<HFDownloadStatus | null>(null)
  const [localFiles, setLocalFiles] = useState<{ path: string; filename: string; size_gb: number }[]>([])
  const [modelsDir, setModelsDir] = useState('')
  const [notice, setNotice] = useState<string | null>(null)
  const [loadedPath, setLoadedPath] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)

  // ── search ─────────────────────────────────────────────────────────────

  const runSearch = useCallback(async (q: string, s: string) => {
    setSearching(true)
    setError(null)
    try {
      const data = await hfApi.search(q, s as any, 20)
      setModels(data.models)
      if (data.models.length === 0 && q) setError(`No GGUF repos found for "${q}".`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed — check your connection.')
    }
    setSearching(false)
  }, [])

  useEffect(() => {
    void runSearch('', 'likes')
    void refreshLocal()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── local models ───────────────────────────────────────────────────────

  const refreshLocal = useCallback(async () => {
    try {
      const data = await hfApi.local()
      setLocalFiles(data.models)
      setModelsDir(data.models_dir)
    } catch { /* ok */ }
  }, [])

  // ── download polling ───────────────────────────────────────────────────

  useEffect(() => {
    const poll = async () => {
      try {
        const s = await hfApi.downloadStatus()
        setDl(s)
        if (s.status === 'done' || s.status === 'error' || s.status === 'cancelled') {
          void refreshLocal()
        }
      } catch { /* backend gone */ }
    }
    void poll()
    pollRef.current = window.setInterval(poll, 1000)
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current)
    }
  }, [refreshLocal])

  const dlPct = dl && dl.total_bytes > 0
    ? Math.min(100, (dl.bytes_downloaded / dl.total_bytes) * 100)
    : 0

  // ── expand repo -> fetch file sizes ────────────────────────────────────

  const toggleExpand = useCallback(async (m: HFModel) => {
    if (expanded === m.repo_id) {
      setExpanded(null)
      return
    }
    setExpanded(m.repo_id)
    if (sizes[m.repo_id]) return
    // optimistic placeholder entries
    setSizes((prev) => ({
      ...prev,
      [m.repo_id]: Object.fromEntries(m.gguf_files.map((f) => [f.path, { size: 0, loading: true }])),
    }))
    try {
      const data = await hfApi.filesInfo(m.repo_id, m.gguf_files.map((f) => f.path))
      setSizes((prev) => ({
        ...prev,
        [m.repo_id]: Object.fromEntries(
          m.gguf_files.map((f) => [f.path, { size: data.files[f.path]?.size ?? 0, loading: false }]),
        ),
      }))
    } catch {
      setSizes((prev) => ({
        ...prev,
        [m.repo_id]: Object.fromEntries(m.gguf_files.map((f) => [f.path, { size: 0, loading: false }])),
      }))
    }
  }, [expanded, sizes])

  // ── actions ────────────────────────────────────────────────────────────

  const startDownload = async (repoId: string, filePath: string) => {
    setNotice(null)
    try {
      await hfApi.download(repoId, filePath)
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Could not start the download.')
    }
  }

  const cancelDownload = async () => {
    try { await hfApi.cancelDownload() } catch { /* ok */ }
  }

  const loadModel = async (path: string) => {
    setNotice(null)
    try {
      await modelApi.load(path)
      setLoadedPath(path)
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Could not load the model.')
    }
  }

  const localNames = useMemo(() => new Set(localFiles.map((f) => f.filename)), [localFiles])

  return (
    <div className="h-full flex flex-col overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border sticky top-0 bg-surface z-10">
        <h3 className="text-sm font-semibold text-text flex items-center gap-2">
          <Globe size={14} className="text-accent" />
          Model Browser
        </h3>
        <button
          onClick={() => { void refreshLocal(); void runSearch(query, sort) }}
          title="Refresh"
          className="p-1.5 text-text-muted hover:text-text rounded hover:bg-elevated transition-colors"
        >
          <RefreshCw size={13} className={searching ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Search bar */}
      <div className="px-3 py-2 border-b border-border space-y-2">
        <form
          onSubmit={(e) => { e.preventDefault(); void runSearch(query, sort) }}
          className="flex items-center gap-1.5"
        >
          <div className="flex-1 min-w-0 relative">
            <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search HuggingFace GGUFs… (empty = trending)"
              className="w-full bg-elevated border border-border rounded-lg pl-7 pr-2 py-1.5 text-xs text-text placeholder:text-text-muted focus:outline-none focus:border-accent/50"
            />
          </div>
          <button
            type="submit"
            disabled={searching}
            className="text-[11px] font-medium px-3 py-1.5 rounded-lg bg-accent/15 text-accent hover:bg-accent/25 disabled:opacity-40 transition-colors"
          >
            {searching ? <RefreshCw size={12} className="animate-spin" /> : 'Go'}
          </button>
        </form>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-text-muted uppercase tracking-wide">Sort</span>
          <div className="flex gap-0.5 bg-elevated rounded-lg p-0.5">
            {SORTS.map((s) => (
              <button
                key={s.key}
                onClick={() => { setSort(s.key); void runSearch(query, s.key) }}
                className={`text-[10px] px-2 py-0.5 rounded-md transition-colors ${sort === s.key ? 'bg-accent/15 text-accent font-medium' : 'text-text-muted hover:text-text'}`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Download progress (live) */}
      {dl && (dl.status === 'downloading' || dl.status === 'verifying') && (
        <div className="mx-3 mt-2 rounded-lg border border-accent/30 bg-accent/5 px-3 py-2 space-y-1.5">
          <div className="flex items-center justify-between text-[10px]">
            <span className="flex items-center gap-1 text-accent font-medium truncate">
              <Download size={10} className="animate-pulse" />
              {dl.file_path?.split('/').pop()}
            </span>
            <span className="text-text-muted">{fmtMBps(dl.speed_bps)}</span>
          </div>
          <div className="h-1.5 bg-elevated rounded-full overflow-hidden">
            <div className="h-full bg-accent rounded-full transition-all duration-500" style={{ width: `${dlPct}%` }} />
          </div>
          <div className="flex items-center justify-between text-[9px] text-text-muted">
            <span>{fmtGB(dl.bytes_downloaded)} / {fmtGB(dl.total_bytes)} ({dlPct.toFixed(0)}%)</span>
            <button onClick={() => void cancelDownload()} className="flex items-center gap-0.5 text-red-400 hover:text-red-300">
              <X size={9} /> cancel
            </button>
          </div>
        </div>
      )}
      {dl?.status === 'done' && dl.path && (
        <div className="mx-3 mt-2 rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 flex items-center gap-2">
          <Check size={12} className="text-emerald-400 shrink-0" />
          <span className="text-[10px] text-text-sec truncate flex-1">{dl.path.split('/').pop()} downloaded</span>
          <button
            onClick={() => { void loadModel(dl.path!); void refreshLocal() }}
            disabled={loadedPath === dl.path}
            className="text-[10px] px-2 py-0.5 rounded-md bg-accent/15 text-accent hover:bg-accent/25 disabled:opacity-40 transition-colors"
          >
            {loadedPath === dl.path ? 'Loaded' : 'Load'}
          </button>
        </div>
      )}
      {dl?.status === 'error' && (
        <div className="mx-3 mt-2 text-[10px] text-red-400 bg-red-500/10 border border-red-500/30 rounded px-2 py-1.5">
          {dl.error || 'Download failed — partial file kept, retry resumes it.'}
        </div>
      )}

      {notice && (
        <div className="mx-3 mt-2 text-[10px] text-amber-400 bg-amber-500/10 border border-amber-500/30 rounded px-2 py-1.5">
          {notice}
        </div>
      )}

      {/* Local models */}
      {localFiles.length > 0 && (
        <div className="px-3 py-2 border-b border-border space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-text-muted uppercase tracking-wide flex items-center gap-1">
              <HardDrive size={10} /> On disk ({localFiles.length})
            </span>
            {modelsDir && (
              <span className="text-[9px] text-text-muted truncate max-w-[140px]" title={modelsDir}>{modelsDir}</span>
            )}
          </div>
          <div className="space-y-0.5 max-h-32 overflow-y-auto">
            {localFiles.map((f) => (
              <div key={f.path} className="flex items-center gap-2 text-[10px] group">
                <Package size={10} className="text-text-muted shrink-0" />
                <span className="text-text truncate flex-1" title={f.path}>{f.filename}</span>
                <span className="text-text-muted">{f.size_gb.toFixed(1)} GB</span>
                <button
                  onClick={() => void loadModel(f.path)}
                  disabled={loadedPath === f.path}
                  title={loadedPath === f.path ? 'Loaded' : 'Load this model'}
                  className={`p-0.5 rounded transition-colors ${loadedPath === f.path ? 'text-emerald-400' : 'text-text-muted hover:text-accent opacity-0 group-hover:opacity-100'}`}
                >
                  {loadedPath === f.path ? <Check size={11} /> : <Zap size={11} />}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      <div className="flex-1 px-3 py-2 space-y-2">
        {searching && models.length === 0 && (
          <div className="text-[10px] text-text-muted py-6 text-center">Searching HuggingFace…</div>
        )}
        {error && models.length === 0 && (
          <div className="text-[10px] text-amber-400 py-6 text-center px-4">{error}</div>
        )}
        {models.map((m) => {
          const isOpen = expanded === m.repo_id
          const repoSizes = sizes[m.repo_id] || {}
          return (
            <div key={m.repo_id} className="rounded-lg border border-border bg-elevated/40 overflow-hidden">
              <button
                onClick={() => void toggleExpand(m)}
                className="w-full flex items-center gap-2 px-2.5 py-2 hover:bg-elevated/80 transition-colors text-left"
              >
                {isOpen ? <ChevronDown size={12} className="text-text-muted shrink-0" /> : <ChevronRight size={12} className="text-text-muted shrink-0" />}
                <span className="text-[11px] text-text font-medium truncate flex-1">{m.model_name || m.repo_id}</span>
                <span className="flex items-center gap-1 text-[9px] text-text-muted shrink-0">
                  <Heart size={9} /> {fmtCount(m.likes)}
                </span>
                <span className="flex items-center gap-1 text-[9px] text-text-muted shrink-0">
                  <ArrowDownToLine size={9} /> {fmtCount(m.downloads)}
                </span>
                <span className="text-[9px] text-accent shrink-0">{m.gguf_files.length} GGUF</span>
              </button>

              {isOpen && (
                <div className="border-t border-border px-2.5 py-2 space-y-1 max-h-72 overflow-y-auto">
                  {m.gguf_files.map((f) => {
                    const fname = f.path.split('/').pop() || f.path
                    const q = quantOf(fname)
                    const s = repoSizes[f.path]
                    const isDownloadingThis = dl?.status === 'downloading' && dl.repo_id === m.repo_id && dl.file_path === f.path
                    const isLocal = localNames.has(fname)
                    return (
                      <div key={f.path} className="flex items-center gap-2 py-1">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5">
                            <span className="text-[10px] text-text-sec truncate" title={f.path}>{fname}</span>
                            {q && (
                              <span className={`text-[8px] px-1 py-px rounded border shrink-0 ${q.tone}`} title={q.hint}>
                                {q.label}
                              </span>
                            )}
                            {isLocal && (
                              <span className="text-[8px] px-1 py-px rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 shrink-0" title="Already in your models folder">
                                on disk
                              </span>
                            )}
                          </div>
                          <div className="text-[9px] text-text-muted">
                            {s?.loading ? 'size…' : s ? `${fmtGB(s.size)}${q ? ` · ${q.hint}` : ''}` : ''}
                          </div>
                        </div>
                        {isDownloadingThis ? (
                          <span className="text-[9px] text-accent shrink-0 tabular-nums">{dlPct.toFixed(0)}%</span>
                        ) : isLocal ? (
                          <button
                            onClick={() => {
                              const local = localFiles.find((lf) => lf.filename === fname)
                              if (local) void loadModel(local.path)
                            }}
                            disabled={loadedPath?.endsWith(fname) === true}
                            className="text-[9px] px-2 py-1 rounded-md bg-accent/15 text-accent hover:bg-accent/25 disabled:opacity-40 transition-colors shrink-0"
                          >
                            {loadedPath?.endsWith(fname) ? 'Loaded' : 'Load'}
                          </button>
                        ) : (
                          <button
                            onClick={() => void startDownload(m.repo_id, f.path)}
                            disabled={dl?.status === 'downloading' || dl?.status === 'verifying'}
                            title={`Download ${fname} into your models folder`}
                            className="flex items-center gap-1 text-[9px] px-2 py-1 rounded-md bg-elevated border border-border text-text-sec hover:border-accent/40 hover:text-accent disabled:opacity-40 transition-colors shrink-0"
                          >
                            <Download size={10} /> Get
                          </button>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
        {!searching && !error && models.length === 0 && (
          <div className="text-[10px] text-text-muted py-6 text-center">No results.</div>
        )}
      </div>

      {/* Footer hint */}
      <div className="px-3 py-2 border-t border-border text-[9px] text-text-muted leading-relaxed">
        Downloads resume automatically if interrupted. Files land in the models folder and appear in
        the Playground, Testing Studio, and Agent model pickers.
      </div>
    </div>
  )
}
