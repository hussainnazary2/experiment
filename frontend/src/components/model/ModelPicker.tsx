import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ChevronDown,
  HardDrive,
  Loader2,
  Zap,
  Download,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Wrench,
  Undo2,
} from 'lucide-react'
import { modelApi } from '../../api/client'
import { useModelStore } from '../../stores/modelStore'
import { useDownloadStore } from '../../stores/downloadStore'
import { toast } from '../../stores/toastStore'

/**
 * A GGUF file as returned by GET /api/model/catalog. The route returns the
 * profile fields flat (family/quantization/architecture) but older callers
 * expected them nested under `profile`, so both shapes are accepted.
 */
export interface FolderModel {
  path: string
  filename: string
  size_gb?: number
  size_human?: string
  family?: string
  quantization?: string
  architecture?: string
  profile?: {
    family?: string
    quantization?: string
    architecture?: string
  }
}

/** localStorage key for the folder the picker scans (shared across views). */
export const MODEL_FOLDER_KEY = 'ggufloader_model_folder'

interface ModelPickerProps {
  /** Popup opens upward (composer) or downward (top bar). */
  align?: 'top' | 'bottom'
  /**
   * Loader override. Defaults to the shared model store, which loads
   * through POST /api/model/load. The chatbot passes its own loader to
   * keep its CPU-only speed-demo semantics.
   */
  load?: (path: string) => Promise<unknown>
  /** Called after a model finished loading. */
  onLoaded?: (path: string) => void
  /** Extra classes for the wrapper. */
  className?: string
}

function familyOf(model: FolderModel) {
  return model.profile?.family || model.family || ''
}

function quantOf(model: FolderModel) {
  return model.profile?.quantization || model.quantization || ''
}

/**
 * The model picker: browse a folder of GGUFs, list them with family /
 * quantization / size, and load one with a click.
 *
 * Shared by agent mode (MessageInput) and the chatbot (PlaygroundChat) so
 * the model selection and the model list are identical in both views.
 */
export function ModelPicker({ align = 'top', load, onLoaded, className = '' }: ModelPickerProps) {
  const [folder, setFolder] = useState('')
  const [models, setModels] = useState<FolderModel[]>([])
  const [scanning, setScanning] = useState(false)
  const [open, setOpen] = useState(false)
  const [loadingPath, setLoadingPath] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  // What the backend can say about the file that just failed to load: a
  // metadata problem it can fix in place, an explanation it cannot fix (for
  // example a tensor-name mismatch), and whether a previous repair is undoable.
  const [diagnosis, setDiagnosis] = useState<{
    path: string
    filename: string
    fix: string | null
    explanation: { summary: string; detail: string } | null
    hasBackup: boolean
  } | null>(null)
  const [repairing, setRepairing] = useState(false)

  const info = useModelStore((s) => s.info)
  const dl = useDownloadStore((s) => s.dl)
  const startDownload = useDownloadStore((s) => s.startDownload)

  const scanFolder = useCallback(async (dir: string) => {
    if (!dir) return
    setScanning(true)
    setError(null)
    try {
      const data = await modelApi.catalog(dir)
      setModels(data?.models || [])
    } catch (e: any) {
      setError(e?.message || 'Could not scan that folder')
      setModels([])
    }
    setScanning(false)
  }, [])

  // Restore the remembered folder on mount. With none remembered, fall back
  // to the app's default models folder so the list is never empty on a
  // first run (the catalog scan with no directory uses it).
  useEffect(() => {
    let saved = ''
    try {
      saved = localStorage.getItem(MODEL_FOLDER_KEY) || ''
    } catch {
      /* ignore */
    }
    if (saved) {
      setFolder(saved)
      void scanFolder(saved)
      return
    }
    void (async () => {
      try {
        const data = await modelApi.catalog()
        if (data?.directory) setFolder(data.directory)
        setModels(data?.models || [])
      } catch {
        /* ignore */
      }
    })()
  }, [scanFolder])

  // When the shared pinned-model download finishes, point the picker at the
  // folder it landed in so the fresh file appears in the list.
  const dlDoneHandled = useRef(false)
  useEffect(() => {
    if (dl?.status === 'done') {
      if (dlDoneHandled.current) return
      dlDoneHandled.current = true
      const dir = String(dl.path || '').replace(/(\\|\/)[^\\/]+\.gguf$/i, '')
      if (dir) {
        setFolder(dir)
        try {
          localStorage.setItem(MODEL_FOLDER_KEY, dir)
        } catch {
          /* ignore */
        }
        void scanFolder(dir)
      }
    } else {
      dlDoneHandled.current = false
    }
  }, [dl, scanFolder])

  async function handleBrowse() {
    let path: string | null = null
    if ((window as any).electronAPI?.openFolderDialog) {
      path = await (window as any).electronAPI.openFolderDialog()
    } else {
      path = prompt('Enter the path to your models folder:')
    }
    if (!path) return
    setFolder(path)
    try {
      localStorage.setItem(MODEL_FOLDER_KEY, path)
    } catch {
      /* ignore */
    }
    await scanFolder(path)
  }

  /**
   * After a failed load: ask the backend to explain the file (read-only).
   * The load message is passed along because it names what llama.cpp wanted -
   * e.g. a tensor - which is what makes a tensor check possible.
   */
  async function diagnoseModel(model: FolderModel, message: string) {
    try {
      const diagnosed = await modelApi.diagnose(model.path, message)
      const findings = diagnosed?.findings || []
      const fixable = findings.find((f: any) => f?.repairable)
      const explained = findings.find((f: any) => !f?.repairable)
      setDiagnosis({
        path: model.path,
        filename: model.filename,
        fix: fixable ? (fixable.fix || fixable.summary) : null,
        explanation: !fixable && explained
          ? { summary: explained.summary, detail: explained.detail }
          : null,
        hasBackup: Boolean(diagnosed?.has_header_backup),
      })
    } catch {
      setDiagnosis(null)                    // diagnosis is best-effort
    }
  }

  async function handleLoad(model: FolderModel) {
    setLoadingPath(model.path)
    setError(null)
    setDiagnosis(null)
    try {
      if (load) {
        await load(model.path)
      } else {
        await useModelStore.getState().loadModel(model.path)
      }
      toast.success(`Loaded ${model.filename}`)
      onLoaded?.(model.path)
      setOpen(false)
    } catch (e: any) {
      // Previously this failure was swallowed entirely, so clicking a model
      // that the backend rejected looked like nothing happened at all.
      const message = e?.message || `Failed to load ${model.filename}`
      setError(message)
      toast.error(message)
      void diagnoseModel(model, message)
    }
    setLoadingPath(null)
  }

  /**
   * Repair the file's metadata in place, then retry the load. Tensor data is
   * never rewritten (the header is adjusted and verified server-side), which
   * is why this is safe to offer for a multi-GB model.
   */
  async function handleRepair() {
    if (!diagnosis) return
    const target = models.find((m) => m.path === diagnosis.path)
    setRepairing(true)
    try {
      const result = await modelApi.repair(diagnosis.path, true)
      if (!result?.applied) {
        setError(result?.message || 'The metadata could not be repaired')
        return
      }
      toast.success(result.message || 'Metadata repaired')
      setDiagnosis(null)
      if (target) await handleLoad(target)
    } catch (e: any) {
      setError(e?.message || 'Repair failed')
    } finally {
      setRepairing(false)
    }
  }

  /**
   * Put the file back exactly as it was before the metadata repair. Offered
   * whenever the backup is still beside the model, because a repair that did
   * not fix the load (the file failed on something else) should not leave the
   * user's multi-GB file modified without them knowing.
   */
  async function handleUndoRepair() {
    if (!diagnosis) return
    setRepairing(true)
    try {
      const result = await modelApi.undoRepair(diagnosis.path)
      toast.success(result?.message || 'Original header restored')
      setDiagnosis(null)
    } catch (e: any) {
      setError(e?.message || 'Undo failed')
    } finally {
      setRepairing(false)
    }
  }

  const loadedName = info.loaded ? info.filename : ''
  const label = loadedName || 'Select model…'
  const folderName = folder ? folder.split(/[/\\]/).pop() : 'No folder selected'

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 px-2 py-1 text-[11px] text-text-muted hover:text-text-sec rounded-lg hover:bg-bg/50 transition-colors max-w-[190px]"
        title={info.loaded ? `${info.filename}${info.quantization ? ` · ${info.quantization}` : ''}` : 'Select a GGUF model'}
      >
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${info.loaded ? 'bg-success' : 'bg-text-muted'}`} />
        <span className="truncate">{label}</span>
        <ChevronDown size={10} className="flex-shrink-0" />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div
            className={`absolute right-0 w-80 bg-elevated border border-border rounded-xl shadow-xl z-50 overflow-hidden ${
              align === 'top' ? 'bottom-full mb-1' : 'top-full mt-1'
            }`}
          >
            {/* Folder header */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-bg/50">
              <div className="flex items-center gap-2 min-w-0">
                <HardDrive size={12} className="text-accent flex-shrink-0" />
                <span className="text-[10px] text-text-muted truncate" title={folder}>{folderName}</span>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                {folder && (
                  <button
                    onClick={() => void scanFolder(folder)}
                    title="Rescan folder"
                    className="text-text-muted hover:text-text"
                  >
                    <RefreshCw size={11} className={scanning ? 'animate-spin' : ''} />
                  </button>
                )}
                <button
                  onClick={handleBrowse}
                  className="text-[10px] text-accent hover:text-accent-hover"
                >
                  Browse
                </button>
              </div>
            </div>

            {/* Loaded model summary */}
            {info.loaded && (
              <div className="px-3 py-2 border-b border-border bg-accent/5">
                <div className="flex items-center gap-1.5 text-[11px] text-text">
                  <CheckCircle2 size={11} className="text-success flex-shrink-0" />
                  <span className="truncate font-mono">{info.filename}</span>
                </div>
                <div className="flex items-center gap-1.5 mt-0.5 text-[10px] text-text-muted">
                  {info.family && <span>{info.family}</span>}
                  {info.quantization && <span>· {info.quantization}</span>}
                  {info.context_length ? <span>· ctx {info.context_length}</span> : null}
                  <span className="ml-auto">{info.gpu ? 'GPU' : 'CPU'}</span>
                </div>
              </div>
            )}

            {scanning && (
              <div className="flex items-center gap-2 px-3 py-3 text-xs text-text-muted">
                <Loader2 size={12} className="animate-spin" />
                Scanning folder...
              </div>
            )}

            {/* Model list */}
            <div className="max-h-72 overflow-y-auto">
              {models.map((model) => (
                <button
                  key={model.path}
                  onClick={() => void handleLoad(model)}
                  disabled={loadingPath === model.path}
                  className={`w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-bg/50 transition-colors ${
                    info.path === model.path ? 'bg-accent/10' : ''
                  } ${loadingPath === model.path ? 'opacity-60' : ''}`}
                >
                  {loadingPath === model.path ? (
                    <Loader2 size={12} className="text-accent animate-spin flex-shrink-0" />
                  ) : (
                    <div
                      className={`w-2 h-2 rounded-full flex-shrink-0 ${
                        info.path === model.path ? 'bg-accent' : 'bg-text-muted'
                      }`}
                    />
                  )}
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-medium text-text truncate">{model.filename}</div>
                    <div className="flex items-center gap-1 text-[10px] text-text-muted">
                      {familyOf(model) && <span>{familyOf(model)}</span>}
                      {quantOf(model) && <span>· {quantOf(model)}</span>}
                      {model.size_gb ? <span>· {model.size_gb} GB</span> : null}
                    </div>
                  </div>
                  <Zap size={10} className="text-accent flex-shrink-0" />
                </button>
              ))}
            </div>

            {/* Nothing found: offer the tuned Gemma 4 download instead. */}
            {!scanning && models.length === 0 && (
              info.loaded ? (
                <div className="border-t border-border px-3 py-3">
                  <div className="flex items-center gap-1.5 text-[11px] text-green-400">
                    <CheckCircle2 size={11} className="flex-shrink-0" />
                    <span className="truncate">Model ready: {info.filename}</span>
                  </div>
                  <p className="mt-1 text-[10px] text-text-muted leading-snug">
                    This folder has no GGUF files. Browse to a folder of models to switch.
                  </p>
                </div>
              ) : (
                <div className="border-t border-border px-3 py-3 space-y-2">
                  <p className="text-[11px] text-text-muted leading-snug">
                    {folder ? 'No GGUF models found in this folder.' : 'No model folder selected.'}{' '}
                    Browse to a folder, or download the tuned Gemma 4 12B Q4_K_M:
                  </p>
                  {(!dl || dl.status === 'idle' || dl.status === 'disabled') && (
                    <button
                      onClick={() => void startDownload()}
                      className="w-full flex items-center justify-center gap-1.5 px-3 py-2 bg-accent/10 border border-accent/25 rounded-lg text-[11px] font-medium text-accent hover:bg-accent/20 transition-colors"
                    >
                      <Download size={11} /> Download Gemma 4 12B Q4_K_M (~8 GB)
                    </button>
                  )}
                  {dl?.status === 'downloading' && (
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-1.5 text-[10px] text-text-muted">
                        <Loader2 size={10} className="animate-spin" />
                        <span>Downloading...</span>
                        <span className="ml-auto font-mono text-text-sec">{Math.round(dl.progress * 100)}%</span>
                      </div>
                      <div className="h-1.5 bg-elevated border border-border rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent transition-all duration-500"
                          style={{ width: `${Math.min(100, Math.max(0, Math.round(dl.progress * 100)))}%` }}
                        />
                      </div>
                    </div>
                  )}
                  {dl?.status === 'done' && (
                    <div className="flex items-center gap-1.5 text-[11px] text-green-400">
                      <CheckCircle2 size={11} /> Downloaded — loading model...
                    </div>
                  )}
                  {dl?.status === 'error' && (
                    <div className="space-y-1.5">
                      <p className="text-[10px] text-red-400 leading-snug">{dl.message || 'Download failed'}</p>
                      <button
                        onClick={() => void startDownload()}
                        className="w-full px-3 py-1.5 bg-elevated border border-border rounded-lg text-[11px] text-text-sec hover:border-accent transition-colors"
                      >
                        Retry download
                      </button>
                    </div>
                  )}
                </div>
              )
            )}

            {/* Last load failure - the backend rejects with a reason worth showing. */}
            {error && (
              <div className="border-t border-border px-3 py-2">
                <div className="flex items-start gap-1.5 text-[10px] text-red-400 leading-snug">
                  <AlertCircle size={11} className="flex-shrink-0 mt-0.5" />
                  <span className="break-words">{error}</span>
                </div>
              </div>
            )}

            {/* What the backend can say about the file that just failed. */}
            {diagnosis && (
              <div className="border-t border-border px-3 py-2 space-y-1.5">
                {diagnosis.fix && (
                  <>
                    <p className="text-[10px] text-text-muted leading-snug">
                      <span className="text-text-sec font-medium">{diagnosis.filename}</span> has a
                      metadata problem llama.cpp refuses: {diagnosis.fix}.
                    </p>
                    <button
                      onClick={() => void handleRepair()}
                      disabled={repairing}
                      className="w-full flex items-center justify-center gap-1.5 px-3 py-2 bg-accent/10 border border-accent/25 rounded-lg text-[11px] font-medium text-accent hover:bg-accent/20 disabled:opacity-50 transition-colors"
                    >
                      {repairing
                        ? <Loader2 size={11} className="animate-spin" />
                        : <Wrench size={11} />}
                      Repair metadata &amp; load
                    </button>
                    <p className="text-[9.5px] text-text-muted leading-snug">
                      Rewrites the header only - the file keeps its size and the model weights are
                      never touched. The original header is saved next to the model.
                    </p>
                  </>
                )}

                {diagnosis.explanation && (
                  <div className="space-y-1">
                    <div className="flex items-start gap-1.5 text-[10px] text-amber-400 leading-snug">
                      <AlertCircle size={11} className="flex-shrink-0 mt-0.5" />
                      <span>{diagnosis.explanation.summary}</span>
                    </div>
                    <p className="text-[9.5px] text-text-muted leading-snug">
                      {diagnosis.explanation.detail}
                    </p>
                  </div>
                )}

                {diagnosis.hasBackup && (
                  <button
                    onClick={() => void handleUndoRepair()}
                    disabled={repairing}
                    className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 bg-elevated border border-border rounded-lg text-[10.5px] text-text-sec hover:border-accent disabled:opacity-50 transition-colors"
                  >
                    <Undo2 size={11} /> Undo metadata repair
                  </button>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
