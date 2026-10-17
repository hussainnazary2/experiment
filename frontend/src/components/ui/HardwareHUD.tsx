import { useEffect, useState } from 'react'
import { Activity, Cpu, MemoryStick, Gauge, ChevronDown, ChevronUp } from 'lucide-react'
import { useLiveHardware } from '../../hooks/useLiveHardware'
import { useModelStore } from '../../stores/modelStore'

const STORAGE_KEY = 'ggufloader_hw_hud_open'

function Bar({ pct, tone }: { pct: number; tone: string }) {
  return (
    <div className="h-1 bg-white/20 rounded-full overflow-hidden w-14">
      <div
        className={`h-full rounded-full transition-all duration-500 ${tone}`}
        style={{ width: `${Math.min(100, Math.max(2, pct))}%` }}
      />
    </div>
  )
}

function Row({
  icon, label, value, pct, tone,
}: {
  icon: React.ReactNode
  label: string
  value: string
  pct?: number
  tone?: string
}) {
  return (
    <div className="flex items-center gap-2 text-[10px]">
      <span className="text-white/60 shrink-0">{icon}</span>
      <span className="text-white/70 w-9 shrink-0">{label}</span>
      <span className="font-mono tabular-nums text-white/90 w-[86px] shrink-0">{value}</span>
      {pct != null && <Bar pct={pct} tone={tone || 'bg-white/70'} />}
    </div>
  )
}

/**
 * Floating live-hardware HUD (bottom-left).
 *
 * Renders RAM / CPU / GPU / model-memory readouts so screen recordings
 * always show what the machine is doing while a local model runs.
 * Collapsible; the open state persists in localStorage.
 */
export function HardwareHUD() {
  const [open, setOpen] = useState(true)
  const { info } = useModelStore()
  const hw = useLiveHardware(true)

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved === '0') setOpen(false)
    } catch { /* ignore */ }
  }, [])

  const toggle = () => {
    setOpen((o) => {
      try { localStorage.setItem(STORAGE_KEY, o ? '0' : '1') } catch { /* ignore */ }
      return !o
    })
  }

  const gpu = hw?.gpu
  const ramPct = hw?.ram_pct ?? 0
  const cpuPct = hw?.cpu_pct ?? 0
  const rss = hw?.process_rss_gb ?? 0

  return (
    <div className="pointer-events-auto fixed bottom-4 left-4 z-40 select-none">
      <div className="bg-black/80 backdrop-blur border border-white/15 rounded-xl shadow-2xl overflow-hidden">
        {/* Header — always visible, click to expand/collapse */}
        <button
          onClick={toggle}
          className="flex items-center gap-2 px-3 py-2 text-white hover:bg-white/5 transition-colors w-[248px]"
          title={open ? 'Collapse hardware HUD' : 'Expand hardware HUD'}
        >
          <Activity size={12} className="text-accent shrink-0" />
          <span className="text-[10px] font-semibold tracking-wide uppercase flex-1 text-left text-white/90">
            Hardware
          </span>
          {hw && (
            <span className="font-mono tabular-nums text-[10px] text-white/70">
              {hw.cpu_pct != null ? `${hw.cpu_pct.toFixed(0)}% cpu` : ''}
              {hw.ram_pct != null ? ` · ${hw.ram_pct.toFixed(0)}% ram` : ''}
            </span>
          )}
          {open ? <ChevronDown size={12} className="text-white/50" /> : <ChevronUp size={12} className="text-white/50" />}
        </button>

        {open && (
          <div className="px-3 pb-2.5 space-y-1.5 border-t border-white/10 pt-2">
            {hw ? (
              <>
                <Row
                  icon={<MemoryStick size={10} />}
                  label="RAM"
                  value={hw.ram_total_gb != null ? `${(hw.ram_used_gb ?? 0).toFixed(1)} / ${hw.ram_total_gb.toFixed(1)} GB` : '—'}
                  pct={ramPct}
                  tone={ramPct > 90 ? 'bg-red-400' : ramPct > 70 ? 'bg-amber-400' : 'bg-emerald-400'}
                />
                <Row
                  icon={<Cpu size={10} />}
                  label="CPU"
                  value={hw.cpu_pct != null ? `${hw.cpu_pct.toFixed(0)}%` : '—'}
                  pct={cpuPct}
                  tone={cpuPct > 85 ? 'bg-red-400' : cpuPct > 50 ? 'bg-amber-400' : 'bg-emerald-400'}
                />
                {gpu ? (
                  <Row
                    icon={<Gauge size={10} />}
                    label="GPU"
                    value={`${gpu.utilization_pct.toFixed(0)}% · ${(gpu.memory_used_mb / 1024).toFixed(1)}/${(gpu.memory_total_mb / 1024).toFixed(0)} GB`}
                    pct={gpu.utilization_pct}
                    tone="bg-sky-400"
                  />
                ) : (
                  <Row icon={<Gauge size={10} />} label="GPU" value={hw.llama_gpu_available ? 'available' : 'CPU-only'} />
                )}
                <div className="flex items-center gap-2 text-[10px] pt-0.5 border-t border-white/10">
                  <span className="text-white/50">Model</span>
                  <span className="font-mono text-white/85 truncate flex-1">
                    {info.loaded ? (info.filename || 'loaded') : 'none'}
                  </span>
                  {hw.process_rss_gb != null && info.loaded && (
                    <span className="font-mono tabular-nums text-accent">{rss.toFixed(2)} GB</span>
                  )}
                </div>
              </>
            ) : (
              <div className="text-[10px] text-white/50 py-1">waiting for telemetry…</div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
