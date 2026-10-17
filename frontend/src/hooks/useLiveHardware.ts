import { useEffect, useRef, useState } from 'react'
import { gpuApi } from '../api/client'

/** Live hardware telemetry from /api/gpu/live (polled ~1 Hz). */
export interface LiveHardware {
  ram_used_gb: number | null
  ram_total_gb: number | null
  ram_pct: number | null
  cpu_pct: number | null
  /** This backend process's RSS — a good proxy for loaded-model memory on CPU. */
  process_rss_gb: number | null
  gpu: {
    name: string
    utilization_pct: number
    memory_used_mb: number
    memory_total_mb: number
    temperature_c: number | null
  } | null
  llama_gpu_available: boolean
}

/**
 * Poll /api/gpu/live at ~1 Hz while `active`. Errors are swallowed (the
 * HUD is cosmetic — losing the backend must not crash the app).
 */
export function useLiveHardware(active = true, intervalMs = 1000) {
  const [hw, setHw] = useState<LiveHardware | null>(null)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (!active) {
      if (timer.current !== null) window.clearInterval(timer.current)
      timer.current = null
      return
    }
    let alive = true
    const tick = async () => {
      try {
        const data = await gpuApi.live()
        if (alive) setHw(data as LiveHardware)
      } catch { /* backend unreachable — keep last snapshot */ }
    }
    void tick()
    timer.current = window.setInterval(tick, intervalMs)
    return () => {
      alive = false
      if (timer.current !== null) window.clearInterval(timer.current)
      timer.current = null
    }
  }, [active, intervalMs])

  return hw
}
