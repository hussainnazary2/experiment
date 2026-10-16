import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Records the screen / app window to a WebM video using the browser's
 * getDisplayMedia + MediaRecorder APIs. Used by the Model Testing Studio so a
 * benchmark run can be recorded and shared.
 *
 * In a packaged Electron build the finished video can be written straight to
 * disk through the `save-recording` IPC bridge; in a plain browser it is
 * offered as a download instead.
 */

const MIME_CANDIDATES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
]

function pickMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return ''
  for (const mime of MIME_CANDIDATES) {
    try {
      if (MediaRecorder.isTypeSupported(mime)) return mime
    } catch {
      /* isTypeSupported can throw on some builds */
    }
  }
  return ''
}

export interface ScreenRecorder {
  supported: boolean
  recording: boolean
  error: string | null
  durationMs: number
  blob: Blob | null
  videoUrl: string | null
  start: () => Promise<void>
  stop: () => void
  reset: () => void
  download: () => void
  /** Save via Electron when available; returns the saved path (or null). */
  save: (filename?: string) => Promise<string | null>
}

export function useScreenRecorder(): ScreenRecorder {
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [durationMs, setDurationMs] = useState(0)
  const [blob, setBlob] = useState<Blob | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const startedAtRef = useRef(0)
  const timerRef = useRef<number | null>(null)

  const supported =
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getDisplayMedia &&
    typeof MediaRecorder !== 'undefined'

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const stopTracks = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }, [])

  const reset = useCallback(() => {
    setBlob(null)
    setVideoUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev)
      return null
    })
    setDurationMs(0)
    setError(null)
  }, [])

  const stop = useCallback(() => {
    const rec = recorderRef.current
    if (rec && rec.state !== 'inactive') {
      rec.stop() // triggers onstop -> blob assembly
    }
  }, [])

  const start = useCallback(async () => {
    if (!supported) {
      setError('Screen recording is not supported in this environment.')
      return
    }
    setError(null)
    reset()
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30 },
        audio: false,
      })
      streamRef.current = stream

      // If the user stops sharing from the browser's own UI, finish cleanly.
      stream.getVideoTracks().forEach((track) => {
        track.addEventListener('ended', () => stop())
      })

      const mimeType = pickMimeType()
      const rec = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      recorderRef.current = rec
      chunksRef.current = []

      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
      }

      rec.onstop = () => {
        const type = pickMimeType() || 'video/webm'
        const out = new Blob(chunksRef.current, { type })
        chunksRef.current = []
        setBlob(out)
        setVideoUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev)
          return URL.createObjectURL(out)
        })
        setRecording(false)
        clearTimer()
        setDurationMs(Date.now() - startedAtRef.current)
        stopTracks()
      }

      rec.onerror = () => {
        setError('Recording failed.')
        setRecording(false)
        clearTimer()
        stopTracks()
      }

      startedAtRef.current = Date.now()
      rec.start(1000) // gather chunks every second
      setRecording(true)
      timerRef.current = window.setInterval(() => {
        setDurationMs(Date.now() - startedAtRef.current)
      }, 500)
    } catch (e) {
      // The user dismissing the picker also lands here; keep the message short.
      setError(e instanceof Error ? e.message : 'Could not start screen recording.')
      setRecording(false)
      stopTracks()
    }
  }, [supported, reset, stop, clearTimer, stopTracks])

  const download = useCallback(() => {
    if (!videoUrl) return
    const stamp = new Date().toISOString().replace(/[:.]/g, '-')
    const a = document.createElement('a')
    a.href = videoUrl
    a.download = `gguf-speed-test-${stamp}.webm`
    document.body.appendChild(a)
    a.click()
    a.remove()
  }, [videoUrl])

  const save = useCallback(
    async (filename?: string): Promise<string | null> => {
      if (!blob) return null
      const api = (window as any).electronAPI
      const name = filename || `gguf-speed-test-${Date.now()}.webm`
      if (api?.saveRecording) {
        try {
          const buffer = await blob.arrayBuffer()
          return await api.saveRecording(name, buffer)
        } catch (e) {
          setError(e instanceof Error ? e.message : 'Failed to save recording.')
          return null
        }
      }
      download()
      return null
    },
    [blob, download],
  )

  useEffect(() => {
    return () => {
      clearTimer()
      const rec = recorderRef.current
      if (rec && rec.state !== 'inactive') {
        try {
          rec.stop()
        } catch {
          /* ignore */
        }
      }
      stopTracks()
    }
  }, [clearTimer, stopTracks])

  return {
    supported,
    recording,
    error,
    durationMs,
    blob,
    videoUrl,
    start,
    stop,
    reset,
    download,
    save,
  }
}
