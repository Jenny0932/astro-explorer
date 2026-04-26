import { useEffect, useRef, useState } from 'react'
import './App.css'

type Target = {
  id: string
  name: string
  blurb: string
  topics: string[]
}

type ApodEntry = {
  date: string
  title: string
  explanation: string
  media_type: 'image' | 'video' | string
  copyright?: string
}

type Transient = {
  oid: string
  class: string
  classifier: string
  probability: number
  ndet: number
  firstmjd: number
  lastmjd: number
  ra: number
  dec: number
}

type StampKind = 'science' | 'template' | 'difference'

type DetectionMethod = 'sources' | 'blobs' | 'patches'

type SourceDetection = {
  type: 'source'
  x: number
  y: number
  flux: number
  sharpness: number
  roundness: number
  score: number
}

type BlobDetection = {
  type: 'blob'
  x: number
  y: number
  radius: number
  sigma: number
  score: number
}

type PatchDetection = {
  type: 'patch'
  x: number
  y: number
  w: number
  h: number
  mean_z: number
  std_z: number
  score: number
}

type Detection = SourceDetection | BlobDetection | PatchDetection

type ExplainResponse = {
  explanation: string
  model: string
  crop_size: [number, number]
  tokens: { input: number | null; output: number | null }
}

type AnomalyResponse = {
  image_width: number
  image_height: number
  method: DetectionMethod
  params: Record<string, number>
  detections: Detection[]
  kind?: StampKind
}

const METHODS: { id: DetectionMethod; label: string; hint: string }[] = [
  { id: 'sources', label: 'Sources (DAO)', hint: 'Point-like stars via DAOStarFinder' },
  { id: 'blobs', label: 'Blobs (LoG)', hint: 'Multi-scale Laplacian-of-Gaussian — extended objects' },
  { id: 'patches', label: 'Patches', hint: 'Tile + z-score — unusual regions' },
]

type Selected =
  | { kind: 'target'; target: Target }
  | { kind: 'apod'; apod: ApodEntry }
  | { kind: 'transient'; transient: Transient }

type Tab = 'featured' | 'daily' | 'transients'

const STAMP_KINDS: StampKind[] = ['science', 'template', 'difference']

type OverlayOpts = {
  strokeWidth: number
  fontSize: number
  topN: number
  defaultRadius: number
}

function renderDetection(d: Detection, i: number, opts: OverlayOpts) {
  const highlight = i < opts.topN
  const stroke = highlight ? '#ff4d6d' : '#ffd166'
  const showLabel = highlight && opts.fontSize > 0
  if (d.type === 'patch') {
    return (
      <g key={i}>
        <rect
          x={d.x}
          y={d.y}
          width={d.w}
          height={d.h}
          fill="none"
          stroke={stroke}
          strokeWidth={opts.strokeWidth}
        />
        {showLabel && (
          <text
            x={d.x + 4}
            y={d.y + opts.fontSize}
            fill={stroke}
            fontSize={opts.fontSize}
            fontWeight="bold"
          >
            #{i + 1}
          </text>
        )}
      </g>
    )
  }
  const r = d.type === 'blob' ? d.radius : opts.defaultRadius
  return (
    <g key={i}>
      <circle
        cx={d.x}
        cy={d.y}
        r={r}
        fill="none"
        stroke={stroke}
        strokeWidth={opts.strokeWidth}
      />
      {showLabel && (
        <text
          x={d.x + r + 4}
          y={d.y - r + opts.fontSize * 0.3}
          fill={stroke}
          fontSize={opts.fontSize}
          fontWeight="bold"
        >
          #{i + 1}
        </text>
      )}
    </g>
  )
}

function ExplainCell({
  d,
  i,
  onExplain,
  explainingLabel,
}: {
  d: Detection
  i: number
  onExplain: (d: Detection, label: string) => void
  explainingLabel: string | null
}) {
  const label = `#${i + 1} (${d.type})`
  const busy = explainingLabel === label
  return (
    <td>
      <button
        className="link-btn"
        onClick={() => onExplain(d, label)}
        disabled={busy || i >= 3}
        title={i < 3 ? 'Ask the vision model' : 'Only top-3 detections are explainable'}
      >
        {busy ? '…' : '🤖'}
      </button>
    </td>
  )
}

function DetectionTable({
  detections,
  method,
  onExplain,
  explainingLabel,
}: {
  detections: Detection[]
  method: DetectionMethod
  onExplain: (d: Detection, label: string) => void
  explainingLabel: string | null
}) {
  const rows = detections.slice(0, 10)
  if (method === 'sources') {
    return (
      <div className="anomaly-list">
        <h3>Top sources (brightness + sharpness score)</h3>
        <table>
          <thead>
            <tr><th>#</th><th>x, y</th><th>Flux</th><th>Sharpness</th><th>Score</th><th></th></tr>
          </thead>
          <tbody>
            {rows.map((d, i) => {
              const s = d as SourceDetection
              return (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>({s.x.toFixed(0)}, {s.y.toFixed(0)})</td>
                  <td>{s.flux.toFixed(1)}</td>
                  <td>{s.sharpness.toFixed(2)}</td>
                  <td>{s.score.toFixed(2)}</td>
                  <ExplainCell d={s} i={i} onExplain={onExplain} explainingLabel={explainingLabel} />
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    )
  }
  if (method === 'blobs') {
    return (
      <div className="anomaly-list">
        <h3>Top blobs (LoG multi-scale)</h3>
        <table>
          <thead>
            <tr><th>#</th><th>x, y</th><th>Radius</th><th>σ</th><th>Score</th><th></th></tr>
          </thead>
          <tbody>
            {rows.map((d, i) => {
              const b = d as BlobDetection
              return (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>({b.x.toFixed(0)}, {b.y.toFixed(0)})</td>
                  <td>{b.radius.toFixed(1)}</td>
                  <td>{b.sigma.toFixed(2)}</td>
                  <td>{b.score.toFixed(2)}</td>
                  <ExplainCell d={b} i={i} onExplain={onExplain} explainingLabel={explainingLabel} />
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    )
  }
  return (
    <div className="anomaly-list">
      <h3>Top anomalous patches (|mean z| ∨ |std z|)</h3>
      <table>
        <thead>
          <tr><th>#</th><th>x, y</th><th>Size</th><th>mean z</th><th>std z</th><th>Score</th><th></th></tr>
        </thead>
        <tbody>
          {rows.map((d, i) => {
            const p = d as PatchDetection
            return (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>({p.x.toFixed(0)}, {p.y.toFixed(0)})</td>
                <td>{p.w.toFixed(0)}×{p.h.toFixed(0)}</td>
                <td>{p.mean_z.toFixed(2)}</td>
                <td>{p.std_z.toFixed(2)}</td>
                <td>{p.score.toFixed(2)}</td>
                <ExplainCell d={p} i={i} onExplain={onExplain} explainingLabel={explainingLabel} />
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function mjdToDate(mjd: number): string {
  // MJD 0 = 1858-11-17 00:00 UTC
  const ms = (mjd - 40587) * 86400 * 1000  // 40587 = MJD of 1970-01-01
  return new Date(ms).toISOString().slice(0, 10)
}

type ChatMessage = { role: 'user' | 'assistant'; content: string }

type ChatContext = {
  entity: 'target' | 'apod' | 'transient'
  id: string
  kind?: StampKind
}

function ChatPanel({ context }: { context: ChatContext }) {
  // Parent remounts this component via `key` whenever the selection changes,
  // which naturally resets all state — no reset effect needed.
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight })
  }, [messages, busy])

  const send = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || busy) return
    const next: ChatMessage[] = [...messages, { role: 'user', content: trimmed }]
    setMessages(next)
    setInput('')
    setBusy(true)
    setErr(null)
    try {
      const r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: next,
          entity: context.entity,
          id: context.id,
          kind: context.kind,
        }),
      })
      if (!r.ok) {
        const msg = await r.text().catch(() => '')
        throw new Error(`HTTP ${r.status}: ${msg.slice(0, 200)}`)
      }
      const data = (await r.json()) as { reply: string }
      setMessages([...next, { role: 'assistant', content: data.reply }])
    } catch (e) {
      setErr(`Chat failed: ${e}`)
      setMessages(next.slice(0, -1)) // roll back the user message on error
    } finally {
      setBusy(false)
    }
  }

  const suggestions = [
    'What am I looking at?',
    'What are the most interesting features?',
    'How far away is this?',
    'How was this image captured?',
  ]

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <h3>💬 Ask the astronomer</h3>
        <span className="muted">
          Chat with a vision LLM about this image — and astronomy in general.
        </span>
      </div>

      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && !busy && (
          <div className="chat-empty">
            <p className="muted">
              Ask anything about this image or the universe. The model can see the
              current view.
            </p>
            <div className="chat-suggestions">
              {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  className="chat-suggestion"
                  onClick={() => send(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg chat-msg-${m.role}`}>
            <div className="chat-msg-role">
              {m.role === 'user' ? 'You' : '🔭 Astronomer'}
            </div>
            <div className="chat-msg-body">{m.content}</div>
          </div>
        ))}
        {busy && (
          <div className="chat-msg chat-msg-assistant">
            <div className="chat-msg-role">🔭 Astronomer</div>
            <div className="chat-msg-body muted">Thinking…</div>
          </div>
        )}
      </div>

      {err && <p className="error">{err}</p>}

      <form
        className="chat-input-row"
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
      >
        <input
          type="text"
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about this image or anything in astronomy…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          {busy ? '…' : 'Send'}
        </button>
      </form>
    </div>
  )
}

function App() {
  const [tab, setTab] = useState<Tab>('featured')
  const [targets, setTargets] = useState<Target[]>([])
  const [apod, setApod] = useState<ApodEntry[]>([])
  const [apodFetched, setApodFetched] = useState(false)
  const [transients, setTransients] = useState<Transient[]>([])
  const [transientsFetched, setTransientsFetched] = useState(false)
  const apodLoading = tab === 'daily' && !apodFetched
  const transientsLoading = tab === 'transients' && !transientsFetched
  const [selected, setSelected] = useState<Selected | null>(null)
  const [stampKind, setStampKind] = useState<StampKind>('difference')
  const [method, setMethod] = useState<DetectionMethod>('sources')
  const [anomalies, setAnomalies] = useState<AnomalyResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [explain, setExplain] = useState<ExplainResponse | null>(null)
  const [explainFor, setExplainFor] = useState<string | null>(null) // label of the target being explained
  const [explaining, setExplaining] = useState(false)
  const [showOverlay, setShowOverlay] = useState(true)
  const imgRef = useRef<HTMLImageElement>(null)
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 })

  useEffect(() => {
    fetch('/api/targets')
      .then((r) => r.json())
      .then(setTargets)
      .catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    if (tab !== 'daily' || apodFetched) return
    fetch('/api/apod?days=30')
      .then((r) => r.json())
      .then((data: ApodEntry[]) => setApod(data))
      .catch((e) => setError(String(e)))
      .finally(() => setApodFetched(true))
  }, [tab, apodFetched])

  useEffect(() => {
    if (tab !== 'transients' || transientsFetched) return
    fetch('/api/transients?limit=24')
      .then((r) => r.json())
      .then(setTransients)
      .catch((e) => setError(String(e)))
      .finally(() => setTransientsFetched(true))
  }, [tab, transientsFetched])

  const selectItem = (s: Selected) => {
    setSelected(s)
    setAnomalies(null)
    setError(null)
    setExplain(null)
    setExplainFor(null)
    setImgSize({ w: 0, h: 0 })
    if (s.kind === 'transient') setStampKind('difference')
  }

  const explainUrlFor = (s: Selected, k: StampKind = stampKind): string => {
    if (s.kind === 'target') return `/api/targets/${s.target.id}/explain`
    if (s.kind === 'apod') return `/api/apod/${s.apod.date}/explain`
    return `/api/transients/${s.transient.oid}/explain?kind=${k}`
  }

  const runExplain = async (detection: Detection | null, label: string) => {
    if (!selected) return
    setExplaining(true)
    setExplain(null)
    setExplainFor(label)
    setError(null)
    try {
      const r = await fetch(explainUrlFor(selected), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detection }),
      })
      if (!r.ok) {
        const msg = await r.text().catch(() => '')
        throw new Error(`HTTP ${r.status}: ${msg.slice(0, 200)}`)
      }
      setExplain((await r.json()) as ExplainResponse)
    } catch (e) {
      setError(`Explain failed: ${e}`)
      setExplainFor(null)
    } finally {
      setExplaining(false)
    }
  }

  const imageUrlFor = (s: Selected, k: StampKind = stampKind): string => {
    if (s.kind === 'target') return `/api/targets/${s.target.id}/image`
    if (s.kind === 'apod') return `/api/apod/${s.apod.date}/image`
    return `/api/transients/${s.transient.oid}/stamp/${k}`
  }

  const anomaliesUrlFor = (s: Selected, k: StampKind = stampKind): string => {
    const base =
      s.kind === 'target'
        ? `/api/targets/${s.target.id}/anomalies`
        : s.kind === 'apod'
          ? `/api/apod/${s.apod.date}/anomalies`
          : `/api/transients/${s.transient.oid}/anomalies?kind=${k}`
    return `${base}${base.includes('?') ? '&' : '?'}method=${method}`
  }

  const runAnomalyDetection = async () => {
    if (!selected) return
    if (selected.kind === 'apod' && selected.apod.media_type !== 'image') {
      setError("Can't run detection on a video entry.")
      return
    }
    setLoading(true)
    setError(null)
    try {
      const r = await fetch(anomaliesUrlFor(selected))
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data = (await r.json()) as AnomalyResponse
      setAnomalies(data)
      setShowOverlay(true)
    } catch (e) {
      setError(`Detection failed: ${e}`)
    } finally {
      setLoading(false)
    }
  }

  const onImgLoad = () => {
    if (imgRef.current) {
      setImgSize({
        w: imgRef.current.clientWidth,
        h: imgRef.current.clientHeight,
      })
    }
  }

  const scaleX = anomalies && imgSize.w ? imgSize.w / anomalies.image_width : 1
  const scaleY = anomalies && imgSize.h ? imgSize.h / anomalies.image_height : 1

  const selectedKey =
    selected?.kind === 'target'
      ? `t-${selected.target.id}`
      : selected?.kind === 'apod'
        ? `a-${selected.apod.date}`
        : selected?.kind === 'transient'
          ? `x-${selected.transient.oid}`
          : null

  // Title/blurb/topics per selected kind ---------------------------------------
  const title = !selected
    ? ''
    : selected.kind === 'target'
      ? selected.target.name
      : selected.kind === 'apod'
        ? selected.apod.title
        : `${selected.transient.oid} — ${selected.transient.class}`

  const blurb = !selected
    ? ''
    : selected.kind === 'target'
      ? selected.target.blurb
      : selected.kind === 'apod'
        ? selected.apod.explanation
        : `A ZTF transient classified as ${selected.transient.class} (${(selected.transient.probability * 100).toFixed(0)}% confidence by ${selected.transient.classifier}). ${selected.transient.ndet} detections between ${mjdToDate(selected.transient.firstmjd)} and ${mjdToDate(selected.transient.lastmjd)}. The three cutouts show, left to right: the science image (current sky), the template (historical reference), and the difference (what's new). Bright sources in the difference image are what changed.`

  const topics = selected?.kind === 'target' ? selected.target.topics : []

  return (
    <div className="app">
      <header>
        <h1>🔭 Astro Explorer</h1>
        <p className="tagline">
          Explore the universe through telescope imagery — and find the anomalies.
        </p>
      </header>

      <div className="tabs">
        <button
          className={tab === 'featured' ? 'tab active' : 'tab'}
          onClick={() => setTab('featured')}
        >
          ⭐ Featured JWST
        </button>
        <button
          className={tab === 'daily' ? 'tab active' : 'tab'}
          onClick={() => setTab('daily')}
        >
          🗓️ Daily (APOD)
        </button>
        <button
          className={tab === 'transients' ? 'tab active' : 'tab'}
          onClick={() => setTab('transients')}
        >
          🌟 Transients (ZTF)
        </button>
      </div>

      <div className="layout">
        <aside className="gallery">
          {tab === 'featured' && (
            <>
              <h2>Targets</h2>
              {targets.length === 0 && <p className="muted">Loading…</p>}
              <ul>
                {targets.map((t) => (
                  <li
                    key={t.id}
                    className={selectedKey === `t-${t.id}` ? 'active' : ''}
                    onClick={() => selectItem({ kind: 'target', target: t })}
                  >
                    <strong>{t.name}</strong>
                    <div className="topics">
                      {t.topics.map((topic) => (
                        <span key={topic} className="tag">
                          {topic}
                        </span>
                      ))}
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}

          {tab === 'daily' && (
            <>
              <h2>Last 30 Days</h2>
              {apodLoading && <p className="muted">Loading APOD feed…</p>}
              <div className="apod-grid">
                {apod.map((a) => (
                  <div
                    key={a.date}
                    className={`apod-card ${selectedKey === `a-${a.date}` ? 'active' : ''}`}
                    onClick={() => selectItem({ kind: 'apod', apod: a })}
                  >
                    {a.media_type === 'image' ? (
                      <img
                        src={`/api/apod/${a.date}/image`}
                        alt={a.title}
                        loading="lazy"
                      />
                    ) : (
                      <div className="apod-video">🎬 video</div>
                    )}
                    <div className="apod-meta">
                      <div className="apod-date">{a.date}</div>
                      <div className="apod-title">{a.title}</div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {tab === 'transients' && (
            <>
              <h2>Recent Transients</h2>
              {transientsLoading && (
                <p className="muted">Loading ZTF alerts…</p>
              )}
              <ul>
                {transients.map((t) => (
                  <li
                    key={t.oid}
                    className={selectedKey === `x-${t.oid}` ? 'active' : ''}
                    onClick={() => selectItem({ kind: 'transient', transient: t })}
                  >
                    <div className="tr-row">
                      <strong>{t.oid}</strong>
                      <span className={`tag class-${(t.class || '').replace(/[^a-z0-9]/gi, '')}`}>
                        {t.class}
                      </span>
                    </div>
                    <div className="muted tr-meta">
                      last {mjdToDate(t.lastmjd)} · {t.ndet} dets
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </aside>

        <main className="viewer">
          {!selected && (
            <div className="placeholder">
              <p>
                {tab === 'featured'
                  ? 'Select a target from the left to begin.'
                  : tab === 'daily'
                    ? 'Select a day from the left to begin.'
                    : 'Select a transient from the left to begin.'}
              </p>
            </div>
          )}

          {selected && (
            <>
              <h2>{title}</h2>
              {selected.kind === 'apod' && (
                <p className="muted">
                  {selected.apod.date}
                  {selected.apod.copyright ? ` · © ${selected.apod.copyright}` : ''}
                </p>
              )}
              {selected.kind === 'transient' && (
                <p className="muted">
                  RA {selected.transient.ra.toFixed(4)}°, Dec {selected.transient.dec.toFixed(4)}°
                </p>
              )}
              <p className="blurb">{blurb}</p>

              {selected.kind === 'transient' ? (
                <>
                  <div className="stamp-row">
                    {STAMP_KINDS.map((k) => (
                      <div
                        key={k}
                        className={`stamp ${stampKind === k ? 'selected' : ''}`}
                        onClick={() => {
                          setStampKind(k)
                          setAnomalies(null)
                          setImgSize({ w: 0, h: 0 })
                        }}
                      >
                        <div className="stamp-label">{k}</div>
                        <div className="stamp-img-wrap">
                          <img
                            key={`${selectedKey}-${k}`}
                            ref={stampKind === k ? imgRef : undefined}
                            src={imageUrlFor(selected, k)}
                            alt={k}
                            onLoad={stampKind === k ? onImgLoad : undefined}
                          />
                          {stampKind === k && showOverlay && anomalies && (
                            <svg
                              className="overlay"
                              viewBox={`0 0 ${anomalies.image_width} ${anomalies.image_height}`}
                              preserveAspectRatio="none"
                              style={{ width: imgSize.w, height: imgSize.h }}
                            >
                              {anomalies.detections.map((d, i) =>
                                renderDetection(d, i, {
                                  strokeWidth: 1.5 / Math.min(scaleX, scaleY),
                                  fontSize: 0,
                                  topN: 3,
                                  defaultRadius: Math.max(6, 10 - i * 0.2),
                                }),
                              )}
                            </svg>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="muted">
                    Click a panel to select it — anomaly detection runs on the selected cutout.
                    The <strong>difference</strong> image is where new sources stand out.
                  </p>
                </>
              ) : selected.kind === 'apod' && selected.apod.media_type !== 'image' ? (
                <div className="placeholder">
                  <p>This APOD entry is a video — no image to analyze.</p>
                </div>
              ) : (
                <div className="image-wrapper">
                  <img
                    key={selectedKey ?? 'none'}
                    ref={imgRef}
                    src={imageUrlFor(selected)}
                    alt={title}
                    onLoad={onImgLoad}
                  />
                  {showOverlay && anomalies && (
                    <svg
                      className="overlay"
                      viewBox={`0 0 ${anomalies.image_width} ${anomalies.image_height}`}
                      preserveAspectRatio="none"
                      style={{ width: imgSize.w, height: imgSize.h }}
                    >
                      {anomalies.detections.map((d, i) =>
                        renderDetection(d, i, {
                          strokeWidth: 2 / Math.min(scaleX, scaleY),
                          fontSize: 20 / Math.min(scaleX, scaleY),
                          topN: 5,
                          defaultRadius: Math.max(8, 14 - i * 0.1),
                        }),
                      )}
                    </svg>
                  )}
                </div>
              )}

              <div className="controls">
                <div className="method-picker" role="tablist">
                  {METHODS.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      className={`method-btn ${method === m.id ? 'active' : ''}`}
                      title={m.hint}
                      onClick={() => {
                        setMethod(m.id)
                        setAnomalies(null)
                      }}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
                <button
                  onClick={runAnomalyDetection}
                  disabled={
                    loading ||
                    (selected.kind === 'apod' &&
                      selected.apod.media_type !== 'image')
                  }
                >
                  {loading
                    ? 'Detecting…'
                    : selected.kind === 'transient'
                      ? `🔍 Detect on ${stampKind}`
                      : '🔍 Detect Anomalies'}
                </button>
                {anomalies && (
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={showOverlay}
                      onChange={(e) => setShowOverlay(e.target.checked)}
                    />
                    Show overlay
                  </label>
                )}
                {anomalies && (
                  <span className="muted">
                    {anomalies.detections.length} {anomalies.method} found
                  </span>
                )}
                <button
                  onClick={() => runExplain(null, 'whole image')}
                  disabled={
                    explaining ||
                    (selected.kind === 'apod' &&
                      selected.apod.media_type !== 'image')
                  }
                  className="secondary"
                  title="Ask the vision model to describe the whole image"
                >
                  {explaining && explainFor === 'whole image'
                    ? '🤖 Explaining…'
                    : '🤖 Explain image'}
                </button>
              </div>

              {error && <p className="error">{error}</p>}

              {anomalies && anomalies.detections.length > 0 && (
                <DetectionTable
                  detections={anomalies.detections}
                  method={anomalies.method}
                  onExplain={(d, label) => runExplain(d, label)}
                  explainingLabel={explaining ? explainFor : null}
                />
              )}

              {(explain || (explaining && explainFor)) && (
                <div className="explain-panel">
                  <div className="explain-header">
                    <h3>🤖 Vision model: {explainFor}</h3>
                    {explain && (
                      <span className="muted">
                        {explain.model} · crop {explain.crop_size[0]}×{explain.crop_size[1]}
                        {explain.tokens.input != null && (
                          <> · {explain.tokens.input}→{explain.tokens.output} tok</>
                        )}
                      </span>
                    )}
                  </div>
                  {explaining && !explain && (
                    <p className="muted">Asking the vision model…</p>
                  )}
                  {explain && (
                    <p className="explain-text">{explain.explanation}</p>
                  )}
                </div>
              )}

              {!(selected.kind === 'apod' && selected.apod.media_type !== 'image') && (
                <ChatPanel
                  key={
                    selected.kind === 'transient'
                      ? `x-${selected.transient.oid}-${stampKind}`
                      : (selectedKey ?? 'none')
                  }
                  context={
                    selected.kind === 'target'
                      ? { entity: 'target', id: selected.target.id }
                      : selected.kind === 'apod'
                        ? { entity: 'apod', id: selected.apod.date }
                        : {
                            entity: 'transient',
                            id: selected.transient.oid,
                            kind: stampKind,
                          }
                  }
                />
              )}

              {selected.kind === 'transient' && (
                <div className="learn">
                  <h3>Learn more</h3>
                  <ul>
                    <li>
                      <a
                        href={`https://alerce.online/object/${selected.transient.oid}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Full ALeRCE record →
                      </a>
                    </li>
                    <li>
                      <a
                        href={`https://en.wikipedia.org/wiki/${encodeURIComponent(selected.transient.class)}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        About {selected.transient.class} →
                      </a>
                    </li>
                  </ul>
                </div>
              )}

              {topics.length > 0 && (
                <div className="learn">
                  <h3>Learn more</h3>
                  <ul>
                    {topics.map((topic) => (
                      <li key={topic}>
                        <a
                          href={`https://en.wikipedia.org/wiki/${encodeURIComponent(topic)}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {topic} →
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
