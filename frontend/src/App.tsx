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

type Source = {
  x: number
  y: number
  flux: number
  sharpness: number
  roundness: number
  score: number
}

type AnomalyResponse = {
  image_width: number
  image_height: number
  params: { fwhm: number; nsigma: number }
  sources: Source[]
}

type Selected =
  | { kind: 'target'; target: Target }
  | { kind: 'apod'; apod: ApodEntry }

type Tab = 'featured' | 'daily'

function selectedImageUrl(s: Selected): string {
  return s.kind === 'target'
    ? `/api/targets/${s.target.id}/image`
    : `/api/apod/${s.apod.date}/image`
}

function selectedAnomaliesUrl(s: Selected): string {
  return s.kind === 'target'
    ? `/api/targets/${s.target.id}/anomalies`
    : `/api/apod/${s.apod.date}/anomalies`
}

function selectedTitle(s: Selected): string {
  return s.kind === 'target' ? s.target.name : s.apod.title
}

function selectedBlurb(s: Selected): string {
  return s.kind === 'target' ? s.target.blurb : s.apod.explanation
}

function selectedTopics(s: Selected): string[] {
  return s.kind === 'target' ? s.target.topics : []
}

function App() {
  const [tab, setTab] = useState<Tab>('featured')
  const [targets, setTargets] = useState<Target[]>([])
  const [apod, setApod] = useState<ApodEntry[]>([])
  const [apodLoading, setApodLoading] = useState(false)
  const [selected, setSelected] = useState<Selected | null>(null)
  const [anomalies, setAnomalies] = useState<AnomalyResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
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
    if (tab !== 'daily' || apod.length > 0) return
    setApodLoading(true)
    fetch('/api/apod?days=30')
      .then((r) => r.json())
      .then((data: ApodEntry[]) => setApod(data))
      .catch((e) => setError(String(e)))
      .finally(() => setApodLoading(false))
  }, [tab, apod.length])

  const selectItem = (s: Selected) => {
    setSelected(s)
    setAnomalies(null)
    setError(null)
    setImgSize({ w: 0, h: 0 })
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
      const r = await fetch(selectedAnomaliesUrl(selected))
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
        : null

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
          🗓️ Daily from Space (APOD)
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
        </aside>

        <main className="viewer">
          {!selected && (
            <div className="placeholder">
              <p>
                {tab === 'featured'
                  ? 'Select a target from the left to begin.'
                  : 'Select a day from the left to begin.'}
              </p>
            </div>
          )}

          {selected && (
            <>
              <h2>{selectedTitle(selected)}</h2>
              {selected.kind === 'apod' && (
                <p className="muted">
                  {selected.apod.date}
                  {selected.apod.copyright ? ` · © ${selected.apod.copyright}` : ''}
                </p>
              )}
              <p className="blurb">{selectedBlurb(selected)}</p>

              {selected.kind === 'apod' && selected.apod.media_type !== 'image' ? (
                <div className="placeholder">
                  <p>This APOD entry is a video — no image to analyze.</p>
                </div>
              ) : (
                <div className="image-wrapper">
                  <img
                    key={selectedKey ?? 'none'}
                    ref={imgRef}
                    src={selectedImageUrl(selected)}
                    alt={selectedTitle(selected)}
                    onLoad={onImgLoad}
                  />
                  {showOverlay && anomalies && (
                    <svg
                      className="overlay"
                      viewBox={`0 0 ${anomalies.image_width} ${anomalies.image_height}`}
                      preserveAspectRatio="none"
                      style={{ width: imgSize.w, height: imgSize.h }}
                    >
                      {anomalies.sources.map((s, i) => (
                        <g key={i}>
                          <circle
                            cx={s.x}
                            cy={s.y}
                            r={Math.max(8, 14 - i * 0.1)}
                            fill="none"
                            stroke={i < 5 ? '#ff4d6d' : '#ffd166'}
                            strokeWidth={2 / Math.min(scaleX, scaleY)}
                          />
                          {i < 5 && (
                            <text
                              x={s.x + 16}
                              y={s.y - 10}
                              fill="#ff4d6d"
                              fontSize={20 / Math.min(scaleX, scaleY)}
                              fontWeight="bold"
                            >
                              #{i + 1}
                            </text>
                          )}
                        </g>
                      ))}
                    </svg>
                  )}
                </div>
              )}

              <div className="controls">
                <button
                  onClick={runAnomalyDetection}
                  disabled={
                    loading ||
                    (selected.kind === 'apod' &&
                      selected.apod.media_type !== 'image')
                  }
                >
                  {loading ? 'Detecting…' : '🔍 Detect Anomalies'}
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
                    {anomalies.sources.length} sources found
                  </span>
                )}
              </div>

              {error && <p className="error">{error}</p>}

              {anomalies && anomalies.sources.length > 0 && (
                <div className="anomaly-list">
                  <h3>Top anomalies (brightness + sharpness score)</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>x, y</th>
                        <th>Flux</th>
                        <th>Sharpness</th>
                        <th>Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {anomalies.sources.slice(0, 10).map((s, i) => (
                        <tr key={i}>
                          <td>{i + 1}</td>
                          <td>
                            ({s.x.toFixed(0)}, {s.y.toFixed(0)})
                          </td>
                          <td>{s.flux.toFixed(1)}</td>
                          <td>{s.sharpness.toFixed(2)}</td>
                          <td>{s.score.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {selectedTopics(selected).length > 0 && (
                <div className="learn">
                  <h3>Learn more</h3>
                  <ul>
                    {selectedTopics(selected).map((topic) => (
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
