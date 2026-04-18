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
  kind?: StampKind
}

type Selected =
  | { kind: 'target'; target: Target }
  | { kind: 'apod'; apod: ApodEntry }
  | { kind: 'transient'; transient: Transient }

type Tab = 'featured' | 'daily' | 'transients'

const STAMP_KINDS: StampKind[] = ['science', 'template', 'difference']

function mjdToDate(mjd: number): string {
  // MJD 0 = 1858-11-17 00:00 UTC
  const ms = (mjd - 40587) * 86400 * 1000  // 40587 = MJD of 1970-01-01
  return new Date(ms).toISOString().slice(0, 10)
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
    setImgSize({ w: 0, h: 0 })
    if (s.kind === 'transient') setStampKind('difference')
  }

  const imageUrlFor = (s: Selected, k: StampKind = stampKind): string => {
    if (s.kind === 'target') return `/api/targets/${s.target.id}/image`
    if (s.kind === 'apod') return `/api/apod/${s.apod.date}/image`
    return `/api/transients/${s.transient.oid}/stamp/${k}`
  }

  const anomaliesUrlFor = (s: Selected, k: StampKind = stampKind): string => {
    if (s.kind === 'target') return `/api/targets/${s.target.id}/anomalies`
    if (s.kind === 'apod') return `/api/apod/${s.apod.date}/anomalies`
    return `/api/transients/${s.transient.oid}/anomalies?kind=${k}`
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
                              {anomalies.sources.map((s, i) => (
                                <circle
                                  key={i}
                                  cx={s.x}
                                  cy={s.y}
                                  r={Math.max(6, 10 - i * 0.2)}
                                  fill="none"
                                  stroke={i < 3 ? '#ff4d6d' : '#ffd166'}
                                  strokeWidth={1.5 / Math.min(scaleX, scaleY)}
                                />
                              ))}
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
                    {anomalies.sources.length} sources found
                  </span>
                )}
              </div>

              {error && <p className="error">{error}</p>}

              {anomalies && anomalies.sources.length > 0 && (
                <div className="anomaly-list">
                  <h3>Top sources (brightness + sharpness score)</h3>
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
