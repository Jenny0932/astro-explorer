# Astro Explorer

Explore the universe via three complementary feeds — curated JWST press-release
imagery, NASA's Astronomy Picture of the Day, and live ZTF transient alerts —
all with classical-CV anomaly detection and astronomy learning links.

## Environment

Copy the example file and fill in your keys:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

`start.sh` auto-sources `.env` on launch. `.env` is gitignored.

Variables:

| Var | Required? | Default | Purpose |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | yes, for `/explain` | — | Vision-LLM anomaly explanations. Get one at [platform.openai.com](https://platform.openai.com/api-keys). The rest of the app works without it; `/explain` returns 503 until set. |
| `OPENAI_VISION_MODEL` | no | `gpt-4o-mini` | Swap to `gpt-4o` for richer explanations at ~20× the cost. |
| `NASA_API_KEY` | no | `DEMO_KEY` | Lifts the APOD rate limit beyond 50 req/day. Free at [api.nasa.gov](https://api.nasa.gov/). |

## Run

```bash
./start.sh
```

Starts backend (port 8000) and frontend (port 5173), auto-installs frontend deps
on first run, and shuts both down on Ctrl-C. Open http://localhost:5173.

### Run manually

**Backend** (FastAPI + astropy + photutils, managed with `uv`):

```bash
cd backend
uv run uvicorn main:app --reload --port 8000
```

**Frontend** (React + Vite + TypeScript):

```bash
cd frontend
npm install   # first time only
npm run dev
```

## Development

Install [pre-commit](https://pre-commit.com) hooks to run ruff (backend), ESLint
and `tsc` (frontend), and general file hygiene checks on every commit:

```bash
uv tool install pre-commit   # or: brew install pre-commit
pre-commit install
pre-commit run --all-files   # optional: run against the whole repo
```

See `.pre-commit-config.yaml` for the hook definitions.

## Three feeds

- **⭐ Featured JWST** — 5 curated press-release images (Wikimedia-hosted):
  SMACS 0723, Cosmic Cliffs (NGC 3324), Southern Ring Nebula, Stephan's Quintet,
  Pillars of Creation.
- **🗓️ Daily (APOD)** — last 30 days of NASA's Astronomy Picture of the Day.
  Updates daily with an expert-written explanation for each image.
- **🌟 Transients (ZTF)** — recent Zwicky Transient Facility alerts via the
  [ALeRCE broker](https://alerce.online). Shows the classic 3-panel cutout:
  science (current sky), template (historical reference), and difference
  (what's new). Anomaly detection runs on whichever panel you select — the
  difference image is where brand-new supernovae stand out most clearly.

## API

- `GET /api/targets`, `GET /api/targets/{id}/{image,anomalies}`, `POST /api/targets/{id}/explain`
- `GET /api/apod?days=30`, `GET /api/apod/{date}/{image,anomalies}`, `POST /api/apod/{date}/explain`
- `GET /api/transients?limit=24`,
  `GET /api/transients/{oid}/{candid,stamp/{kind},anomalies}`,
  `POST /api/transients/{oid}/explain?kind=...`
  where `kind ∈ {science, template, difference}`

All image endpoints proxy through the backend so the browser stays on one
origin. Anomaly detection accepts `method=sources|blobs|patches`:

- **sources** — `photutils.DAOStarFinder` on a sigma-clipped background; ranks point-like sources by `0.7·flux_z + 0.3·sharpness_z`.
- **blobs** — `skimage.feature.blob_log` multi-scale Laplacian-of-Gaussian; catches extended objects.
- **patches** — tile the image, z-score each tile by mean and std; flag the top `max(|mean_z|, |std_z|)`.

For ZTF `kind=difference`, detection runs on a locally-computed *normalized residual* (z-score science and template independently, subtract, remap so 128 = zero) rather than ALeRCE's display-stretched diff PNG — so bright-peak detectors focus on new appearances, not disappearances.

`POST /{entity}/explain` sends a 512px crop around an optional `{detection}` object (or the whole image if omitted) to an OpenAI vision model and returns a short natural-language explanation.

## Project layout

```
astro-explorer/
├── start.sh              # starts both servers
├── backend/
│   ├── pyproject.toml    # uv-managed deps
│   └── main.py           # FastAPI app
└── frontend/
    ├── package.json
    └── src/App.tsx       # gallery + viewer + overlay
```

## Notes

- "Flux" on press-release images is pixel intensity after color-composite
  processing — not a calibrated astrophysical measurement. Good enough to spot
  bright stars, diffraction spikes, compact galaxies, and visually unusual
  features. ZTF difference cutouts are closer to real science-grade pixels.
- ZTF data is public and hosted by ALeRCE; their list endpoint can be slow (~2s
  for the first uncached hit). Results are cached for 30 min on our side.
- Next steps: fetch calibrated `*_i2d.fits` from MAST and render with an asinh
  stretch for true multi-filter composites; hook TNS for named supernovae with
  spectroscopic confirmations.
