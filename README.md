# Astro Explorer

Explore iconic JWST press-release imagery with classical-CV anomaly detection and astronomy learning links.

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

## How it works

- `GET /api/targets` — 5 curated JWST targets with educational blurbs and topic tags.
- `GET /api/targets/{id}/image` — proxy that returns the NASA/STScI/ESA press-release
  color image (hosted on Wikimedia Commons). Served through the backend so the
  frontend stays on one origin and images are cached in memory.
- `GET /api/targets/{id}/anomalies?fwhm=3&nsigma=5` — grayscale-converts the image,
  estimates the sky background with `sigma_clipped_stats`, runs `DAOStarFinder`,
  and ranks sources by `0.7·flux_z + 0.3·sharpness_z`. Returns up to the top 50.

The UI draws circles over detected sources (top 5 in red with rank labels,
rest in yellow) and shows a table of the top 10 with flux/sharpness/score.

## Targets

1. SMACS 0723 — Webb's First Deep Field
2. Cosmic Cliffs — Carina Nebula (NGC 3324)
3. Southern Ring Nebula (NGC 3132)
4. Stephan's Quintet
5. Pillars of Creation

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

- Anomaly detection runs on the 2D grayscale of the press-release JPG/PNG. That
  means "flux" here is pixel intensity after color-composite processing — not a
  calibrated astrophysical measurement. Good enough to spot bright stars,
  diffraction spikes, compact galaxies, and visually unusual features.
- To work with calibrated science data (real fluxes, WCS coordinates, multi-filter
  composites), the next step is fetching `*_i2d.fits` from MAST and rendering with
  an asinh stretch in the backend.
