"""Astro Explorer backend — JWST press-release gallery + APOD feed + CV anomaly detection."""

from __future__ import annotations

import io
import os
import time
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import requests
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from photutils.detection import DAOStarFinder
from PIL import Image

NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")

app = FastAPI(title="Astro Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

USER_AGENT = "astro-explorer/0.1 (educational use; Wikimedia press-release images)"

# Curated JWST targets — press-release images hosted on Wikimedia Commons.
# These are the finished color composites NASA/STScI/ESA published, not raw pipeline previews.
CURATED_TARGETS: list[dict[str, Any]] = [
    {
        "id": "smacs-0723",
        "name": "SMACS 0723 — Webb's First Deep Field",
        "blurb": (
            "The first full-color image released from JWST. A galaxy cluster "
            "4.6 billion light-years away acting as a gravitational lens, "
            "magnifying even more distant galaxies behind it."
        ),
        "topics": ["gravitational lensing", "deep field", "galaxy cluster"],
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/"
            "Webb%27s_First_Deep_Field.jpg/1280px-Webb%27s_First_Deep_Field.jpg"
        ),
    },
    {
        "id": "carina-nebula",
        "name": "Cosmic Cliffs — Carina Nebula (NGC 3324)",
        "blurb": (
            "A stellar nursery ~7,600 light-years away. The 'cliffs' are the "
            "edge of a bubble carved by radiation from hot young stars."
        ),
        "topics": ["star formation", "nebula", "H II region"],
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/"
            "NASA%E2%80%99s_Webb_Reveals_Cosmic_Cliffs%2C_Glittering_Landscape_of_Star_Birth.jpg/"
            "1280px-NASA%E2%80%99s_Webb_Reveals_Cosmic_Cliffs%2C_Glittering_Landscape_of_Star_Birth.jpg"
        ),
    },
    {
        "id": "southern-ring",
        "name": "Southern Ring Nebula (NGC 3132)",
        "blurb": (
            "A planetary nebula — the expanding gas shells of a dying Sun-like "
            "star. JWST revealed the central star is actually a binary system."
        ),
        "topics": ["planetary nebula", "stellar death", "binary stars"],
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/"
            "Southern_Ring_Nebula_%28NIRCam_Image%29.png/"
            "1280px-Southern_Ring_Nebula_%28NIRCam_Image%29.png"
        ),
    },
    {
        "id": "stephans-quintet",
        "name": "Stephan's Quintet",
        "blurb": (
            "A visual grouping of five galaxies; four are locked in a cosmic "
            "dance, producing shock waves and triggering new star formation."
        ),
        "topics": ["galaxy interaction", "shock waves", "compact group"],
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/"
            "Stephan%27s_Quintet_%28MIRI_Image%29_%282022-034-01G7DBCJA1M1SSGKDMH7F5XMBE%29.png/"
            "1280px-Stephan%27s_Quintet_%28MIRI_Image%29_%282022-034-01G7DBCJA1M1SSGKDMH7F5XMBE%29.png"
        ),
    },
    {
        "id": "pillars-of-creation",
        "name": "Pillars of Creation",
        "blurb": (
            "Columns of cool gas and dust in the Eagle Nebula, actively forming "
            "new stars. JWST's infrared view pierces the dust to reveal protostars."
        ),
        "topics": ["star formation", "protostars", "molecular cloud"],
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/"
            "Pillars_of_Creation_%28NIRCam_Image%29.jpg/"
            "1280px-Pillars_of_Creation_%28NIRCam_Image%29.jpg"
        ),
    },
]


@app.get("/api/targets")
def list_targets() -> list[dict[str, Any]]:
    # Don't leak image_url — the frontend uses our proxy endpoint instead.
    return [{k: v for k, v in t.items() if k != "image_url"} for t in CURATED_TARGETS]


def _find_target(target_id: str) -> dict[str, Any]:
    for t in CURATED_TARGETS:
        if t["id"] == target_id:
            return t
    raise HTTPException(status_code=404, detail=f"Unknown target: {target_id}")


@lru_cache(maxsize=8)
def _download_image(url: str) -> tuple[bytes, str]:
    """Download (cached) the image bytes + content-type for a given URL."""
    r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "image/jpeg")


@app.get("/api/targets/{target_id}/image")
def get_target_image(target_id: str) -> Response:
    """Proxy the press-release image so the frontend stays on our origin."""
    t = _find_target(target_id)
    try:
        content, content_type = _download_image(t["image_url"])
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {e}") from e
    return Response(content=content, media_type=content_type)


def _detect_sources(
    gray: np.ndarray, fwhm: float, nsigma: float
) -> list[dict[str, Any]]:
    """DAOStarFinder on a sigma-clipped background; score by brightness + sharpness."""
    _, median, std = sigma_clipped_stats(gray, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=nsigma * std)
    tbl = finder(gray - median)
    if tbl is None or len(tbl) == 0:
        return []

    fluxes = np.array(tbl["flux"])
    sharps = np.abs(np.array(tbl["sharpness"]))
    flux_z = (fluxes - np.median(fluxes)) / (np.std(fluxes) + 1e-6)
    sharp_z = (sharps - np.median(sharps)) / (np.std(sharps) + 1e-6)
    score = 0.7 * flux_z + 0.3 * sharp_z

    results = [
        {
            "x": float(tbl["xcentroid"][i]),
            "y": float(tbl["ycentroid"][i]),
            "flux": float(tbl["flux"][i]),
            "sharpness": float(tbl["sharpness"][i]),
            "roundness": float(tbl["roundness1"][i]),
            "score": float(score[i]),
        }
        for i in range(len(tbl))
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:50]


def _run_detection_on_url(url: str, fwhm: float, nsigma: float) -> dict[str, Any]:
    try:
        content, _ = _download_image(url)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {e}") from e
    img = Image.open(io.BytesIO(content)).convert("L")
    gray = np.asarray(img, dtype=np.float32)
    return {
        "image_width": img.width,
        "image_height": img.height,
        "params": {"fwhm": fwhm, "nsigma": nsigma},
        "sources": _detect_sources(gray, fwhm=fwhm, nsigma=nsigma),
    }


@app.get("/api/targets/{target_id}/anomalies")
def detect_anomalies(
    target_id: str,
    fwhm: float = Query(3.0, ge=1.0, le=20.0),
    nsigma: float = Query(5.0, ge=1.0, le=20.0),
) -> JSONResponse:
    t = _find_target(target_id)
    result = _run_detection_on_url(t["image_url"], fwhm, nsigma)
    return JSONResponse({"target_id": target_id, **result})


# -- APOD (Astronomy Picture of the Day) ---------------------------------------

# Cache the APOD feed for 1 hour so we don't hammer NASA's API.
_APOD_CACHE: dict[str, Any] = {"ts": 0.0, "days": 0, "data": []}


def _fetch_apod_range(days: int) -> list[dict[str, Any]]:
    now = time.time()
    if (
        _APOD_CACHE["data"]
        and _APOD_CACHE["days"] == days
        and now - _APOD_CACHE["ts"] < 3600
    ):
        return _APOD_CACHE["data"]

    end = date.today()
    start = end - timedelta(days=days - 1)
    try:
        r = requests.get(
            "https://api.nasa.gov/planetary/apod",
            params={
                "api_key": NASA_API_KEY,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "thumbs": "true",
            },
            timeout=30,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"NASA APOD fetch failed: {e}"
        ) from e

    entries = r.json()
    # Normalize: only keep what the frontend needs, newest first.
    normalized = []
    for e in entries:
        img_url = e.get("hdurl") or e.get("url") or e.get("thumbnail_url")
        if not img_url:
            continue
        normalized.append(
            {
                "date": e.get("date"),
                "title": e.get("title"),
                "explanation": e.get("explanation"),
                "media_type": e.get("media_type"),
                "image_url": img_url,
                "copyright": e.get("copyright"),
            }
        )
    normalized.sort(key=lambda x: x["date"] or "", reverse=True)
    _APOD_CACHE.update({"ts": now, "days": days, "data": normalized})
    return normalized


def _find_apod(entries: list[dict[str, Any]], d: str) -> dict[str, Any]:
    for e in entries:
        if e["date"] == d:
            return e
    raise HTTPException(status_code=404, detail=f"APOD not found for date: {d}")


@app.get("/api/apod")
def list_apod(days: int = Query(30, ge=1, le=30)) -> list[dict[str, Any]]:
    entries = _fetch_apod_range(days)
    # Strip image_url — frontend hits our proxy instead.
    return [{k: v for k, v in e.items() if k != "image_url"} for e in entries]


@app.get("/api/apod/{apod_date}/image")
def get_apod_image(apod_date: str) -> Response:
    entries = _fetch_apod_range(30)
    e = _find_apod(entries, apod_date)
    if e["media_type"] != "image":
        raise HTTPException(status_code=415, detail="APOD entry is not an image")
    try:
        content, content_type = _download_image(e["image_url"])
    except requests.RequestException as ex:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {ex}") from ex
    return Response(content=content, media_type=content_type)


@app.get("/api/apod/{apod_date}/anomalies")
def detect_apod_anomalies(
    apod_date: str,
    fwhm: float = Query(3.0, ge=1.0, le=20.0),
    nsigma: float = Query(5.0, ge=1.0, le=20.0),
) -> JSONResponse:
    entries = _fetch_apod_range(30)
    e = _find_apod(entries, apod_date)
    if e["media_type"] != "image":
        raise HTTPException(status_code=415, detail="APOD entry is not an image")
    result = _run_detection_on_url(e["image_url"], fwhm, nsigma)
    return JSONResponse({"date": apod_date, **result})


# -- ZTF transients via ALeRCE broker ------------------------------------------
#
# ALeRCE (alerce.online) publishes public ZTF alert data: recent variable/transient
# objects plus the classic "science / template / difference" image cutouts that
# astronomers use to find new sources. The difference image is the real magic —
# it's "science minus reference," so only things that *changed* remain.

ALERCE_API = "https://api.alerce.online/ztf/v1"
ALERCE_STAMPS = "https://avro.alerce.online/get_stamp"
_TRANSIENT_LIST_CACHE: dict[str, Any] = {"ts": 0.0, "key": None, "data": []}
_TRANSIENT_CANDID_CACHE: dict[str, str] = {}  # oid -> latest candid with a stamp


def _fetch_transient_list(limit: int, class_filter: str | None) -> list[dict[str, Any]]:
    """Recently first-detected ZTF objects, sorted client-side by last detection."""
    now = time.time()
    key = f"{limit}:{class_filter or ''}"
    if _TRANSIENT_LIST_CACHE["key"] == key and now - _TRANSIENT_LIST_CACHE["ts"] < 1800:
        return _TRANSIENT_LIST_CACHE["data"]

    # ALeRCE's server-side sort is very slow, but the firstmjd filter is indexed.
    # Pull the last ~60 days of newly-discovered objects, then sort locally.
    mjd_today = Time(date.today().isoformat()).mjd
    params: dict[str, Any] = {
        "page_size": max(limit * 3, 60),  # overfetch so we can filter+sort locally
        "ranking": 1,
        "firstmjd": int(mjd_today - 60),
    }
    if class_filter:
        params["class"] = class_filter
    try:
        r = requests.get(f"{ALERCE_API}/objects/", params=params, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"ALeRCE fetch failed: {e}") from e

    items = r.json().get("items", [])
    items = [it for it in items if (it.get("ndet") or 0) >= 3]
    items.sort(key=lambda it: it.get("lastmjd") or 0, reverse=True)
    items = items[:limit]

    normalized = [
        {
            "oid": it["oid"],
            "class": it.get("class"),
            "classifier": it.get("classifier"),
            "probability": it.get("probability"),
            "ndet": it.get("ndet"),
            "firstmjd": it.get("firstmjd"),
            "lastmjd": it.get("lastmjd"),
            "ra": it.get("meanra"),
            "dec": it.get("meandec"),
        }
        for it in items
    ]
    _TRANSIENT_LIST_CACHE.update({"ts": now, "key": key, "data": normalized})
    return normalized


def _latest_stamped_candid(oid: str) -> str:
    """Find the most recent detection for `oid` that has a stamp, and return its candid."""
    if oid in _TRANSIENT_CANDID_CACHE:
        return _TRANSIENT_CANDID_CACHE[oid]
    try:
        r = requests.get(f"{ALERCE_API}/objects/{oid}/lightcurve", timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502, detail=f"ALeRCE lightcurve failed: {e}"
        ) from e

    dets = [d for d in r.json().get("detections", []) if d.get("has_stamp")]
    if not dets:
        raise HTTPException(status_code=404, detail=f"No stamped detection for {oid}")
    latest = max(dets, key=lambda d: d.get("mjd", 0))
    candid = str(latest["candid"])
    _TRANSIENT_CANDID_CACHE[oid] = candid
    return candid


@app.get("/api/transients")
def list_transients(
    limit: int = Query(24, ge=1, le=100),
    class_filter: str | None = Query(None, alias="class"),
) -> list[dict[str, Any]]:
    return _fetch_transient_list(limit, class_filter)


@app.get("/api/transients/{oid}/candid")
def get_transient_candid(oid: str) -> dict[str, str]:
    return {"oid": oid, "candid": _latest_stamped_candid(oid)}


@app.get("/api/transients/{oid}/stamp/{kind}")
def get_transient_stamp(oid: str, kind: str) -> Response:
    if kind not in ("science", "template", "difference"):
        raise HTTPException(
            status_code=400, detail="kind must be science|template|difference"
        )
    candid = _latest_stamped_candid(oid)
    url = f"{ALERCE_STAMPS}?oid={oid}&candid={candid}&type={kind}&format=png"
    try:
        content, content_type = _download_image(url)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Stamp fetch failed: {e}") from e
    return Response(content=content, media_type=content_type)


@app.get("/api/transients/{oid}/anomalies")
def detect_transient_anomalies(
    oid: str,
    kind: str = Query("difference"),
    fwhm: float = Query(2.0, ge=1.0, le=20.0),
    nsigma: float = Query(3.0, ge=1.0, le=20.0),
) -> JSONResponse:
    if kind not in ("science", "template", "difference"):
        raise HTTPException(
            status_code=400, detail="kind must be science|template|difference"
        )
    candid = _latest_stamped_candid(oid)
    url = f"{ALERCE_STAMPS}?oid={oid}&candid={candid}&type={kind}&format=png"
    result = _run_detection_on_url(url, fwhm, nsigma)
    return JSONResponse({"oid": oid, "kind": kind, "candid": candid, **result})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
