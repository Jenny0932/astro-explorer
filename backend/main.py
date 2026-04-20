"""Astro Explorer backend — JWST press-release gallery + APOD feed + CV anomaly detection."""

from __future__ import annotations

import base64
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
from openai import OpenAI, OpenAIError
from photutils.detection import DAOStarFinder
from PIL import Image
from pydantic import BaseModel
from skimage.feature import blob_log

NASA_API_KEY = os.environ.get("NASA_API_KEY", "DEMO_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
_OPENAI_CLIENT: OpenAI | None = None


def _openai() -> OpenAI:
    global _OPENAI_CLIENT
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set. Export it in the backend environment to enable /explain.",
        )
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


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


SUPPORTED_METHODS = ("sources", "blobs", "patches")


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
            "type": "source",
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


def _detect_blobs(
    gray: np.ndarray, min_sigma: float, max_sigma: float, threshold: float
) -> list[dict[str, Any]]:
    """Multi-scale Laplacian-of-Gaussian blobs; catches extended sources DAO misses."""
    norm = gray / 255.0
    blobs = blob_log(
        norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=10,
        threshold=threshold,
    )
    if len(blobs) == 0:
        return []

    _, median, std = sigma_clipped_stats(gray, sigma=3.0)
    h, w = gray.shape
    results: list[dict[str, Any]] = []
    for y, x, sigma in blobs:
        yi, xi = int(round(y)), int(round(x))
        yi = max(0, min(h - 1, yi))
        xi = max(0, min(w - 1, xi))
        # Local brightness above background, normalized by sky noise.
        score = float((gray[yi, xi] - median) / (std + 1e-6))
        results.append(
            {
                "type": "blob",
                "x": float(x),
                "y": float(y),
                "radius": float(sigma * np.sqrt(2)),
                "sigma": float(sigma),
                "score": score,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:50]


def _detect_patches(
    gray: np.ndarray, patch_size: int, top_n: int
) -> list[dict[str, Any]]:
    """Tile the image and z-score each patch by mean + stddev — flags unusual regions."""
    h, w = gray.shape
    ny, nx = max(1, h // patch_size), max(1, w // patch_size)
    if ny == 0 or nx == 0:
        return []

    trimmed = gray[: ny * patch_size, : nx * patch_size]
    tiles = trimmed.reshape(ny, patch_size, nx, patch_size).swapaxes(1, 2)
    means = tiles.mean(axis=(2, 3))
    stds = tiles.std(axis=(2, 3))

    mean_z = (means - np.median(means)) / (np.std(means) + 1e-6)
    std_z = (stds - np.median(stds)) / (np.std(stds) + 1e-6)
    # A patch is "anomalous" if it's unusually bright *or* unusually textured.
    score = np.maximum(np.abs(mean_z), np.abs(std_z))

    flat = [
        {
            "type": "patch",
            "x": float(j * patch_size),
            "y": float(i * patch_size),
            "w": float(patch_size),
            "h": float(patch_size),
            "mean_z": float(mean_z[i, j]),
            "std_z": float(std_z[i, j]),
            "score": float(score[i, j]),
        }
        for i in range(ny)
        for j in range(nx)
    ]
    flat.sort(key=lambda r: r["score"], reverse=True)
    return flat[:top_n]


def _fetch_gray(url: str) -> np.ndarray:
    try:
        content, _ = _download_image(url)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {e}") from e
    img = Image.open(io.BytesIO(content)).convert("L")
    return np.asarray(img, dtype=np.float32)


def _zscore(gray: np.ndarray) -> np.ndarray:
    """Sigma-clipped robust z-score: sky background has mean 0, std 1."""
    _, median, std = sigma_clipped_stats(gray, sigma=3.0)
    return (gray - median) / (std + 1e-6)


def _normalized_residual(science: np.ndarray, template: np.ndarray) -> np.ndarray:
    """z-score science and template separately, subtract, remap signed residual to 0-255.

    Each image's sky background is brought to mean 0, std 1 before
    subtraction, so the residual is dominated by what actually changed
    rather than global scaling. Polarity is preserved:

      positive residual (new source)   -> pixel > 128 (bright)
      zero residual (unchanged sky)    -> pixel ~ 128
      negative residual (disappeared)  -> pixel < 128 (dark)

    Downstream bright-peak detectors (DAO, blob_log) then find only new
    appearances, which is the more diagnostic signal for transients.
    Clip range is set by the 99th percentile of |residual| to stay robust
    to a few saturated pixels.
    """
    h = min(science.shape[0], template.shape[0])
    w = min(science.shape[1], template.shape[1])
    residual = _zscore(science[:h, :w]) - _zscore(template[:h, :w])
    p99 = float(np.percentile(np.abs(residual), 99))
    scaled = np.clip(residual, -p99, p99) / (p99 + 1e-6) * 127.0 + 128.0
    return scaled.astype(np.float32)


def _run_detection_on_gray(
    gray: np.ndarray,
    method: str,
    fwhm: float,
    nsigma: float,
    min_sigma: float,
    max_sigma: float,
    blob_threshold: float,
    patch_size: int,
    top_n: int,
) -> dict[str, Any]:
    if method not in SUPPORTED_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"method must be one of {SUPPORTED_METHODS}",
        )
    h, w = gray.shape

    if method == "sources":
        detections = _detect_sources(gray, fwhm=fwhm, nsigma=nsigma)
        params = {"fwhm": fwhm, "nsigma": nsigma}
    elif method == "blobs":
        detections = _detect_blobs(
            gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=blob_threshold
        )
        params = {
            "min_sigma": min_sigma,
            "max_sigma": max_sigma,
            "threshold": blob_threshold,
        }
    else:  # patches
        detections = _detect_patches(gray, patch_size=patch_size, top_n=top_n)
        params = {"patch_size": patch_size, "top_n": top_n}

    return {
        "image_width": w,
        "image_height": h,
        "method": method,
        "params": params,
        "detections": detections,
    }


def _run_detection_on_url(
    url: str,
    method: str,
    fwhm: float,
    nsigma: float,
    min_sigma: float,
    max_sigma: float,
    blob_threshold: float,
    patch_size: int,
    top_n: int,
) -> dict[str, Any]:
    gray = _fetch_gray(url)
    return _run_detection_on_gray(
        gray,
        method,
        fwhm,
        nsigma,
        min_sigma,
        max_sigma,
        blob_threshold,
        patch_size,
        top_n,
    )


# -- Vision explanation (OpenAI) -----------------------------------------------

VISION_SYSTEM_PROMPT = (
    "You are an astronomy image analyst reviewing a grayscale crop from a "
    "telescope image (JWST press-release, NASA APOD, or a ZTF transient stamp). "
    "A classical computer-vision detector flagged a region as unusual. Your job: "
    "in 3-5 short sentences, describe (1) what you visually see in the crop "
    "(point source, extended blob, diffraction spike, edge artifact, noise, etc.), "
    "(2) the most likely astronomical or instrumental interpretation, and "
    "(3) how confident you are and why. Avoid speculation beyond what is visible. "
    "Do not invent coordinates or magnitudes. Be concise and specific."
)


def _crop_for_explanation(
    pil_img: Image.Image, detection: dict[str, Any] | None, pad: int = 48
) -> Image.Image:
    """Return a crop centered on the detection (with padding), or a resized full image."""
    max_side = 512
    if detection is None:
        img = pil_img.copy()
    elif detection.get("type") == "patch":
        x = int(detection.get("x", 0))
        y = int(detection.get("y", 0))
        w = int(detection.get("w", 48))
        h = int(detection.get("h", 48))
        img = pil_img.crop(
            (
                max(0, x - pad),
                max(0, y - pad),
                min(pil_img.width, x + w + pad),
                min(pil_img.height, y + h + pad),
            )
        )
    else:
        cx = int(detection.get("x", pil_img.width // 2))
        cy = int(detection.get("y", pil_img.height // 2))
        r = int(detection.get("radius", 16))
        half = max(64, r * 4)
        img = pil_img.crop(
            (
                max(0, cx - half),
                max(0, cy - half),
                min(pil_img.width, cx + half),
                min(pil_img.height, cy + half),
            )
        )
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


def _explain_image(
    pil_img: Image.Image,
    detection: dict[str, Any] | None,
    context: str,
) -> dict[str, Any]:
    """Send crop + context to OpenAI vision and return the explanation."""
    crop = _crop_for_explanation(pil_img, detection)
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    det_line = ""
    if detection is not None:
        det_line = f"\nDetector flagged a {detection.get('type', 'region')} at image coords ({detection.get('x')}, {detection.get('y')})"
        if "score" in detection:
            det_line += f" with score {detection['score']:.2f}"
        det_line += "."

    user_text = (
        f"Context: {context}{det_line}\n"
        "Please describe what you see in this crop and what it likely is."
    )

    try:
        resp = _openai().chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=350,
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                },
            ],
        )
    except OpenAIError as e:
        raise HTTPException(
            status_code=502, detail=f"OpenAI vision call failed: {e}"
        ) from e

    usage = resp.usage
    return {
        "model": OPENAI_MODEL,
        "explanation": (resp.choices[0].message.content or "").strip(),
        "crop_size": list(crop.size),
        "tokens": {
            "input": usage.prompt_tokens if usage else None,
            "output": usage.completion_tokens if usage else None,
        },
    }


class ExplainRequest(BaseModel):
    detection: dict[str, Any] | None = None
    context: str | None = None


def _pil_from_gray(gray: np.ndarray) -> Image.Image:
    arr = np.clip(gray, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _pil_from_url(url: str) -> Image.Image:
    try:
        content, _ = _download_image(url)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Image fetch failed: {e}") from e
    return Image.open(io.BytesIO(content)).convert("L")


@app.get("/api/targets/{target_id}/anomalies")
def detect_anomalies(
    target_id: str,
    method: str = Query("sources"),
    fwhm: float = Query(3.0, ge=1.0, le=20.0),
    nsigma: float = Query(5.0, ge=1.0, le=20.0),
    min_sigma: float = Query(2.0, ge=0.5, le=20.0),
    max_sigma: float = Query(15.0, ge=1.0, le=50.0),
    blob_threshold: float = Query(0.05, ge=0.001, le=1.0),
    patch_size: int = Query(48, ge=8, le=256),
    top_n: int = Query(30, ge=1, le=200),
) -> JSONResponse:
    t = _find_target(target_id)
    result = _run_detection_on_url(
        t["image_url"],
        method,
        fwhm,
        nsigma,
        min_sigma,
        max_sigma,
        blob_threshold,
        patch_size,
        top_n,
    )
    return JSONResponse({"target_id": target_id, **result})


@app.post("/api/targets/{target_id}/explain")
def explain_target(target_id: str, req: ExplainRequest) -> dict[str, Any]:
    t = _find_target(target_id)
    pil = _pil_from_url(t["image_url"])
    context = f"JWST target: {t['name']}. {t['blurb']}"
    if req.context:
        context += f" Additional note: {req.context}"
    return _explain_image(pil, req.detection, context)


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
    method: str = Query("sources"),
    fwhm: float = Query(3.0, ge=1.0, le=20.0),
    nsigma: float = Query(5.0, ge=1.0, le=20.0),
    min_sigma: float = Query(2.0, ge=0.5, le=20.0),
    max_sigma: float = Query(15.0, ge=1.0, le=50.0),
    blob_threshold: float = Query(0.05, ge=0.001, le=1.0),
    patch_size: int = Query(48, ge=8, le=256),
    top_n: int = Query(30, ge=1, le=200),
) -> JSONResponse:
    entries = _fetch_apod_range(30)
    e = _find_apod(entries, apod_date)
    if e["media_type"] != "image":
        raise HTTPException(status_code=415, detail="APOD entry is not an image")
    result = _run_detection_on_url(
        e["image_url"],
        method,
        fwhm,
        nsigma,
        min_sigma,
        max_sigma,
        blob_threshold,
        patch_size,
        top_n,
    )
    return JSONResponse({"date": apod_date, **result})


@app.post("/api/apod/{apod_date}/explain")
def explain_apod(apod_date: str, req: ExplainRequest) -> dict[str, Any]:
    entries = _fetch_apod_range(30)
    e = _find_apod(entries, apod_date)
    if e["media_type"] != "image":
        raise HTTPException(status_code=415, detail="APOD entry is not an image")
    pil = _pil_from_url(e["image_url"])
    context = (
        f"NASA Astronomy Picture of the Day, {e.get('date')}: "
        f"{e.get('title')}. {e.get('explanation', '')}"
    )
    if req.context:
        context += f" Additional note: {req.context}"
    return _explain_image(pil, req.detection, context)


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
    method: str = Query("sources"),
    fwhm: float = Query(2.0, ge=1.0, le=20.0),
    nsigma: float = Query(3.0, ge=1.0, le=20.0),
    min_sigma: float = Query(1.0, ge=0.5, le=20.0),
    max_sigma: float = Query(8.0, ge=1.0, le=50.0),
    blob_threshold: float = Query(0.05, ge=0.001, le=1.0),
    patch_size: int = Query(16, ge=4, le=64),
    top_n: int = Query(15, ge=1, le=100),
) -> JSONResponse:
    if kind not in ("science", "template", "difference"):
        raise HTTPException(
            status_code=400, detail="kind must be science|template|difference"
        )
    candid = _latest_stamped_candid(oid)
    detection_input = kind
    if kind == "difference":
        # Bypass ALeRCE's display-stretched diff PNG: z-score science and
        # template independently, subtract, and detect on |residual|.
        sci_url = f"{ALERCE_STAMPS}?oid={oid}&candid={candid}&type=science&format=png"
        tmp_url = f"{ALERCE_STAMPS}?oid={oid}&candid={candid}&type=template&format=png"
        gray = _normalized_residual(_fetch_gray(sci_url), _fetch_gray(tmp_url))
        result = _run_detection_on_gray(
            gray,
            method,
            fwhm,
            nsigma,
            min_sigma,
            max_sigma,
            blob_threshold,
            patch_size,
            top_n,
        )
        detection_input = "normalized-residual"
    else:
        url = f"{ALERCE_STAMPS}?oid={oid}&candid={candid}&type={kind}&format=png"
        result = _run_detection_on_url(
            url,
            method,
            fwhm,
            nsigma,
            min_sigma,
            max_sigma,
            blob_threshold,
            patch_size,
            top_n,
        )
    return JSONResponse(
        {
            "oid": oid,
            "kind": kind,
            "candid": candid,
            "detection_input": detection_input,
            **result,
        }
    )


@app.post("/api/transients/{oid}/explain")
def explain_transient(
    oid: str,
    req: ExplainRequest,
    kind: str = Query("difference"),
) -> dict[str, Any]:
    if kind not in ("science", "template", "difference"):
        raise HTTPException(
            status_code=400, detail="kind must be science|template|difference"
        )
    candid = _latest_stamped_candid(oid)
    if kind == "difference":
        sci_url = f"{ALERCE_STAMPS}?oid={oid}&candid={candid}&type=science&format=png"
        tmp_url = f"{ALERCE_STAMPS}?oid={oid}&candid={candid}&type=template&format=png"
        gray = _normalized_residual(_fetch_gray(sci_url), _fetch_gray(tmp_url))
        pil = _pil_from_gray(gray)
        input_note = (
            "normalized residual (science z-score − template z-score, "
            "remapped so 128 = zero, brighter = new source)"
        )
    else:
        url = f"{ALERCE_STAMPS}?oid={oid}&candid={candid}&type={kind}&format=png"
        pil = _pil_from_url(url)
        input_note = f"ALeRCE {kind} stamp"
    context = (
        f"ZTF transient {oid}, {kind} cutout ({input_note}). "
        "These are ~60x60 arcsec cutouts around a variable/transient source."
    )
    if req.context:
        context += f" Additional note: {req.context}"
    return _explain_image(pil, req.detection, context)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
