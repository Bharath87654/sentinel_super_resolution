"""
FastAPI Backend for 4× Sentinel-2 Super-Resolution Pipeline

Scientific Scope
────────────────
This service exposes DEPLOYMENT INFERENCE for the trained 4× SR model.
Input: REAL 10m Sentinel-2 four-band imagery (B02, B03, B04, B08).
Output: 2.5m-scale INFERRED product.

IMPORTANT: The output is a model-inferred product. It does NOT represent
true 2.5m ground-truth imagery. The model was supervised exclusively on
a synthetic 40m-equivalent LR4x → 10m HR benchmark.
"""

import io
import json
import re
import shutil
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
import rasterio
from rasterio.transform import Affine
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ─── Path Registration ────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_project_root / "src" / "models"))

from msr_resnet_4x import MSRResNet4x

# ─── Configuration ────────────────────────────────────────────────────────────
CHECKPOINT_PATH = _project_root / "outputs" / "checkpoints" / "checkpoint_best_4x.pth"
API_JOBS_DIR    = _project_root / "outputs" / "api_jobs"
QUALITY_MAPS_DIR = _project_root / "outputs" / "quality_maps_4x"
MAX_FILE_SIZE   = 20 * 1024 * 1024  # 20 MB

API_JOBS_DIR.mkdir(parents=True, exist_ok=True)

app_state = {}


# ─── Lifespan (Startup / Shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    # Load metadata and instantiate on CPU solely for parameter counting
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model_config = ckpt["model_config"]
    epoch = ckpt.get("epoch", "?")
    
    model = MSRResNet4x(**model_config)
    total_params = sum(p.numel() for p in model.parameters())
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    app_state["model_info"] = {
        "model_name": "MSRResNet4x",
        "checkpoint_filename": CHECKPOINT_PATH.name,
        "epoch": epoch,
        "parameters": total_params,
        "device": device,
        "scale_factor": model_config.get("scale_factor", 4),
        "input_bands": ["B02", "B03", "B04", "B08"],
        "input_gsd_m": 10,
        "output_gsd_m": 2.5,
        "scientific_scope": (
            "Synthetic 40m-equivalent LR4x -> 10m HR training benchmark. "
            "Deployment output is a 2.5m-scale model-inferred product, "
            "not genuine 2.5m ground-truth imagery."
        )
    }
    
    print("\n" + "=" * 60)
    print("FastAPI Server Startup: 4× SR Pipeline")
    print("=" * 60)
    print(f"Model      : {app_state['model_info']['model_name']}")
    print(f"Checkpoint : {app_state['model_info']['checkpoint_filename']}")
    print(f"Epoch      : {app_state['model_info']['epoch']}")
    print(f"Device     : {app_state['model_info']['device']}")
    print(f"Parameters : {app_state['model_info']['parameters']:,}")
    print("=" * 60 + "\n")
    
    yield


# ─── FastAPI App Initialization ───────────────────────────────────────────────
app = FastAPI(
    title="Sentinel-2 4× Super-Resolution API",
    description="Infers a 2.5m-scale product from 10m Sentinel-2 imagery.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173",
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the isolated jobs directory for frontend retrieval
app.mount("/outputs", StaticFiles(directory=str(API_JOBS_DIR)), name="outputs")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health Check")
def health():
    """Returns basic service health and pipeline resolution capability."""
    return {
        "status": "ok",
        "model": "MSRResNet4x",
        "scale_factor": 4,
        "input_gsd_m": 10,
        "output_gsd_m": 2.5
    }


@app.get("/model-info", summary="Model Metadata")
def model_info():
    """Returns the loaded MSRResNet4x checkpoint metadata and parameters."""
    return app_state["model_info"]


@app.post("/infer", summary="Run 4× Inference on 10m Patch")
def infer(
    file: UploadFile = File(..., description="The .npy (4,64,64) Sentinel-2 patch"),
    metadata_file: UploadFile = File(..., description="The .json metadata from extraction")
):
    """
    Accepts a 10m Sentinel-2 `.npy` patch (uint16) AND its genuine extraction 
    `.json` metadata. Generates a 2.5m-scale model-inferred product, including 
    a georeferenced GeoTIFF, RGB preview, and quality maps.
    """
    # 1. Validate NPY Extension & Size
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Image upload must be a .npy file.")
    
    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20 MB.")
    
    # 2. Validate Metadata Extension
    if not metadata_file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Metadata upload must be a .json file.")
    
    try:
        meta_dict = json.load(metadata_file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in metadata file.")
        
    # 3. Enforce Mandatory Geospatial Metadata
    req_keys = {"crs", "patch_transform", "bands", "shape", "pixel_size_m"}
    if not req_keys.issubset(meta_dict.keys()):
        missing = req_keys - meta_dict.keys()
        raise HTTPException(status_code=400, detail=f"Metadata missing required keys: {missing}")
        
    if meta_dict["shape"] != [4, 64, 64]:
        raise HTTPException(
            status_code=400, 
            detail=f"Metadata shape invalid. Expected [4, 64, 64], got {meta_dict['shape']}"
        )
        
    if meta_dict["bands"] != ["B02", "B03", "B04", "B08"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Metadata bands invalid. Expected B02,B03,B04,B08, got {meta_dict['bands']}"
        )
        
    if meta_dict["pixel_size_m"] != 10:
        raise HTTPException(
            status_code=400, 
            detail=f"Metadata pixel_size_m invalid. Expected 10, got {meta_dict['pixel_size_m']}"
        )

    # 4. Load and Validate NumPy Array
    try:
        arr = np.load(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid .npy file format.")
        
    if arr.shape != (4, 64, 64):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid payload shape. Expected (4, 64, 64), got {arr.shape}."
        )
        
    if arr.dtype != np.uint16:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid payload dtype. Expected uint16, got {arr.dtype}."
        )
        
    arr_f = arr.astype(np.float32)
    if np.isnan(arr_f).any() or np.isinf(arr_f).any():
        raise HTTPException(status_code=400, detail="Input payload contains NaN or Inf values.")
        
    # 5. Create Isolated Job Directory
    job_id  = str(uuid.uuid4())
    job_dir = API_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Use job_id as the filename prefix to isolate quality_maps script outputs
    input_npy_path  = job_dir / f"{job_id}.npy"
    input_meta_path = job_dir / f"{job_id}_metadata.json"
    
    np.save(input_npy_path, arr)
    with open(input_meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2)
        
    # Generate LR RGB Preview
    try:
        lr_rgb = arr[[2, 1, 0], :, :].astype(np.float32)
        lr_rgb = lr_rgb.transpose(1, 2, 0)
        p2 = float(np.percentile(lr_rgb, 2.0))
        p98 = float(np.percentile(lr_rgb, 98.0))
        lr_rgb_stretch = np.clip((lr_rgb - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
        lr_rgb_uint8 = (lr_rgb_stretch * 255.0).astype(np.uint8)
        lr_preview_path = job_dir / f"{job_id}_lr_rgb.png"
        Image.fromarray(lr_rgb_uint8).save(lr_preview_path)
    except Exception as e:
        print(f"Failed to generate LR preview: {e}")
        
    # ─── Execute Inference Subprocess ─────────────────────────────────────────
    infer_cmd = [
        sys.executable, str(_project_root / "src" / "inference" / "infer_4x.py"),
        "--input", str(input_npy_path),
        "--output_dir", str(job_dir)
    ]
    
    try:
        res = subprocess.run(infer_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("\n[Inference Error STDOUT]\n", e.stdout)
        print("\n[Inference Error STDERR]\n", e.stderr)
        raise HTTPException(
            status_code=500, 
            detail="Inference script failed internally. See server logs."
        )
        
    stdout = res.stdout
    
    def extract_float(pattern, text, default=0.0):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else default
        
    inf_time    = extract_float(r"Inference time\s*:\s*([\d\.]+)", stdout)
    ref_min     = extract_float(r"Reflectance min:\s*([\-\d\.]+)", stdout)
    ref_max     = extract_float(r"Reflectance max:\s*([\-\d\.]+)", stdout)
    ref_mean    = extract_float(r"Reflectance avg:\s*([\-\d\.]+)", stdout)
    clipped_pct = extract_float(r"Total uint16-clipped percentage:\s*([\d\.]+)", stdout)
    
    # ─── Execute GeoTIFF Subprocess ───────────────────────────────────────────
    sr_ref_path  = job_dir / f"{job_id}_sr_reflectance.npy"
    out_tif_path = job_dir / f"{job_id}_sr.tif"
    
    geotiff_cmd = [
        sys.executable, str(_project_root / "src" / "inference" / "export_geotiff_4x.py"),
        "--reflectance", str(sr_ref_path),
        "--metadata", str(input_meta_path),
        "--output", str(out_tif_path)
    ]
    
    try:
        subprocess.run(geotiff_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("\n[GeoTIFF Error STDOUT]\n", e.stdout)
        print("\n[GeoTIFF Error STDERR]\n", e.stderr)
        raise HTTPException(
            status_code=500, 
            detail="GeoTIFF export failed internally. See server logs."
        )
        
    # ─── Execute Quality Maps Subprocess ──────────────────────────────────────
    quality_cmd = [
        sys.executable, str(_project_root / "src" / "evaluation" / "quality_maps_4x.py"),
        "--real_input", str(input_npy_path)
    ]
    
    quality_metrics = {}
    quality_map_urls = {}
    q_error = None
    
    try:
        subprocess.run(quality_cmd, capture_output=True, text=True, check=True)
        # Move generated files from outputs/quality_maps_4x/real_<job_id> to job_dir/quality_maps
        src_qm_dir = QUALITY_MAPS_DIR / f"real_{job_id}"
        dest_qm_dir = job_dir / "quality_maps"
        
        if src_qm_dir.exists():
            shutil.move(str(src_qm_dir), str(dest_qm_dir))
            
            # Read the metrics
            q_metrics_path = dest_qm_dir / "quality_metrics.json"
            if q_metrics_path.exists():
                with open(q_metrics_path, "r") as f:
                    quality_metrics = json.load(f)
            
            # Find generated PNGs and map to URLs
            for png_file in dest_qm_dir.glob("*.png"):
                quality_map_urls[png_file.stem] = f"/outputs/{job_id}/quality_maps/{png_file.name}"
                
    except subprocess.CalledProcessError as e:
        print("\n[Quality Maps Error STDOUT]\n", e.stdout)
        print("\n[Quality Maps Error STDERR]\n", e.stderr)
        q_error = "Quality maps generation failed internally."
        
    # ─── Return API Response ──────────────────────────────────────────────────
    response = {
        "status": "success",
        "job_id": job_id,
        "input_shape": [4, 64, 64],
        "output_shape": [4, 256, 256],
        "input_gsd_m": 10,
        "output_gsd_m": 2.5,
        "inference_time_sec": inf_time,
        "reflectance_min": ref_min,
        "reflectance_max": ref_max,
        "reflectance_mean": ref_mean,
        "clipped_uint16_pct": clipped_pct,
        "source_safe": meta_dict.get("source_safe", "UNKNOWN"),
        "source_row": meta_dict.get("row_off", -1),
        "source_col": meta_dict.get("col_off", -1),
        "crs": meta_dict.get("crs", "UNKNOWN"),
        "scientific_scope": (
            "2.5m-scale model-inferred product, not ground-truth 2.5m imagery. "
            "TTA disagreement is a proxy, not calibrated probabilistic uncertainty."
        ),
        "artifacts": {
            "lr_rgb_preview":  f"/outputs/{job_id}/{job_id}_lr_rgb.png",
            "rgb_preview":     f"/outputs/{job_id}/{job_id}_sr_rgb.png",
            "geotiff":         f"/outputs/{job_id}/{job_id}_sr.tif",
            "reflectance_npy": f"/outputs/{job_id}/{job_id}_sr_reflectance.npy",
            "uint16_npy":      f"/outputs/{job_id}/{job_id}_sr_uint16.npy"
        },
        "quality_maps": quality_map_urls,
        "quality_metrics": quality_metrics
    }
    
    if q_error:
        response["warnings"] = [q_error]
        
    return response


# ─── /infer-geotiff Endpoint ──────────────────────────────────────────────────
REQUIRED_BANDS      = ["B02", "B03", "B04", "B08"]
PATCH_SIZE          = 64
PIXEL_SIZE_NOMINAL  = 10.0        # metres
PIXEL_SIZE_TOLERANCE = 0.15       # ±15 % (handles minor CRS/reprojection drift)
MAX_GEOTIFF_SIZE    = 200 * 1024 * 1024   # 200 MB — scenes can be large


@app.post("/infer-geotiff", summary="Run 4× Inference on a 4-band Sentinel-2 GeoTIFF")
def infer_geotiff(
    geotiff: UploadFile = File(
        ...,
        description=(
            "A 4-band Sentinel-2 GeoTIFF with bands in order B02, B03, B04, B08 "
            "at 10 m resolution. The centre 64×64 patch will be extracted and processed."
        )
    )
):
    """
    Accepts a single 4-band Sentinel-2 GeoTIFF (B02, B03, B04, B08 in that exact order)
    at 10 m resolution.  The centre 64×64 pixel patch is extracted automatically.
    Downstream inference, GeoTIFF export, and quality-map generation are identical to
    the /infer endpoint. The original /infer endpoint (NPY + JSON) is preserved.

    Scientific note: Output is a 2.5m-scale model-inferred product, not ground-truth.
    """
    # ── 1. Extension check ────────────────────────────────────────────────────
    fname = (geotiff.filename or "").lower()
    if not (fname.endswith(".tif") or fname.endswith(".tiff")):
        raise HTTPException(
            status_code=400,
            detail=(
                "Upload must be a GeoTIFF file (.tif or .tiff). "
                "JPG, PNG, and other formats are not supported because they "
                "cannot carry four spectral bands and geospatial metadata."
            )
        )

    # ── 2. Size check (read raw bytes once) ───────────────────────────────────
    raw_bytes = geotiff.file.read()
    if len(raw_bytes) > MAX_GEOTIFF_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum accepted size is 200 MB."
        )
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── 3. Open with rasterio from an in-memory buffer ────────────────────────
    try:
        with rasterio.MemoryFile(raw_bytes) as mem:
            with mem.open() as src:
                n_bands    = src.count
                width      = src.width
                height     = src.height
                crs        = src.crs
                transform  = src.transform       # Affine
                descriptions = [
                    (d or "").strip() for d in (src.descriptions or [])
                ]
                px_x = abs(transform.a)
                px_y = abs(transform.e)

                # ── 3a. Band count ────────────────────────────────────────────
                if n_bands != 4:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"GeoTIFF has {n_bands} band(s). Exactly 4 bands are required "
                            f"in the order B02, B03, B04, B08. "
                            f"Standard RGB (3-band) or single-band files are not accepted."
                        )
                    )

                # ── 3b. Band name / order validation ─────────────────────────
                # Accept descriptions that match exactly, or upper-cased equivalents.
                normalised = [d.upper() for d in descriptions]
                required_upper = [b.upper() for b in REQUIRED_BANDS]

                if normalised != required_upper:
                    # Provide actionable guidance rather than a vague error.
                    found_str = (
                        ", ".join(descriptions) if any(descriptions) else "none (no band descriptions embedded)"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Band descriptions do not match the required order. "
                            f"Required: {', '.join(REQUIRED_BANDS)} (in that exact order). "
                            f"Found: {found_str}. "
                            "Please export your GeoTIFF with explicit band descriptions set to "
                            "B02, B03, B04, B08 in that order. "
                            "In GDAL/rasterio you can set band descriptions using "
                            "`dst.set_band_description(i, name)` when writing."
                        )
                    )

                # ── 3c. CRS presence ─────────────────────────────────────────
                if crs is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "GeoTIFF has no CRS (coordinate reference system). "
                            "Please provide a properly georeferenced Sentinel-2 file."
                        )
                    )

                # ── 3d. Pixel size (10 m ± tolerance) ────────────────────────
                lo = PIXEL_SIZE_NOMINAL * (1.0 - PIXEL_SIZE_TOLERANCE)
                hi = PIXEL_SIZE_NOMINAL * (1.0 + PIXEL_SIZE_TOLERANCE)
                if not (lo <= px_x <= hi and lo <= px_y <= hi):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Pixel size is approximately {px_x:.2f}×{px_y:.2f} m. "
                            f"Expected ~10 m Sentinel-2 data (accepted range: {lo:.1f}–{hi:.1f} m). "
                            "High-resolution aerial or drone imagery is not supported."
                        )
                    )

                # ── 3e. Minimum dimensions ────────────────────────────────────
                if width < PATCH_SIZE or height < PATCH_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"GeoTIFF dimensions are {width}×{height} pixels. "
                            f"A minimum of {PATCH_SIZE}×{PATCH_SIZE} pixels is required."
                        )
                    )

                # ── 4. Extract centre 64×64 patch ────────────────────────────
                row_start = (height - PATCH_SIZE) // 2
                col_start = (width  - PATCH_SIZE) // 2
                window = rasterio.windows.Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=PATCH_SIZE,
                    height=PATCH_SIZE
                )
                patch_arr = src.read(window=window)   # shape (4, 64, 64)

                # Derive the affine transform for this exact patch
                patch_transform = rasterio.windows.transform(window, transform)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded file as a GeoTIFF: {exc}"
        )

    # ── 5. dtype → uint16 (model expects uint16 or float32 reflectance) ───────
    if patch_arr.dtype == np.uint16:
        arr_u16 = patch_arr
    elif patch_arr.dtype in (np.float32, np.float64):
        # Assume values are already reflectance (0–1 or Sentinel-2 DN / 10000).
        arr_f = patch_arr.astype(np.float32)
        arr_u16 = np.clip(np.round(arr_f * 10000.0), 0, 65535).astype(np.uint16)
    elif np.issubdtype(patch_arr.dtype, np.integer):
        arr_u16 = patch_arr.astype(np.uint16)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported band dtype '{patch_arr.dtype}'. Expected uint16 or float32."
        )

    if np.isnan(arr_u16.astype(np.float32)).any():
        raise HTTPException(status_code=400, detail="Patch contains NaN values.")

    # ── 6. Shape double-check ─────────────────────────────────────────────────
    if arr_u16.shape != (4, PATCH_SIZE, PATCH_SIZE):
        raise HTTPException(
            status_code=500,
            detail=f"Internal patch extraction error: got shape {arr_u16.shape}."
        )

    # ── 7. Synthesise metadata JSON (identical schema to extract_real_patch_4x) ──
    pt = patch_transform
    meta_dict = {
        "source_safe":    geotiff.filename or "uploaded_geotiff",
        "bands":          REQUIRED_BANDS,
        "shape":          [4, PATCH_SIZE, PATCH_SIZE],
        "pixel_size_m":   PIXEL_SIZE_NOMINAL,
        "crs":            str(crs),
        "patch_transform": [pt.a, pt.b, pt.c, pt.d, pt.e, pt.f],
        "row_off":        (height - PATCH_SIZE) // 2,
        "col_off":        (width  - PATCH_SIZE) // 2,
        "geotiff_source_dimensions": [height, width],
        "extraction_note": (
            "Centre 64×64 patch extracted automatically from uploaded GeoTIFF. "
            "Not the full scene."
        )
    }

    # ── 8. Persist to isolated job directory ──────────────────────────────────
    job_id  = str(uuid.uuid4())
    job_dir = API_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_npy_path  = job_dir / f"{job_id}.npy"
    input_meta_path = job_dir / f"{job_id}_metadata.json"

    np.save(input_npy_path, arr_u16)
    with open(input_meta_path, "w") as f:
        json.dump(meta_dict, f, indent=2)

    # ── 9. LR RGB preview ────────────────────────────────────────────────────
    try:
        lr_rgb = arr_u16[[2, 1, 0], :, :].astype(np.float32)   # B04, B03, B02
        lr_rgb = lr_rgb.transpose(1, 2, 0)
        p2  = float(np.percentile(lr_rgb, 2.0))
        p98 = float(np.percentile(lr_rgb, 98.0))
        lr_rgb_stretch = np.clip((lr_rgb - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
        lr_rgb_uint8   = (lr_rgb_stretch * 255.0).astype(np.uint8)
        lr_preview_path = job_dir / f"{job_id}_lr_rgb.png"
        Image.fromarray(lr_rgb_uint8).save(lr_preview_path)
    except Exception as e:
        print(f"[/infer-geotiff] Failed to generate LR preview: {e}")

    # ── 10. Inference subprocess (identical to /infer) ─────────────────────
    infer_cmd = [
        sys.executable, str(_project_root / "src" / "inference" / "infer_4x.py"),
        "--input",      str(input_npy_path),
        "--output_dir", str(job_dir)
    ]
    try:
        res = subprocess.run(infer_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("\n[/infer-geotiff Inference Error STDOUT]\n", e.stdout)
        print("\n[/infer-geotiff Inference Error STDERR]\n", e.stderr)
        raise HTTPException(status_code=500, detail="Inference script failed. See server logs.")

    stdout = res.stdout

    def _extract_float(pattern, text, default=0.0):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    inf_time    = _extract_float(r"Inference time\s*:\s*([\d\.]+)", stdout)
    ref_min     = _extract_float(r"Reflectance min:\s*([\-\d\.]+)", stdout)
    ref_max     = _extract_float(r"Reflectance max:\s*([\-\d\.]+)", stdout)
    ref_mean    = _extract_float(r"Reflectance avg:\s*([\-\d\.]+)", stdout)
    clipped_pct = _extract_float(r"Total uint16-clipped percentage:\s*([\d\.]+)", stdout)

    # ── 11. GeoTIFF export subprocess ────────────────────────────────────────
    sr_ref_path  = job_dir / f"{job_id}_sr_reflectance.npy"
    out_tif_path = job_dir / f"{job_id}_sr.tif"

    geotiff_cmd = [
        sys.executable, str(_project_root / "src" / "inference" / "export_geotiff_4x.py"),
        "--reflectance", str(sr_ref_path),
        "--metadata",    str(input_meta_path),
        "--output",      str(out_tif_path)
    ]
    try:
        subprocess.run(geotiff_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("\n[/infer-geotiff GeoTIFF Error STDOUT]\n", e.stdout)
        print("\n[/infer-geotiff GeoTIFF Error STDERR]\n", e.stderr)
        raise HTTPException(status_code=500, detail="GeoTIFF export failed. See server logs.")

    # ── 12. Quality maps subprocess ───────────────────────────────────────────
    quality_cmd = [
        sys.executable, str(_project_root / "src" / "evaluation" / "quality_maps_4x.py"),
        "--real_input", str(input_npy_path)
    ]
    quality_metrics  = {}
    quality_map_urls = {}
    q_error = None

    try:
        subprocess.run(quality_cmd, capture_output=True, text=True, check=True)
        src_qm_dir  = QUALITY_MAPS_DIR / f"real_{job_id}"
        dest_qm_dir = job_dir / "quality_maps"

        if src_qm_dir.exists():
            shutil.move(str(src_qm_dir), str(dest_qm_dir))
            q_metrics_path = dest_qm_dir / "quality_metrics.json"
            if q_metrics_path.exists():
                with open(q_metrics_path, "r") as f:
                    quality_metrics = json.load(f)
            for png_file in dest_qm_dir.glob("*.png"):
                quality_map_urls[png_file.stem] = f"/outputs/{job_id}/quality_maps/{png_file.name}"
    except subprocess.CalledProcessError as e:
        print("\n[/infer-geotiff Quality Maps Error STDOUT]\n", e.stdout)
        print("\n[/infer-geotiff Quality Maps Error STDERR]\n", e.stderr)
        q_error = "Quality maps generation failed internally."

    # ── 13. Build response (identical schema to /infer) ──────────────────────
    response = {
        "status":            "success",
        "job_id":            job_id,
        "input_shape":       [4, 64, 64],
        "output_shape":      [4, 256, 256],
        "input_gsd_m":       PIXEL_SIZE_NOMINAL,
        "output_gsd_m":      2.5,
        "inference_time_sec": inf_time,
        "reflectance_min":   ref_min,
        "reflectance_max":   ref_max,
        "reflectance_mean":  ref_mean,
        "clipped_uint16_pct": clipped_pct,
        "source_safe":       meta_dict["source_safe"],
        "source_row":        meta_dict["row_off"],
        "source_col":        meta_dict["col_off"],
        "crs":               meta_dict["crs"],
        "patch_note":        meta_dict["extraction_note"],
        "scientific_scope":  (
            "2.5m-scale model-inferred product, not ground-truth 2.5m imagery. "
            "TTA disagreement is a proxy, not calibrated probabilistic uncertainty."
        ),
        "artifacts": {
            "lr_rgb_preview":  f"/outputs/{job_id}/{job_id}_lr_rgb.png",
            "rgb_preview":     f"/outputs/{job_id}/{job_id}_sr_rgb.png",
            "geotiff":         f"/outputs/{job_id}/{job_id}_sr.tif",
            "reflectance_npy": f"/outputs/{job_id}/{job_id}_sr_reflectance.npy",
            "uint16_npy":      f"/outputs/{job_id}/{job_id}_sr_uint16.npy"
        },
        "quality_maps":    quality_map_urls,
        "quality_metrics": quality_metrics
    }

    if q_error:
        response["warnings"] = [q_error]

    return response
