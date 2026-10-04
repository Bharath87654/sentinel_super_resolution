import sys
import json
import requests
from pathlib import Path

# Setup paths (script expects to be run from project root)
PROJECT_ROOT = Path(__file__).resolve().parent
INFERENCE_DIR = PROJECT_ROOT / "outputs" / "inference_4x"
NPY_PATH = INFERENCE_DIR / "real_urban_r1024_c5120.npy"
META_PATH = INFERENCE_DIR / "real_urban_r1024_c5120_metadata.json"

API_URL = "http://127.0.0.1:8000"


def run_tests():
    print("=== Testing /health ===")
    try:
        health_resp = requests.get(f"{API_URL}/health")
        print(health_resp.json())
    except requests.exceptions.ConnectionError:
        print(f"FAILED to connect to API at {API_URL}. Is uvicorn running?")
        sys.exit(1)

    print(f"\n=== Preparing Data from {INFERENCE_DIR.name} ===")
    if not NPY_PATH.exists() or not META_PATH.exists():
        print("ERROR: Real input patch or metadata JSON is missing.")
        print(f"Checked: {NPY_PATH}")
        print(f"Checked: {META_PATH}")
        sys.exit(1)

    print(f"Found genuine input patch: {NPY_PATH.name}")
    print(f"Found matching metadata: {META_PATH.name}")

    # Read binary bytes directly from the genuine NPY and JSON files on disk
    with open(NPY_PATH, "rb") as f:
        npy_bytes = f.read()

    with open(META_PATH, "rb") as f:
        meta_bytes = f.read()

    files = {
        "file": (NPY_PATH.name, npy_bytes, "application/octet-stream"),
        "metadata_file": (META_PATH.name, meta_bytes, "application/json")
    }

    print("\n=== Testing /infer ===")
    infer_resp = requests.post(f"{API_URL}/infer", files=files)

    try:
        resp_json = infer_resp.json()
    except Exception as e:
        print("Failed to parse JSON response. Raw text:")
        print(infer_resp.text)
        sys.exit(1)

    print(f"Status: {resp_json.get('status')}")
    if resp_json.get("status") != "success":
        print(f"Error Detail: {json.dumps(resp_json, indent=2)}")
        sys.exit(1)

    print(f"Job ID: {resp_json.get('job_id')}")
    print(f"Artifacts: {json.dumps(resp_json.get('artifacts'), indent=2)}")
    print(f"Quality Maps: {json.dumps(resp_json.get('quality_maps'), indent=2)}")

    print("\n=== Testing Static File Routes ===")

    # 1. Test RGB preview
    rgb_url = resp_json.get("artifacts", {}).get("rgb_preview")
    if rgb_url:
        full_rgb_url = f"{API_URL}{rgb_url}"
        rgb_resp = requests.get(full_rgb_url)
        print(f"Retrieving: {full_rgb_url}")
        print(f" -> Status code: {rgb_resp.status_code}")
        if rgb_resp.status_code == 200:
            print(f" -> Success! Fetched {len(rgb_resp.content)} bytes.")
        else:
            print(" -> Failed to retrieve RGB preview.")

    # 2. Test one quality map
    qmaps = resp_json.get("quality_maps", {})
    if qmaps:
        # Grab the first available quality map URL dynamically
        first_map_key = list(qmaps.keys())[0]
        qmap_url = qmaps[first_map_key]
        full_qmap_url = f"{API_URL}{qmap_url}"
        qmap_resp = requests.get(full_qmap_url)
        print(f"\nRetrieving: {full_qmap_url}")
        print(f" -> Status code: {qmap_resp.status_code}")
        if qmap_resp.status_code == 200:
            print(f" -> Success! Fetched {len(qmap_resp.content)} bytes.")
        else:
            print(" -> Failed to retrieve quality map.")
    else:
        print("\nWARNING: No quality maps were returned in the API response.")


if __name__ == "__main__":
    run_tests()
