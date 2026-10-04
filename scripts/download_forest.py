import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("CDSE_CLIENT_ID")
client_secret = os.getenv("CDSE_CLIENT_SECRET")

if not client_id or not client_secret:
    raise SystemExit("ERROR: Copernicus credentials missing from .env")

product_id = "c4e6937e-aaf8-4470-b337-5f6193043e8c"
product_name = "S2C_MSIL2A_20260924T050651_N0513_R019_T44QKD_20260924T100421.SAFE"

token_url = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

print("Authenticating with Copernicus...")
token_response = requests.post(
    token_url,
    data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    },
    timeout=30,
)
token_response.raise_for_status()
access_token = token_response.json()["access_token"]

output_dir = Path("data/raw/forest")
output_dir.mkdir(parents=True, exist_ok=True)
zip_path = output_dir / f"{product_name}.zip"

download_url = (
    "https://download.dataspace.copernicus.eu/odata/v1/"
    f"Products({product_id})/$value"
)

print("Starting forest product download...")
with requests.get(
    download_url,
    headers={"Authorization": f"Bearer {access_token}"},
    stream=True,
    timeout=(30, 300),
) as response:
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0))
    downloaded = 0

    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    print(
                        f"\rDownloaded: {percent:.1f}% "
                        f"({downloaded / (1024**2):.1f} MB)",
                        end="",
                        flush=True,
                    )

print("\nDownload completed.")
print("Saved to:", zip_path)
