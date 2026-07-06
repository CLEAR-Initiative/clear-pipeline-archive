"""Download WorldPop population GeoTIFF for a country and upload to S3.

Usage:
    python scripts/upload_population_tiff.py                  # SDN (default)
    python scripts/upload_population_tiff.py --iso3 AFG       # Afghanistan
    python scripts/upload_population_tiff.py --iso3 VEN       # Venezuela

Requires S3 env vars: S3_ENDPOINT, S3_BUCKET, S3_REGION, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY
"""

import argparse
import os
import sys
import tempfile

import boto3
import httpx
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file


def build_paths(iso3: str) -> tuple[str, str]:
    """Build the WorldPop source URL and S3 key for a given country.

    WorldPop convention: the URL path uses uppercase ISO3 while the file name
    uses lowercase. We mirror the WorldPop filename in the S3 key so the raster
    is self-describing on disk.
    """
    iso3_upper = iso3.upper()
    iso3_lower = iso3.lower()
    file_name = f"{iso3_lower}_pop_2026_CN_100m_R2025A_v1.tif"
    url = (
        "https://data.worldpop.org/GIS/Population/Global_2015_2030/"
        f"R2025A/2026/{iso3_upper}/v1/100m/constrained/{file_name}"
    )
    s3_key = f"population/{file_name}"
    return url, s3_key


def main():
    parser = argparse.ArgumentParser(
        description="Download a WorldPop GeoTIFF and stage it in S3 for the population service.",
    )
    parser.add_argument(
        "--iso3",
        default="SDN",
        help="Country ISO3 code (default: SDN). Examples: SDN, AFG, VEN.",
    )
    args = parser.parse_args()

    url, s3_key = build_paths(args.iso3)

    endpoint = os.environ.get("S3_ENDPOINT")
    bucket = os.environ.get("S3_BUCKET")
    region = os.environ.get("S3_REGION", "auto")
    access_key = os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY")

    if not all([endpoint, bucket, access_key, secret_key]):
        print("Missing S3 env vars. Set S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY")
        sys.exit(1)

    # Download from WorldPop
    print(f"Downloading from {url} ...")
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
        with httpx.stream("GET", url, timeout=300, follow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            for chunk in resp.iter_bytes(chunk_size=8192):
                tmp.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print(f"\nDownloaded to {tmp_path} ({os.path.getsize(tmp_path) / 1e6:.1f} MB)")

    # Upload to S3
    print(f"Uploading to s3://{bucket}/{s3_key} ...")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    s3.upload_file(tmp_path, bucket, s3_key)
    print("Upload complete!")

    # Cleanup
    os.unlink(tmp_path)
    print("Done.")


if __name__ == "__main__":
    main()
