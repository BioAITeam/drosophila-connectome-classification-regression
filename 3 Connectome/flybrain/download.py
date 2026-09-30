"""Download original MaleCNS v1.0 exports without overwriting raw files."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import uuid

from .paths import RAW, REPO, TEMP, allowed_write, portable_path
BASE_URL = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
SOURCE_FILES = {
    'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'neurotransmitters': 'body-neurotransmitters-male-cns-v1.0.feather',
    'weights': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
}


def ensure_ha(path: str | Path) -> Path:
    """Compatibility alias for the configured repository write boundary."""
    return allowed_write(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _header_metadata(headers) -> dict:
    hashes = {}
    for value in headers.get_all('x-goog-hash', []):
        for item in value.split(','):
            key, _, encoded = item.strip().partition('=')
            hashes[key] = encoded
    return {
        'content_length': int(headers['Content-Length']) if headers.get('Content-Length') else None,
        'server_md5_base64': hashes.get('md5'),
        'etag': headers.get('ETag'),
        'generation': headers.get('x-goog-generation'),
        'last_modified': headers.get('Last-Modified'),
    }


def download_sources(raw_root: str | Path, manifest_dir: str | Path,
                     temp_dir: str | Path) -> dict:
    """Fetch official files, verify recorded and transport hashes, preserve originals.

    Raw files live below ``raw_root/malecns-v1.0``. A preexisting file is hashed
    and compared with its recorded SHA-256; it is never replaced. Without a prior
    manifest, its size and MD5 must match the official object before adoption.
    New downloads must also match a recorded SHA-256 when one is available.
    """
    raw_dir = ensure_ha(raw_root) / 'malecns-v1.0'
    manifest_dir, temp_dir = ensure_ha(manifest_dir), ensure_ha(temp_dir)
    for directory in (raw_dir, manifest_dir, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / 'malecns-v1.0-sources.json'
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = prior.get('files', {})
    manifest = {
        'dataset': 'MaleCNS', 'version': 'v1.0', 'synapse_confidence_min': 0.5,
        'official_catalog': 'https://male-cns.janelia.org/download/',
        'license': 'CC-BY-4.0', 'files': records,
    }
    for role, filename in SOURCE_FILES.items():
        destination = ensure_ha(raw_dir / filename)
        url = BASE_URL + filename
        if destination.exists() and role in records:
            actual = sha256_file(destination)
            if actual != records[role]['sha256']:
                raise RuntimeError(f'Raw integrity mismatch; refusing overwrite: {destination}')
            print(f'Verified existing {filename}: {actual}', flush=True)
            continue
        if destination.exists():
            request = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(request, timeout=90) as response:
                remote = _header_metadata(response.headers)
            md5 = hashlib.md5(usedforsecurity=False)
            with destination.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    md5.update(chunk)
            if (not remote['server_md5_base64'] or
                    base64.b64encode(md5.digest()).decode() != remote['server_md5_base64'] or
                    destination.stat().st_size != remote['content_length']):
                raise RuntimeError(f'Existing raw does not match official object: {destination}')
            digest_hex = sha256_file(destination)
        else:
            stage = temp_dir / f'{filename}.{uuid.uuid4().hex}.part'
            digest, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
            print(f'Downloading {url}', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'malecns-learning/0.1'})
            received = 0
            with urllib.request.urlopen(request, timeout=120) as response, stage.open('xb') as stream:
                remote = _header_metadata(response.headers)
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
                    stream.write(chunk)
                    digest.update(chunk)
                    md5.update(chunk)
                    received += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if remote['content_length'] is not None and received != remote['content_length']:
                raise RuntimeError(f'Incomplete download retained for inspection: {stage}')
            if remote['server_md5_base64'] and (
                base64.b64encode(md5.digest()).decode() != remote['server_md5_base64']
            ):
                raise RuntimeError(f'Official MD5 mismatch: {stage}')
            if not remote['server_md5_base64']:
                raise RuntimeError(f'Official object did not provide integrity MD5: {stage}')
            digest_hex = digest.hexdigest()
            if role in records and digest_hex != records[role]['sha256']:
                raise RuntimeError(f'Download differs from the recorded source SHA-256: {stage}')
            # Hard-link publication is atomic and refuses to replace an existing raw file.
            stage.chmod(0o444)
            os.link(stage, destination)
            stage.unlink()
        records[role] = {
            'path': portable_path(destination), 'filename': filename, 'url': url,
            'sha256': digest_hex, 'bytes': destination.stat().st_size,
            'verified_utc': datetime.now(timezone.utc).isoformat(), **remote,
        }
        manifest_stage = manifest_dir / f'.sources-{uuid.uuid4().hex}.json'
        manifest_stage.write_text(json.dumps(manifest, indent=2) + '\n')
        os.replace(manifest_stage, manifest_path)
        print(f'Installed immutable raw {filename}: {digest_hex}', flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=RAW)
    parser.add_argument('--manifest-dir', type=Path, default=REPO / '2 Data/2 Manifests')
    parser.add_argument('--temp-dir', type=Path, default=TEMP / 'downloads')
    args = parser.parse_args()
    download_sources(args.raw_root, args.manifest_dir, args.temp_dir)


if __name__ == '__main__':
    main()
