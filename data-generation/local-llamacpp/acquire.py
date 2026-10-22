"""Acquire verified, pinned public model/runtime assets into this task only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import threading
import time
import urllib.request
import zipfile

OUT = Path(__file__).resolve().parent
# Downloads and the unpacked llama.cpp runtime. Default: <repo>/work/local-generation.
WORK = Path(os.environ.get("TABCREW_LLAMACPP_WORK") or OUT.parents[1] / "work" / "local-generation")
REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
TAG = "b10809"
ASSETS = [
    {"name": "Qwen3-8B-Q4_K_M.gguf", "purpose": "Local 8B text teacher, 4-bit Q4_K_M",
     "url": f"https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/{REVISION}/Qwen3-8B-Q4_K_M.gguf",
     "bytes": 5027783488, "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
     "published_at": "2025-05-21T06:31:50Z"},
    {"name": "llama-b10809-bin-win-cuda-12.4-x64.zip", "purpose": "Portable CUDA inference runtime",
     "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/llama-b10809-bin-win-cuda-12.4-x64.zip",
     "bytes": 253938543, "sha256": "c77bfcd9ed8d91e8721a2d6a290b907fddd4fa5412a47b21c6fa1709116b85f9",
     "published_at": "2026-09-04T17:55:12Z"},
    {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip", "purpose": "Runtime-local CUDA libraries; no system CUDA installation",
     "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/cudart-llama-bin-win-cuda-12.4-x64.zip",
     "bytes": 391443627, "sha256": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
     "published_at": "2026-09-04T17:55:12Z"},
]
LOCK = threading.Lock()
PROGRESS = {}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def update(name, **fields):
    with LOCK:
        PROGRESS[name] = {**PROGRESS.get(name, {}), **fields}
        atomic(OUT / "acquisition-status.json", {"updated_at": utc(), "assets": PROGRESS})


def download(asset):
    target = WORK / "downloads" / asset["name"]
    if target.exists():
        if sha(target) != asset["sha256"]:
            raise ValueError("Existing asset hash mismatch: " + str(target))
        update(asset["name"], state="verified", received=asset["bytes"], total=asset["bytes"])
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "TabCrew-local-setup"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urllib.request.urlopen(urllib.request.Request(asset["url"], headers=headers), timeout=60) as response:
        if offset and response.status != 206:
            offset = 0
        if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Unexpected range response")
        received, last = offset, 0.0
        with temporary.open("ab" if offset else "wb") as file:
            while chunk := response.read(4 * 1024 * 1024):
                file.write(chunk)
                received += len(chunk)
                if time.monotonic() - last > 2:
                    update(asset["name"], state="downloading", received=received, total=asset["bytes"])
                    last = time.monotonic()
    update(asset["name"], state="verifying", received=received, total=asset["bytes"])
    if temporary.stat().st_size != asset["bytes"] or sha(temporary) != asset["sha256"]:
        raise ValueError("Downloaded asset failed size/hash check: " + asset["name"])
    os.replace(temporary, target)
    update(asset["name"], state="verified", received=received, total=asset["bytes"])
    return target


def main():
    (WORK / "downloads").mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
    for asset in ASSETS:
        if datetime.fromisoformat(asset["published_at"].replace("Z", "+00:00")) > cutoff:
            raise ValueError("Asset is newer than the user's 72-hour minimum")
    if shutil.disk_usage(WORK).free < 15 * 1024**3:
        raise ValueError("Insufficient free disk for verified download and extraction")
    atomic(OUT / "asset-plan.json", {"created_at": utc(), "model": "Qwen/Qwen3-8B-GGUF", "revision": REVISION,
        "runtime_release": "v0.4.0", "runtime_build": TAG, "assets": ASSETS,
        "sources": ["https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b10809",
                    "https://huggingface.co/api/models/Qwen/Qwen3-8B-GGUF?blobs=true"]})
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(download, ASSETS))
    runtime = WORK / "runtime" / TAG
    runtime.mkdir(parents=True, exist_ok=True)
    extracted = {}
    for path in paths[1:]:
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                destination = (runtime / entry.filename).resolve()
                if not destination.is_relative_to(runtime.resolve()) or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Unsafe ZIP entry")
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                data = archive.read(entry)
                if destination.exists() and destination.read_bytes() != data:
                    raise ValueError("Runtime archive collision: " + str(destination))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                extracted[str(destination.relative_to(runtime))] = hashlib.sha256(data).hexdigest()
    executables = list(runtime.rglob("llama-server.exe"))
    if len(executables) != 1:
        raise ValueError("Cannot uniquely locate the portable server")
    for name, url in {"MODEL-LICENSE.txt": f"https://huggingface.co/Qwen/Qwen3-8B-GGUF/raw/{REVISION}/LICENSE",
                      "RUNTIME-LICENSE.txt": f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{TAG}/LICENSE",
                      "SERVER-README.md": f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{TAG}/tools/server/README.md"}.items():
        data = urllib.request.urlopen(url, timeout=30).read()
        (OUT / name).write_bytes(data)
    license_text = (OUT / "MODEL-LICENSE.txt").read_text()
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("Model license differs from reviewed Apache 2.0 identity")
    atomic(OUT / "assets.json", {"completed_at": utc(), "model": "Qwen/Qwen3-8B-GGUF", "revision": REVISION,
        "model_path": str(paths[0]), "model_sha256": ASSETS[0]["sha256"],
        "model_license": "Apache-2.0", "model_license_sha256": sha(OUT / "MODEL-LICENSE.txt"),
        "runtime": str(executables[0]), "runtime_tag": TAG, "runtime_release": "v0.4.0",
        "runtime_license": "MIT", "runtime_file_sha256": extracted})
    print(json.dumps(read_json(OUT / "assets.json")), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
