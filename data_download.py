"""Скачивание данных WWLLN с https://wwlln.net/temp2/ в папку WWLLN рядом со скриптом."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BASE_URL = "https://wwlln.net/temp2/"
USER = os.environ.get("WWLLN_USER", "").strip()
PASS = os.environ.get("WWLLN_PASS", "").strip()

DOWNLOAD_DIR = ROOT / "WWLLN"
BASE_PATH = urlparse(BASE_URL).path  # "/temp2/"

# служебные имена на индексовых страницах — не качаем как данные
SKIP_FILENAMES = {
    "index.html",
    "robots.txt",
    "favicon.ico",
}

session = requests.Session()
session.headers.update({"User-Agent": "wwlln-downloader/1.0"})

visited: set[str] = set()
files_to_download: list[str] = []


def url_to_local_path(url: str) -> Path:
    """https://wwlln.net/temp2/Afiles/foo.gz → WWLLN/Afiles/foo.gz"""
    path = urlparse(url).path
    if path.startswith(BASE_PATH):
        path = path[len(BASE_PATH) :]
    return DOWNLOAD_DIR / path.lstrip("/")


def part_path_for(local_path: Path) -> Path:
    """Временный файл: name.gz → name.gz.part (суффикс .gz не теряется)."""
    return Path(str(local_path) + ".part")


def get_remote_size(url: str) -> int | None:
    """Content-Length через HEAD; None если сервер не отдал."""
    try:
        resp = session.head(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception:
        return None


def is_complete(local_path: Path, expected_size: int | None) -> bool:
    """Готовый файл: есть, не пустой; если известен размер — совпадает."""
    if not local_path.is_file():
        return False
    size = local_path.stat().st_size
    if size <= 0:
        return False
    if expected_size is not None and size != expected_size:
        return False
    return True


def cleanup_part(part_path: Path) -> None:
    if part_path.exists():
        part_path.unlink()


def collect(url: str) -> None:
    if url in visited:
        return
    visited.add(url)
    print(f"Scanning: {url}")
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "text/html" not in ctype.lower():
        return
    soup = BeautifulSoup(resp.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("?") or href in ("/", "../"):
            continue
        full_url = urljoin(url, href)
        if not full_url.startswith(BASE_URL):
            continue
        if full_url.endswith("/"):
            collect(full_url)
            continue
        name = Path(urlparse(full_url).path).name.lower()
        if name in SKIP_FILENAMES:
            continue
        files_to_download.append(full_url)


def download(url: str, index: int, total: int) -> None:
    local_path = url_to_local_path(url)
    part_path = part_path_for(local_path)

    # хвост прошлого обрыва
    cleanup_part(part_path)

    expected = get_remote_size(url)

    if is_complete(local_path, expected):
        print(f"[{index}/{total}] SKIP {local_path}")
        return

    # битый/неполный «финальный» файл
    if local_path.exists():
        print(f"[{index}/{total}] REDOWNLOAD (incomplete) {local_path}")
        local_path.unlink()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{index}/{total}] Downloading {local_path} ...", end=" ", flush=True)

    try:
        with session.get(url, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            cl = resp.headers.get("Content-Length")
            if expected is None and cl is not None:
                expected = int(cl)

            with open(part_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        written = part_path.stat().st_size
        if written <= 0:
            raise IOError("downloaded file is empty")
        if expected is not None and written != expected:
            raise IOError(f"size mismatch: got {written}, expected {expected}")

        part_path.replace(local_path)
        print(f"{written // 1024 // 1024}MB OK")
    except Exception:
        cleanup_part(part_path)
        raise


def main() -> None:
    if not USER or not PASS:
        print(
            "Задайте WWLLN_USER и WWLLN_PASS в файле .env "
            f"(шаблон: {ROOT / '.env.example'})",
            file=sys.stderr,
        )
        sys.exit(1)

    session.auth = (USER, PASS)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Download dir: {DOWNLOAD_DIR}")

    collect(BASE_URL)
    unique = list(dict.fromkeys(files_to_download))
    files_to_download.clear()
    files_to_download.extend(unique)

    total = len(files_to_download)
    print(f"\nTotal files: {total}")
    if total == 0:
        print("Nothing to download.")
        return

    errors = 0
    for i, url in enumerate(files_to_download, 1):
        try:
            download(url, i, total)
        except Exception as e:
            errors += 1
            print(f"ERROR [{i}/{total}] {url}: {e}")

    print(f"Done. errors={errors}")


if __name__ == "__main__":
    main()
