"""
Bulk-download all PDF papers from CVF open-access website for a given CVPR year.

Example
-------
$ python downloader.py 2023      # 下载 CVPR2023
$ 下载 CVPR 202X, ~/Downloads/CVPR202X:
python downloader.py 2024 -j 4 -o ~/hpc-share/revolutionizing-higher-ed-rankings/Downloads/CVPR2024_619
"""
import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from urllib.parse import urljoin

# BASE = "https://openaccess.thecvf.com/CVPR2020?day=2020-06-18"
BASE = "https://openaccess.thecvf.com/ICCV2023?day=2023-10-06"

PDF_TEXT = "pdf"
RETRY = 3
SLEEP_BETWEEN = 0.3

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Referer": BASE})


def crawl_pdf_links(conf: str, year: int) -> List[str]:
    url = f"{BASE}/{conf}{year}?day=all"
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    anchors = soup.find_all("a", string=PDF_TEXT)
    links = [urljoin(BASE + "/", a["href"]) for a in anchors]
    return links


def safe_filename(path: str) -> str:
    """把 CVF 给出的长文件名清洗成安全的本地文件名."""
    name = os.path.basename(path)
    # 去掉 CVPR_2023_paper.pdf 这类尾缀，让路径更简洁
    # name = re.sub(r"_CVPR_\d{4}_paper\.pdf$", ".pdf", name, flags=re.I)
    name = re.sub(r"_ICCV_\d{4}_paper\.pdf$", ".pdf", name, flags=re.I)
    # 防止 Windows 中 <>:"/\|?* 等无效字符
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def need_download(remote_url: str, local_path: Path) -> bool:
    
    if not local_path.exists():
        return True
    # HEAD 查询线上大小
    try:
        r = SESSION.head(remote_url, timeout=10, allow_redirects=True)
        remote_size = int(r.headers.get("Content-Length", -1))
        return remote_size != local_path.stat().st_size
    except Exception:
        return True


def fetch_one(remote_url: str, dest_dir: Path) -> None:
    filename = safe_filename(remote_url)
    out_path = dest_dir / filename
    if not need_download(remote_url, out_path):
        return

    for attempt in range(1, RETRY + 1):
        try:
            with SESSION.get(remote_url, stream=True, timeout=60) as r:
                if r.headers.get("Content-Type", "").split(";")[0] != "application/pdf":
                    raise RuntimeError(
                        f"Unexpected content type: {r.headers.get('Content-Type')}"
                    )
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            return          # 下载成功
        except Exception as e:
            if attempt == RETRY:
                print(f"[FAIL] {filename}: {e}")
            else:
                time.sleep(2 * attempt)


def main():
    parser = argparse.ArgumentParser(description="Download all CVF PDFs for a given year.")
    parser.add_argument("year", type=int, help="CVPR year, e.g. 2023")
    parser.add_argument("-c", "--conf", default="CVPR", help="Conference acronym (default CVPR)")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel download threads")
    parser.add_argument("-o", "--out", default=None, help="Output directory")
    args = parser.parse_args()

    dest_root = Path(args.out or f"{args.conf}{args.year}_PDFs").expanduser()
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"🔎  Crawling {args.conf}{args.year} paper list …")
    pdf_links = crawl_pdf_links(args.conf, args.year)
    print(f"Found {len(pdf_links)} PDFs, downloading to “{dest_root}”")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(fetch_one, url, dest_root) for url in pdf_links]
        for _ in tqdm(as_completed(tasks), total=len(tasks), unit="file"):
            pass
        pool.shutdown(wait=True)

    print("=========== All done ==============")


if __name__ == "__main__":
    main()
