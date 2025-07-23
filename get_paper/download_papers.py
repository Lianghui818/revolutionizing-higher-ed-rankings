# import csv
# import os
# import re
# import sys
# import time
# import urllib.request

# # pip install requests
# import requests
# # pip install beautifulsoup4
# # an html5 reader
# from bs4 import BeautifulSoup

# # Downloads all publications from all conferences within the conferences folder.
# # * Each publication is downloaded into a folder with the name of the conference
# #   CSV it is a from, without "_papers.csv"
# # NOTE: Directory management only tested on a Windows machine
# def download_all():
#     conferences = os.listdir(os.getcwd() + "/Conferences")
#     for file_name in conferences:
#        download_conference(file_name)
#     return

# # Downloads all publications from a specific conference
# # input: the CSV file name (include extension)
# def download_conference(file_name):
#     with open(os.getcwd() + "/Conferences/" + file_name, encoding="utf8") as conf_csv:
#         url_reader = csv.reader(conf_csv, delimiter=',', quotechar='\"')
#         row = next(url_reader)
#         while True:
#             try:
#                 row = next(url_reader)
#                 url = row[4]
#                 paper = download_paper(url, file_name)
#                 if not paper == -4:
#                     time.sleep(1)
#             except StopIteration:
#                 break
#     return
    

# # Downloads a paper from the url parameter.
# # Assumes that the link provided is not a direct link to the pdf, and that a
# # direct link to the PDF is present on the page.
# # arxiv.org and doi.org not supported.
# def download_paper(url, conference):
#     if not ".html" in url:
#         print("Invalid URL")
#         return -4
#     success = 0
#     domains = ["http://papers.nips.cc", "http://proceedings.mlr.press",
#                "https://proceedings.mlr.press", "https://proceedings.neurips.cc",
#                "https://openaccess.thecvf.com", "https://doi.org", "arxiv.org"]
#     if domains[5] in url or domains[6] in url:
#                return # Cannot handle DOI or Arxiv links
#     r = requests.get(url)
#     if r.status_code == 404:
#         print("Error 404: Link unavailable or otherwise inaccessible")
#         return -1
#     html = urllib.request.urlopen(url).read().decode("utf8")
#     soup = BeautifulSoup(html, "html.parser")
#     links = soup.find_all('a')
#     conference_folder = "Publications/" + conference[:-11] + "/" # Remove "_papers.csv" suffix
#     download_link = ""
#     name = ""
#     #print(url)
#     # Check each link for a valid PDF download
#     for link in links:
#         if domains[0] in url:   # papers.nips.cc and proceedings.neurips.cc require extra care
#             if link.string == "Paper":
#                 download_link = domains[0] + "/" + link.get("href")
#                 name = soup.find_all("h4")[0].string
                
#         elif domains[1] in url or domains[2] in url: # proceedings.mlr.press
#             if link.string == "Download PDF":
#                 download_link = link.get("href")
#                 name = soup.find_all("h1")[0].string

#         elif domains[3] in url:
#             if link.string == "Paper":
#                 download_link = domains[3] + "/" + link.get("href")
#                 name = soup.find_all("h4")[0].string

#         elif domains[4] in url: # openaccess.thecvf.com
#             if link.string == "pdf":
#                 if not "html/w" in url:
#                     download_link = domains[4] + link.get("href")[5:] # Remove ../.. beginning
#                 else:    
#                     download_link = domains[4] + link.get("href")[8:] # Remove ../../.. beginning
#                 if not soup.find(id="papertitle").string: # Fixes non-subscriptable error for now
#                     print("Issue with scanning for papertitle")
#                     return -2
#                 if(soup.find(id="papertitle").string[0] == '\n'):
#                     name = soup.find(id="papertitle").string[1:]
#                 else:
#                     name = soup.find(id="papertitle").string

#     # Set the name of the PDF to the name of the publication
#     # Certain characters replaced or removed for a clean filename
#     x = re.search(r"^[a-zA-Z0-9_\-\(\)]*$", name)
#     if not x:
#         name = re.sub(r'\\', '', name)
#         name = re.sub(r'[^a-zA-Z0-9\-\.\:\(\)]', '_', name)
#         name = re.sub(r'[:.]','', name)
#         first = re.search(r'^[a-zA-Z0-9]*$', name[0])
#         #Repeatedly remove first character if it ends up not being alphanumeric (quick fix)
#         while(not first):
#             name = name[1:]
#             first = re.search(r'^[a-zA-Z0-9]*$', name[0])
#     print(name)
#     local_path = conference_folder + name + ".pdf"
#     print(download_link)
#     # Do not redownload if a paper already exists
#     if os.path.exists(local_path):
#         print("Paper already downloaded")
#     else:
#         stored_loc = urllib.request.urlretrieve(download_link, local_path)
#         if(stored_loc):
#             print("Success")
#         else:
#             print("Could not download")

# # Program must be run in the following manner:
# # python download_papers.py conference_name.csv
# # - include .csv extension
# # - error will be thrown if argument is missing or
# #   a valid name is not provided.
# def main():
#     print("Extracting papers from " + sys.argv[1])
#     publication_path = os.getcwd() + "/Publications"
#     conferences_path = os.listdir(os.getcwd() + "/Conferences")
                    
#     if not os.path.exists(publication_path):
#         os.makedirs(publication_path)
    
#     for file_name in conferences_path:
#         publication_path = os.getcwd() + "/Publications/" + file_name[:-11] # Remove "_papers.csv" suffix
#         if not os.path.exists(publication_path):
#             os.makedirs(publication_path)
                             
#     download_conference(sys.argv[1])
#     #download_all()

# if __name__ == "__main__":
#     main()


"""
Bulk-download all PDF papers from multiple ML conference websites.

Supported conferences:
- CVF (CVPR, ICCV, WACV): https://openaccess.thecvf.com/
- MLR Press (ICML, NeurIPS, UAI, CoRL, etc.): https://proceedings.mlr.press/
- NeurIPS archive: https://papers.nips.cc/

Example
-------
$ python downloader.py CVPR 2023              # 下载 CVPR2023
$ python downloader.py ICML 2023              # 下载 ICML2023  
$ python downloader.py NeurIPS 2023           # 下载 NeurIPS2023
$ python downloader.py ICCV 2023 -j 8 -o ~/Downloads/ICCV2023
"""
import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Callable
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from urllib.parse import urljoin, urlparse

RETRY = 3
SLEEP_BETWEEN = 0.3

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# 会议配置映射
CONFERENCE_CONFIG = {
    # CVF conferences
    "CVPR": {
        "site": "cvf",
        "base_url": "https://openaccess.thecvf.com",
        "url_pattern": "https://openaccess.thecvf.com/CVPR{year}?day=all"
    },
    "ICCV": {
        "site": "cvf", 
        "base_url": "https://openaccess.thecvf.com",
        "url_pattern": "https://openaccess.thecvf.com/ICCV{year}?day=all"
    },
    "WACV": {
        "site": "cvf",
        "base_url": "https://openaccess.thecvf.com", 
        "url_pattern": "https://openaccess.thecvf.com/WACV{year}?day=all"
    },
    
    # MLR Press conferences - 需要手动维护年份到卷号的映射
    "ICML": {
        "site": "mlr",
        "base_url": "https://proceedings.mlr.press",
        "volume_mapping": {
            2023: "v202", 2022: "v162", 2021: "v139", 2020: "v119",
            2019: "v97", 2018: "v80", 2017: "v70", 2016: "v48"
        }
    },
    "NeurIPS": {
        "site": "mlr", 
        "base_url": "https://proceedings.mlr.press",
        "volume_mapping": {
            2024: "v262", 2023: "v226", 2022: "v210", 2021: "v176", 2020: "v133"
        }
    },
    "UAI": {
        "site": "mlr",
        "base_url": "https://proceedings.mlr.press", 
        "volume_mapping": {
            2023: "v216", 2022: "v180", 2021: "v161", 2020: "v124"
        }
    },
    "CoRL": {
        "site": "mlr",
        "base_url": "https://proceedings.mlr.press",
        "volume_mapping": {
            2023: "v229", 2022: "v205", 2021: "v164", 2020: "v155"
        }
    },
    
    # NeurIPS 原始网站 (备用)
    "NIPS": {
        "site": "nips",
        "base_url": "https://papers.nips.cc",
        "url_pattern": "https://papers.nips.cc/paper/{year}"
    }
}


def crawl_cvf_papers(conf: str, year: int) -> List[str]:
    """从CVF网站爬取论文链接"""
    config = CONFERENCE_CONFIG[conf]
    url = config["url_pattern"].format(year=year)
    
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # 查找所有包含"pdf"文本的链接
    anchors = soup.find_all("a", string="pdf")
    links = [urljoin(config["base_url"] + "/", a["href"]) for a in anchors]
    return links


def crawl_mlr_papers(conf: str, year: int) -> List[str]:
    """从MLR Press网站爬取论文链接"""
    config = CONFERENCE_CONFIG[conf]
    
    if year not in config["volume_mapping"]:
        raise ValueError(f"Year {year} not supported for {conf}. Available years: {list(config['volume_mapping'].keys())}")
    
    volume = config["volume_mapping"][year]
    url = f"{config['base_url']}/{volume}/"
    
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # MLR Press的PDF链接通常在class="paper"的div中
    links = []
    
    # 方法1: 查找直接的PDF链接
    pdf_links = soup.find_all("a", href=re.compile(r"\.pdf$"))
    for link in pdf_links:
        href = link["href"]
        if not href.startswith("http"):
            href = urljoin(url, href)
        links.append(href)
    
    # 方法2: 查找包含"Download PDF"文本的链接
    pdf_text_links = soup.find_all("a", string=re.compile(r"PDF|pdf", re.I))
    for link in pdf_text_links:
        href = link["href"] 
        if href.endswith(".pdf"):
            if not href.startswith("http"):
                href = urljoin(url, href)
            links.append(href)
    
    return list(set(links))  # 去重


def crawl_nips_papers(conf: str, year: int) -> List[str]:
    """从NeurIPS原始网站爬取论文链接"""
    config = CONFERENCE_CONFIG[conf]
    url = config["url_pattern"].format(year=year)
    
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # 查找PDF链接
    links = []
    pdf_links = soup.find_all("a", href=re.compile(r"\.pdf$"))
    for link in pdf_links:
        href = link["href"]
        if not href.startswith("http"):
            href = urljoin(url, href)
        links.append(href)
    
    return links


def safe_filename(path: str, conf: str) -> str:
    """清洗文件名，根据不同会议采用不同策略"""
    name = os.path.basename(path)
    
    # CVF会议的文件名清理
    if conf in ["CVPR", "ICCV", "WACV"]:
        name = re.sub(rf"_{conf}_\d{{4}}_paper\.pdf$", ".pdf", name, flags=re.I)
    
    # MLR Press的文件名通常比较简洁，不需要特殊处理
    
    # 防止Windows中的无效字符
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    
    # 确保文件名不过长
    if len(name) > 200:
        base, ext = os.path.splitext(name)
        name = base[:190] + ext
    
    return name


def need_download(remote_url: str, local_path: Path) -> bool:
    """检查是否需要下载文件"""
    if not local_path.exists():
        return True
    
    try:
        r = SESSION.head(remote_url, timeout=10, allow_redirects=True)
        remote_size = int(r.headers.get("Content-Length", -1))
        return remote_size != local_path.stat().st_size
    except Exception:
        return True


def fetch_one(remote_url: str, dest_dir: Path, conf: str) -> None:
    """下载单个PDF文件"""
    filename = safe_filename(remote_url, conf)
    out_path = dest_dir / filename
    
    if not need_download(remote_url, out_path):
        return
    
    for attempt in range(1, RETRY + 1):
        try:
            with SESSION.get(remote_url, stream=True, timeout=60) as r:
                # 检查内容类型
                content_type = r.headers.get("Content-Type", "").split(";")[0]
                if content_type and content_type != "application/pdf":
                    print(f"[WARN] {filename}: Unexpected content type: {content_type}")
                
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            return  # 下载成功
        except Exception as e:
            if attempt == RETRY:
                print(f"[FAIL] {filename}: {e}")
            else:
                time.sleep(2 * attempt)


def crawl_pdf_links(conf: str, year: int) -> List[str]:
    """根据会议类型选择合适的爬虫函数"""
    if conf not in CONFERENCE_CONFIG:
        raise ValueError(f"Unsupported conference: {conf}. Supported: {list(CONFERENCE_CONFIG.keys())}")
    
    config = CONFERENCE_CONFIG[conf]
    site_type = config["site"]
    
    if site_type == "cvf":
        return crawl_cvf_papers(conf, year)
    elif site_type == "mlr":
        return crawl_mlr_papers(conf, year)
    elif site_type == "nips":
        return crawl_nips_papers(conf, year)
    else:
        raise ValueError(f"Unknown site type: {site_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Download all PDF papers from multiple ML conference websites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported conferences:
{chr(10).join(f"  {conf}: {config['site']}" for conf, config in CONFERENCE_CONFIG.items())}

Examples:
  python {os.path.basename(__file__)} CVPR 2023
  python {os.path.basename(__file__)} ICML 2023 -j 8
  python {os.path.basename(__file__)} NeurIPS 2023 -o ~/Downloads/
        """
    )
    
    parser.add_argument("conference", choices=list(CONFERENCE_CONFIG.keys()),
                       help="Conference name")
    parser.add_argument("year", type=int, help="Conference year")
    parser.add_argument("-j", "--jobs", type=int, default=4, 
                       help="Parallel download threads (default: 4)")
    parser.add_argument("-o", "--out", default=None, 
                       help="Output directory (default: {CONF}{YEAR}_PDFs)")
    
    args = parser.parse_args()
    
    # 设置输出目录
    dest_root = Path(args.out or f"{args.conference}{args.year}_PDFs").expanduser()
    dest_root.mkdir(parents=True, exist_ok=True)
    
    print(f"🔎  Crawling {args.conference} {args.year} paper list...")
    
    try:
        pdf_links = crawl_pdf_links(args.conference, args.year)
        print(f"Found {len(pdf_links)} PDFs, downloading to \"{dest_root}\"")
        
        if not pdf_links:
            print("⚠️  No PDFs found. Please check the conference and year.")
            return
        
        # 并行下载
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            tasks = [pool.submit(fetch_one, url, dest_root, args.conference) 
                    for url in pdf_links]
            
            for _ in tqdm(as_completed(tasks), total=len(tasks), unit="file", 
                         desc="Downloading"):
                pass
            
            pool.shutdown(wait=True)
        
        print("============== All done! =============")
        
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!! Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main() or 0)