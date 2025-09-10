import os, re, json, time, argparse
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from src.schema import Doc, make_id

HEADERS = {"User-Agent": "Mozilla/5.0 (Research/Student Project)"}

def is_same_host(seed: str, link: str) -> bool:
    try:
        a = urlparse(seed).netloc
        b = urlparse(link).netloc
        return (not b) or (a == b)
    except:
        return False

def normalize_url(base: str, href: str) -> str:
    if not href:
        return ""
    href = href.strip()
    if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
        return ""
    return urljoin(base, href)

def extract_text(soup: BeautifulSoup) -> str:
    # Remove nav/footer/scripts
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text

def crawl(seed_url: str, out_dir: str, limit: int = 200):
    os.makedirs(out_dir, exist_ok=True)
    visited = set()
    queue = [seed_url]
    saved = 0

    while queue and saved < limit:
        url = queue.pop(0)
        if url in visited: 
            continue
        visited.add(url)

        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"[skip] {url} -> {e}")
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text().strip() if soup.title else url
        text = extract_text(soup)

        # 저장
        doc = Doc(
            doc_id=make_id(url),
            url=url,
            title=title,
            text=text,
            doctype="web",
            lang="ko"
        )
        with open(os.path.join(out_dir, f"{doc.doc_id}.json"), "w", encoding="utf-8") as f:
            f.write(doc.model_dump_json(ensure_ascii=False))
        saved += 1

        # 링크 확장
        for a in soup.find_all("a", href=True):
            nxt = normalize_url(url, a["href"])
            if nxt and is_same_host(seed_url, nxt) and nxt not in visited and nxt not in queue:
                queue.append(nxt)

    print(f"[done] saved {saved} docs to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="seed url (e.g., https://www.museum.go.kr)")
    ap.add_argument("--out", default="data_raw/web", help="output dir for raw docs")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    crawl(args.base, args.out, args.limit)
