import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from tqdm import tqdm
from src.schema import Doc, make_id

HEADERS = {"User-Agent": "Mozilla/5.0 (Research/Student Project; gimchaeyeon-nmk-rag)"}
OUT_DIR = "data_raw"


# ======================================================================================
# [설정 영역]
# --------------------------------------------------------------------------------------

# 1. 크롤링 시작점: 상설/특별 전시의 최상위 페이지만 포함
SEED_URLS = [
    "https://www.museum.go.kr/MUSEUM/contents/M0201010000.do",      # 상설전시 층별안내
    "https://www.museum.go.kr/MUSEUM/contents/M0201110000.do",      # 야외전시
    "https://www.museum.go.kr/MUSEUM/contents/M0202010000.do?menuId=current", # 현재 전시
]

# 2. 수집을 허용할 상설 전시관의 ID 목록
ALLOWED_HALL_IDS = ["760", "759", "758", "755", "631120", "757", "756", "406012"]

# 3. 수집할 최근 지난 전시의 개수
PAST_EXHIBITION_LIMIT = 30

# ======================================================================================

def is_same_host(seed: str, link: str) -> bool:
    try:
        a = urlparse(seed).netloc
        b = urlparse(link).netloc
        return (not b) or (a == b)
    except ValueError:
        return False

def normalize_url(base: str, href: str) -> str:
    if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
        return ""
    return urljoin(base, href.strip())

def extract_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text

def discover_past_exhibitions(limit: int) -> list[str]:
    """'지난 전시' 목록을 페이지별로 순회하며 최신 전시 링크를 수집합니다."""
    discovered_urls = []
    base_url = "https://www.museum.go.kr/MUSEUM/contents/M0202030000.do?menuId=past"
    page_num = 1
    print(f"\n'지난 전시' 목록에서 최신 {limit}개를 수집합니다...")

    with tqdm(total=limit, desc="Discovering past exhibitions") as pbar:
        while len(discovered_urls) < limit:
            page_url = f"{base_url}&cpage={page_num}"
            try:
                response = requests.get(page_url, headers=HEADERS, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                items = soup.find_all("div", class_="card")
                if not items:
                    print("\n더 이상 '지난 전시' 항목이 없어 수집을 중단합니다.")
                    break

                for item in items:
                    if len(discovered_urls) >= limit:
                        break
                    
                    link_tag = item.find("a")
                    if link_tag and link_tag.get('href') and not link_tag['href'].startswith("javascript"):
                        full_url = normalize_url(base_url, link_tag['href'])
                        if full_url not in discovered_urls:
                            discovered_urls.append(full_url)
                            pbar.update(1)
                
                page_num += 1
                time.sleep(0.5)

            except requests.RequestException as e:
                print(f"\n'{page_url}' 페이지를 가져오는 중 오류 발생: {e}")
                break
    
    print(f"{len(discovered_urls)}개의 지난 전시 링크를 찾았습니다.")
    return discovered_urls

def crawl():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 지난 전시 링크를 동적으로 수집하여 시드 URL에 추가
    past_exhibition_urls = discover_past_exhibitions(limit=PAST_EXHIBITION_LIMIT)
    all_seed_urls = SEED_URLS + past_exhibition_urls

    queue = all_seed_urls.copy()
    visited = set()

    print(f"\n총 {len(all_seed_urls)}개의 시작 URL로부터 전체 크롤링을 시작합니다.")
    print(f"허용된 상설 전시관 ID: {ALLOWED_HALL_IDS}")

    pbar = tqdm(total=len(queue), desc="Crawling pages")
    while queue:
        url = queue.pop(0)
        if url in visited:
            pbar.update(0)
            continue
        
        visited.add(url)
        pbar.set_description(f"Crawling {url[:80]}")

        try:
            time.sleep(1)
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.get_text().strip() if soup.title else url
            text = extract_text(soup)

            doc = Doc(doc_id=make_id(url), url=url, title=title, text=text, doctype="web", lang="ko")
            with open(os.path.join(OUT_DIR, f"{doc.doc_id}.json"), "w", encoding="utf-8") as f:
                f.write(doc.model_dump_json(ensure_ascii=False, indent=2))

            newly_found = 0
            for a in soup.find_all("a", href=True):
                nxt = normalize_url(url, a["href"])
                
                if not (nxt and is_same_host(all_seed_urls[0], nxt) and nxt not in visited and nxt not in queue):
                    continue

                if "relicId=" in nxt:
                    queue.append(nxt)
                    newly_found += 1
                    continue

                if "showHallId=" in nxt:
                    try:
                        parsed_url = urlparse(nxt)
                        query_params = parse_qs(parsed_url.query)
                        if 'showHallId' in query_params and query_params['showHallId'][0] in ALLOWED_HALL_IDS:
                            queue.append(nxt)
                            newly_found += 1
                            continue
                    except Exception:
                        continue
                
                if "exhiSpThemId=" in nxt:
                    queue.append(nxt)
                    newly_found += 1
                    continue
            
            if newly_found > 0:
                pbar.total += newly_found
                pbar.set_postfix_str(f"Found {newly_found} new links")

        except requests.RequestException as e:
            tqdm.write(f"\n[오류] {url} 페이지를 가져오는 중 오류: {e}")
        except Exception as e:
            tqdm.write(f"\n[오류] {url} 처리 중 예외 발생: {e}")
        
        pbar.update(1)

    pbar.close()
    print(f"\n[완료] 총 {len(visited)}개의 페이지를 크롤링하여 {OUT_DIR} 폴더에 저장했습니다.")

if __name__ == "__main__":
    crawl()
