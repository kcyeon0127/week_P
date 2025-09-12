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
# [수정 영역]
# --------------------------------------------------------------------------------------

# 1. 크롤링 시작점: 상설 전시 전체 개요, 추천 동선 등 최상위 페이지만 포함
SEED_URLS = [
    "https://www.museum.go.kr/MUSEUM/contents/M0201010000.do",  # 상설전시 층별안내
    "https://www.museum.go.kr/MUSEUM/contents/M0201110000.do",  # 야외전시
]

# 2. 수집을 허용할 전시관의 ID 목록
#    - 760: 선사·고대관
#    - 759: 중·근세관
#    - 758: 서화관
#    - 755: 기증관
#    - 631120: 사유의 방
#    - 757: 조각·공예관
#    - 756: 세계문화관
#    - 406012: 역사의 길
ALLOWED_HALL_IDS = ["760", "759", "758", "755", "631120", "757", "756", "406012"]

# ======================================================================================

def is_same_host(seed: str, link: str) -> bool:
    """링크가 시드 URL과 동일한 호스트에 속하는지 확인합니다."""
    try:
        a = urlparse(seed).netloc
        b = urlparse(link).netloc
        return (not b) or (a == b)
    except ValueError:
        return False

def normalize_url(base: str, href: str) -> str:
    """상대 URL을 절대 URL로 변환합니다."""
    if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
        return ""
    return urljoin(base, href.strip())


def extract_text(soup: BeautifulSoup) -> str:
    """HTML에서 불필요한 태그를 제거하고 텍스트를 추출합니다."""
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text

def crawl():
    """시작 페이지에서 출발하여, 허용된 전시관 및 그 안의 전시품 페이지만을 지능적으로 크롤링합니다."""
    os.makedirs(OUT_DIR, exist_ok=True)

    queue = SEED_URLS.copy()
    visited = set()

    print(f"총 {len(SEED_URLS)}개의 시작 URL로부터 크롤링을 시작합니다.")
    print(f"허용된 전시관 ID: {ALLOWED_HALL_IDS}")

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

            # 현재 페이지에서 허용된 링크만 추출하여 큐에 추가
            newly_found = 0
            for a in soup.find_all("a", href=True):
                nxt = normalize_url(url, a["href"])
                
                if not (nxt and is_same_host(SEED_URLS[0], nxt) and nxt not in visited and nxt not in queue):
                    continue

                # 조건 1: 개별 전시품 상세페이지인가? (relicId)
                if "relicId=" in nxt:
                    queue.append(nxt)
                    newly_found += 1
                    continue

                # 조건 2: 우리가 허용한 전시관 페이지인가? (showHallId)
                if "showHallId=" in nxt:
                    try:
                        parsed_url = urlparse(nxt)
                        query_params = parse_qs(parsed_url.query)
                        if 'showHallId' in query_params and query_params['showHallId'][0] in ALLOWED_HALL_IDS:
                            queue.append(nxt)
                            newly_found += 1
                    except Exception:
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
