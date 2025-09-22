import os
import re
import time
import random
import hashlib
import sqlite3
import pickle
import logging
from pathlib import Path
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from src.schema import Doc, make_id

HEADERS = {"User-Agent": "Mozilla/5.0 (Research/Student Project; gimchaeyeon-nmk-rag)"}
OUT_DIR = "data_raw"
STATE_DIR = "crawl_state"
DB_PATH = os.path.join(STATE_DIR, "crawl_state.db")
STATE_FILE = os.path.join(STATE_DIR, "crawl_state.pkl")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ======================================================================================
# [수정 영역]
# --------------------------------------------------------------------------------------

# 1. 크롤링 시작점: 상설/특별 전시의 최상위 페이지만 포함
SEED_URLS = [
    "https://www.museum.go.kr/MUSEUM/contents/M0201010000.do",      # 상설전시 층별안내
    "https://www.museum.go.kr/MUSEUM/contents/M0201110000.do",      # 야외전시
    "https://www.museum.go.kr/MUSEUM/contents/M0202010000.do?menuId=current", # 현재 전시
    "https://www.museum.go.kr/MUSEUM/contents/M0202020000.do?menuId=upcomming"# '예정 전시'
    # '지난 전시' URL도 여기에 추가할 수 있습니다.
]

# 2. 수집을 허용할 전시관의 ID 목록 (상설 전시에만 해당)
ALLOWED_HALL_IDS = ["760", "759", "758", "755", "631120", "757", "756", "406012"]

# ======================================================================================

class URLTracker:
    """메모리 효율적인 URL 추적을 위한 SQLite 기반 클래스"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS visited_urls (
                url_hash TEXT PRIMARY KEY,
                url TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS content_hashes (
                content_hash TEXT PRIMARY KEY,
                url TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def is_url_visited(self, url: str) -> bool:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cursor = self.conn.execute("SELECT 1 FROM visited_urls WHERE url_hash = ?", (url_hash,))
        return cursor.fetchone() is not None

    def is_content_duplicate(self, content: str) -> bool:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cursor = self.conn.execute("SELECT 1 FROM content_hashes WHERE content_hash = ?", (content_hash,))
        return cursor.fetchone() is not None

    def mark_url_visited(self, url: str):
        url_hash = hashlib.md5(url.encode()).hexdigest()
        self.conn.execute("INSERT OR IGNORE INTO visited_urls (url_hash, url) VALUES (?, ?)", (url_hash, url))
        self.conn.commit()

    def mark_content_processed(self, content: str, url: str):
        content_hash = hashlib.md5(content.encode()).hexdigest()
        self.conn.execute("INSERT OR IGNORE INTO content_hashes (content_hash, url) VALUES (?, ?)", (content_hash, url))
        self.conn.commit()

    def close(self):
        self.conn.close()

class CrawlState:
    """크롤링 상태 저장/복원을 위한 클래스"""
    def __init__(self, state_file: str):
        self.state_file = state_file
        os.makedirs(os.path.dirname(state_file), exist_ok=True)

    def save_state(self, queue: list):
        with open(self.state_file, 'wb') as f:
            pickle.dump({'queue': queue}, f)
        logger.info(f"상태 저장됨: {len(queue)}개 URL이 큐에 남아있음")

    def load_state(self) -> list:
        try:
            with open(self.state_file, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"이전 상태 복원됨: {len(data['queue'])}개 URL이 큐에 있음")
                return data['queue']
        except FileNotFoundError:
            logger.info("이전 상태 파일이 없음. 새로 시작함")
            return []

    def clear_state(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            logger.info("상태 파일 삭제됨")

def check_robots_txt(url: str) -> bool:
    """robots.txt를 확인하여 크롤링 허용 여부를 판단합니다."""
    try:
        rp = RobotFileParser()
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        rp.set_url(urljoin(base_url, '/robots.txt'))
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception as e:
        logger.warning(f"robots.txt 확인 실패 {url}: {e}")
        return True  # 확인 실패 시 허용으로 간주

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_with_retry(url: str) -> requests.Response:
    """재시도 로직이 포함된 HTTP 요청"""
    return requests.get(url, headers=HEADERS, timeout=15)

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

def validate_content(text: str, title: str) -> bool:
    """콘텐츠 품질을 검증합니다."""
    # 최소 길이 체크
    if len(text.strip()) < 100:
        return False

    # 의미있는 단어 비율 체크
    words = text.split()
    meaningful_words = [w for w in words if len(w) > 2]
    if len(meaningful_words) < 10:
        return False

    # 한국어 텍스트 비율 체크
    korean_chars = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', text))
    if len(text) > 0 and korean_chars / len(text) < 0.1:
        return False

    return True


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

    # URL 추적 및 상태 관리 초기화
    url_tracker = URLTracker(DB_PATH)
    crawl_state = CrawlState(STATE_FILE)

    # 이전 상태 복원 또는 새로 시작
    queue = crawl_state.load_state()
    if not queue:
        queue = SEED_URLS.copy()
        logger.info(f"새로운 크롤링 시작: {len(SEED_URLS)}개의 시드 URL")
    else:
        logger.info(f"이전 크롤링 재개: {len(queue)}개의 URL이 큐에 있음")

    logger.info(f"허용된 상설 전시관 ID: {ALLOWED_HALL_IDS}")

    processed_count = 0
    pbar = tqdm(total=len(queue), desc="Crawling pages")

    try:
        while queue:
            url = queue.pop(0)

            # URL 방문 여부 확인
            if url_tracker.is_url_visited(url):
                pbar.update(0)
                continue

            pbar.set_description(f"Crawling {url[:60]}...")

            # robots.txt 확인
            if not check_robots_txt(url):
                logger.warning(f"robots.txt에 의해 차단됨: {url}")
                url_tracker.mark_url_visited(url)
                pbar.update(1)
                continue

            try:
                # 랜덤 대기 시간
                time.sleep(random.uniform(1, 3))

                # 재시도 로직이 포함된 요청
                r = fetch_with_retry(url)
                r.raise_for_status()

                soup = BeautifulSoup(r.text, "html.parser")
                title = soup.title.get_text().strip() if soup.title else url
                text = extract_text(soup)

                # 콘텐츠 품질 검증
                if not validate_content(text, title):
                    logger.warning(f"낮은 품질의 콘텐츠로 스킵: {url}")
                    url_tracker.mark_url_visited(url)
                    pbar.update(1)
                    continue

                # 중복 콘텐츠 확인
                if url_tracker.is_content_duplicate(text):
                    logger.info(f"중복 콘텐츠로 스킵: {url}")
                    url_tracker.mark_url_visited(url)
                    pbar.update(1)
                    continue

                # 문서 저장
                doc = Doc(doc_id=make_id(url), url=url, title=title, text=text, doctype="web", lang="ko")
                with open(os.path.join(OUT_DIR, f"{doc.doc_id}.json"), "w", encoding="utf-8") as f:
                    f.write(doc.model_dump_json(ensure_ascii=False, indent=2))

                # URL 및 콘텐츠 처리 완료 표시
                url_tracker.mark_url_visited(url)
                url_tracker.mark_content_processed(text, url)
                processed_count += 1

                logger.info(f"문서 저장됨: {title[:50]}... ({url})")

                # 현재 페이지에서 허용된 링크만 추출하여 큐에 추가
                newly_found = 0
                for a in soup.find_all("a", href=True):
                    nxt = normalize_url(url, a["href"])

                    if not (nxt and is_same_host(SEED_URLS[0], nxt) and
                           not url_tracker.is_url_visited(nxt) and nxt not in queue):
                        continue

                    # 조건 1: 상설 전시의 개별 전시품인가? (relicId)
                    if "relicId=" in nxt:
                        queue.append(nxt)
                        newly_found += 1
                        continue

                    # 조건 2: 허용된 상설 전시관 페이지인가? (showHallId)
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

                    # 조건 3: 특별 전시 상세페이지인가? (exhiSpThemId)
                    if "exhiSpThemId=" in nxt:
                        queue.append(nxt)
                        newly_found += 1
                        continue

                if newly_found > 0:
                    pbar.total += newly_found
                    pbar.set_postfix_str(f"Found {newly_found} new links, Processed: {processed_count}")
                    logger.info(f"새로운 링크 {newly_found}개 발견됨")

                # 주기적으로 상태 저장 (매 10개 페이지마다)
                if processed_count % 10 == 0:
                    crawl_state.save_state(queue)

            except requests.RequestException as e:
                logger.error(f"HTTP 요청 실패 {url}: {e}")
                # 실패한 URL도 방문 처리하여 무한 재시도 방지
                url_tracker.mark_url_visited(url)
            except Exception as e:
                logger.error(f"처리 중 예외 발생 {url}: {e}")
                url_tracker.mark_url_visited(url)

            pbar.update(1)

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨. 상태를 저장합니다...")
        crawl_state.save_state(queue)
        raise

    finally:
        pbar.close()
        url_tracker.close()

        # 완료 시 상태 파일 정리
        if not queue:
            crawl_state.clear_state()

        logger.info(f"크롤링 완료: 총 {processed_count}개의 페이지를 처리하여 {OUT_DIR} 폴더에 저장했습니다.")
        print(f"\n[완료] 총 {processed_count}개의 페이지를 크롤링하여 {OUT_DIR} 폴더에 저장했습니다.")

if __name__ == "__main__":
    try:
        crawl()
    except KeyboardInterrupt:
        logger.info("크롤링이 중단되었습니다. 다음 실행 시 이어서 진행됩니다.")
        print("\n크롤링이 중단되었습니다. 다음 실행 시 이어서 진행됩니다.")
    except Exception as e:
        logger.error(f"크롤링 중 오류 발생: {e}")
        print(f"\n크롤링 중 오류 발생: {e}")