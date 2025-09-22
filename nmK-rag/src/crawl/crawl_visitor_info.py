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
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.schema import Doc, make_id

HEADERS = {"User-Agent": "Mozilla/5.0 (Research/Student Project; gimchaeyeon-nmk-rag)"}

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '../..')
OUT_DIR = os.path.join(PROJECT_ROOT, "data_raw")
STATE_DIR = os.path.join(PROJECT_ROOT, "crawl_state")
DB_PATH = os.path.join(STATE_DIR, "crawl_visitor_info_state.db")
STATE_FILE = os.path.join(STATE_DIR, "crawl_visitor_info_state.pkl")
LOG_FILE = os.path.join(PROJECT_ROOT, "crawl_visitor_info.log")

# 로깅 설정 (디버그 모드 활성화)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ======================================================================================
# [수정 영역]
# --------------------------------------------------------------------------------------

# 1. 크롤링 시작점: 이용 안내 관련 페이지 포함
SEED_URLS = [
    "https://www.museum.go.kr/MUSEUM/contents/M0101000000.do",                 # 관람 안내
    "https://www.museum.go.kr/MUSEUM/contents/M0105000000.do?menuId=accessibility", # 접근성 안내
    "https://www.museum.go.kr/MUSEUM/contents/M0106010000.do?menuId=subway-map",      # 지하철
    "https://www.museum.go.kr/MUSEUM/contents/M0106010000.do?menuId=bus-map",         # 버스
    "https://www.museum.go.kr/MUSEUM/contents/M0106030000.do?menuId=car-map",         # 자기차량
    "https://www.museum.go.kr/MUSEUM/contents/M0106040000.do?menuId=parking-map"      # 주차안내
]

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

def validate_content(text: str, title: str, is_seed: bool = False) -> bool:
    """콘텐츠 품질을 검증합니다. 시드 URL은 더 관대한 기준을 적용합니다."""
    # 최소 길이 체크 (시드 URL은 더 관대하게)
    min_len = 50 if is_seed else 100
    if len(text.strip()) < min_len:
        return False

    # 의미있는 단어 비율 체크 (시드 URL은 더 관대하게)
    words = text.split()
    meaningful_words = [w for w in words if len(w) > 2]
    min_words = 5 if is_seed else 10
    if len(meaningful_words) < min_words:
        return False

    # 한국어 텍스트 비율 체크 (시드 URL은 더 관대하게)
    korean_chars = len(re.findall(r'[\u3131-\u314e\u314f-\u3163\uac00-\ud7a3]', text))
    min_korean_ratio = 0.05 if is_seed else 0.1
    if len(text) > 0 and korean_chars / len(text) < min_korean_ratio:
        return False

    return True

def extract_text(soup: BeautifulSoup) -> str:
    """HTML에서 불필요한 태그를 제거하고 텍스트를 추출합니다."""
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text

def crawl(force: bool = False, interactive: bool = False):
    """시작 페이지에서 출발하여, 관련된 이용 안내 페이지만을 크롤링합니다."""
    # 경로 확인 및 디렉토리 생성
    logger.info(f"출력 디렉토리: {os.path.abspath(OUT_DIR)}")
    logger.info(f"상태 디렉토리: {os.path.abspath(STATE_DIR)}")
    logger.info(f"로그 파일: {os.path.abspath(LOG_FILE)}")
    if force:
        logger.info("강제 모드: 방문 여부 무시")
    if interactive:
        logger.info("대화형 모드: 방문한 페이지 발견 시 사용자에게 문의")

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

    processed_count = 0
    pbar = tqdm(total=len(queue), desc="Crawling visitor info pages")

    try:
        while queue:
            url = queue.pop(0)

            # URL 방문 여부 확인 (강제 모드는 제외)
            if not force and url_tracker.is_url_visited(url):
                if interactive:
                    # 대화형 모드: 사용자에게 재크롤링 여부 문의
                    print(f"\n이미 방문한 페이지를 발견했습니다:")
                    print(f"URL: {url}")
                    print(f"제목: {url.split('/')[-1] if '/' in url else url}")

                    while True:
                        choice = input("다시 크롤링하시겠습니까? [y]es/[n]o/[a]ll(모든 중복 재크롤링)/[s]kip all(모든 중복 스킵): ").lower().strip()

                        if choice in ['y', 'yes']:
                            logger.info(f"[USER] 재크롤링 선택: {url}")
                            break  # 크롤링 계속 진행
                        elif choice in ['n', 'no']:
                            logger.debug(f"[USER] 스킵 선택: {url}")
                            pbar.update(1)
                            break
                        elif choice in ['a', 'all']:
                            logger.info("[USER] 모든 중복 재크롤링 선택")
                            force = True  # 이후 모든 중복에 대해 강제 모드 적용
                            break
                        elif choice in ['s', 'skip', 'skip all']:
                            logger.info("[USER] 모든 중복 스킵 선택")
                            interactive = False  # 이후 모든 중복을 자동으로 스킵
                            pbar.update(1)
                            break
                        else:
                            print("올바른 선택지를 입력해주세요: y/n/a/s")

                    if choice in ['n', 'no'] or not interactive:
                        continue
                else:
                    logger.debug(f"[SKIP] 이미 방문한 URL: {url}")
                    pbar.update(1)  # 진행바 버그 수정
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

                # 콘텐츠 품질 검증 (시드 URL은 더 관대하게)
                is_seed = url in SEED_URLS
                if not validate_content(text, title, is_seed=is_seed):
                    quality_msg = f"낮은 품질의 콘텐츠로 스킵 {'(시드 URL)' if is_seed else ''}: {url}"
                    logger.warning(quality_msg)
                    logger.debug(f"[QUALITY] 제목: {title[:30]}..., 텍스트 길이: {len(text)}")
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
                doc = Doc(doc_id=make_id(url), url=url, title=title, text=text, doctype="web-visitor-info", lang="ko")
                with open(os.path.join(OUT_DIR, f"{doc.doc_id}.json"), "w", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps(doc.model_dump(), ensure_ascii=False, indent=2))

                # URL 및 콘텐츠 처리 완료 표시
                url_tracker.mark_url_visited(url)
                url_tracker.mark_content_processed(text, url)
                processed_count += 1

                logger.info(f"문서 저장됨: {title[:50]}... ({url})")

                # 현재 페이지에서 관련 링크만 추출하여 큐에 추가
                newly_found = 0
                for a in soup.find_all("a", href=True):
                    nxt = normalize_url(url, a["href"])

                    if not (nxt and is_same_host(SEED_URLS[0], nxt) and
                           not url_tracker.is_url_visited(nxt) and nxt not in queue):
                        continue

                    # 조건: 이용 안내 관련 페이지인가? (URL 경로에 /M0105 또는 /M0106 포함)
                    if "/M0101" in nxt or "/M0105" in nxt or "/M0106" in nxt:
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


def show_menu():
    """대화형 메뉴를 표시하고 사용자 선택을 받습니다."""
    print("\n" + "="*60)
    print("🏯  국립중앙박물관 이용안내 크롤러")
    print("="*60)
    print("1. 일반 실행 (이어서 진행)")
    print("2. 완전히 새로 시작 (기존 데이터 삭제 후 시작)")
    print("3. 모든 페이지 강제 재크롤링")
    print("4. 대화형 모드 (중복 발견 시 선택)")
    print("5. 새로 시작 + 대화형 모드")
    print("6. 종료")
    print("="*60)

    while True:
        try:
            choice = input("선택하세요 (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return int(choice)
            else:
                print("⚠️  올바른 번호를 입력해주세요 (1-6)")
        except (ValueError, KeyboardInterrupt):
            print("\n종료됩니다.")
            return 6


if __name__ == "__main__":
    try:
        choice = show_menu()

        if choice == 6:
            print("프로그램을 종료합니다.")
            exit(0)

        # 선택에 따른 매개변수 설정
        fresh = choice in [2, 5]  # 새로 시작 또는 새로 시작 + 대화형
        force = choice == 3       # 강제 재크롤링
        interactive = choice in [4, 5]  # 대화형 모드 또는 새로 시작 + 대화형

        if fresh:
            print("\n🔄 기존 데이터를 삭제하고 새로 시작합니다...")
            # DB/상태파일 제거
            os.makedirs(STATE_DIR, exist_ok=True)
            try:
                os.remove(DB_PATH)
                logger.info("Fresh 모드: 이전 방문 기록 DB 삭제됨")
            except FileNotFoundError:
                pass
            try:
                os.remove(STATE_FILE)
                logger.info("Fresh 모드: 이전 상태 파일 삭제됨")
            except FileNotFoundError:
                pass
            logger.info("Fresh 모드: 상태 초기화 완료. 완전히 새로운 크롤링을 시작합니다.")
        elif force:
            print("\n⚡ 강제 재크롤링 모드로 실행합니다...")
        elif interactive:
            print("\n💬 대화형 모드로 실행합니다...")
        else:
            print("\n▶️  일반 모드로 실행합니다...")

        crawl(force=force, interactive=interactive)

    except KeyboardInterrupt:
        logger.info("크롤링이 중단되었습니다. 다음 실행 시 이어서 진행됩니다.")
        print("\n크롤링이 중단되었습니다. 다음 실행 시 이어서 진행됩니다.")
    except Exception as e:
        logger.error(f"크롤링 중 오류 발생: {e}")
        print(f"\n크롤링 중 오류 발생: {e}")
