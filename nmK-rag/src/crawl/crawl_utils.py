"""
크롤링을 위한 공통 유틸리티 클래스들과 헬퍼 함수들
"""

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
from datetime import datetime, date
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tenacity import retry, stop_after_attempt, wait_exponential


class URLTracker:
    """메모리 효율적인 URL 추적을 위한 SQLite 기반 클래스"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        """필요한 테이블들을 초기화합니다."""
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
        """URL이 이미 방문되었는지 확인합니다."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cursor = self.conn.execute("SELECT 1 FROM visited_urls WHERE url_hash = ?", (url_hash,))
        return cursor.fetchone() is not None

    def is_content_duplicate(self, content: str) -> bool:
        """콘텐츠가 중복인지 확인합니다."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cursor = self.conn.execute("SELECT 1 FROM content_hashes WHERE content_hash = ?", (content_hash,))
        return cursor.fetchone() is not None

    def mark_url_visited(self, url: str):
        """URL을 방문 완료로 표시합니다."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        self.conn.execute("INSERT OR IGNORE INTO visited_urls (url_hash, url) VALUES (?, ?)", (url_hash, url))
        self.conn.commit()

    def mark_content_processed(self, content: str, url: str):
        """콘텐츠를 처리 완료로 표시합니다."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        self.conn.execute("INSERT OR IGNORE INTO content_hashes (content_hash, url) VALUES (?, ?)", (content_hash, url))
        self.conn.commit()

    def get_stats(self) -> dict:
        """처리 통계를 반환합니다."""
        url_count = self.conn.execute("SELECT COUNT(*) FROM visited_urls").fetchone()[0]
        content_count = self.conn.execute("SELECT COUNT(*) FROM content_hashes").fetchone()[0]
        return {
            "visited_urls": url_count,
            "unique_contents": content_count
        }

    def close(self):
        """데이터베이스 연결을 닫습니다."""
        self.conn.close()


class CrawlState:
    """크롤링 상태 저장/복원을 위한 클래스"""

    def __init__(self, state_file: str):
        self.state_file = state_file
        os.makedirs(os.path.dirname(state_file), exist_ok=True)

    def save_state(self, queue: list):
        """현재 크롤링 상태를 저장합니다."""
        with open(self.state_file, 'wb') as f:
            pickle.dump({'queue': queue}, f)
        logging.info(f"상태 저장됨: {len(queue)}개 URL이 큐에 남아있음")

    def load_state(self) -> list:
        """이전 크롤링 상태를 복원합니다."""
        try:
            with open(self.state_file, 'rb') as f:
                data = pickle.load(f)
                logging.info(f"이전 상태 복원됨: {len(data['queue'])}개 URL이 큐에 있음")
                return data['queue']
        except FileNotFoundError:
            logging.info("이전 상태 파일이 없음. 새로 시작함")
            return []

    def clear_state(self):
        """상태 파일을 삭제합니다."""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            logging.info("상태 파일 삭제됨")


class CrawlLogger:
    """크롤링 전용 로거 설정 클래스"""

    @staticmethod
    def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
        """크롤링용 로거를 설정합니다."""
        logger = logging.getLogger(name)

        # 기존 핸들러 제거 (중복 방지)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.setLevel(level)

        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger


# ======================================================================================
# 헬퍼 함수들
# ======================================================================================

def check_robots_txt(url: str, user_agent: str) -> bool:
    """robots.txt를 확인하여 크롤링 허용 여부를 판단합니다."""
    try:
        rp = RobotFileParser()
        base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
        rp.set_url(urljoin(base_url, '/robots.txt'))
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        logging.warning(f"robots.txt 확인 실패 {url}: {e}")
        return True  # 확인 실패 시 허용으로 간주


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_with_retry(url: str, headers: dict, timeout: int = 15) -> requests.Response:
    """재시도 로직이 포함된 HTTP 요청"""
    return requests.get(url, headers=headers, timeout=timeout)


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
    # 불필요한 태그 제거
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # 텍스트 추출 및 정리
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


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
    korean_chars = len(re.findall(r'[\u3131-\u314e\u314f-\u3163\uac00-\ud7a3]', text))
    if len(text) > 0 and korean_chars / len(text) < 0.1:
        return False

    return True


def smart_delay(min_seconds: float = 1.0, max_seconds: float = 3.0):
    """지능적인 대기 시간을 적용합니다."""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def get_content_summary(text: str, max_length: int = 100) -> str:
    """콘텐츠의 요약을 생성합니다."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def should_crawl_url(url: str, allowed_patterns: list) -> bool:
    """URL이 크롤링 대상인지 패턴을 기반으로 판단합니다."""
    for pattern in allowed_patterns:
        if pattern in url:
            return True
    return False


# ======================================================================================
# 설정 클래스
# ======================================================================================

class CrawlConfig:
    """크롤링 설정을 관리하는 클래스"""

    def __init__(self,
                 out_dir: str = "data_raw",
                 state_dir: str = "crawl_state",
                 user_agent: str = "Mozilla/5.0 (Research/Student Project; gimchaeyeon-nmk-rag)",
                 min_delay: float = 1.0,
                 max_delay: float = 3.0,
                 timeout: int = 15,
                 max_retries: int = 3,
                 min_content_length: int = 100,
                 save_state_interval: int = 10):

        self.out_dir = out_dir
        self.state_dir = state_dir
        self.headers = {"User-Agent": user_agent}
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_content_length = min_content_length
        self.save_state_interval = save_state_interval

        # 디렉토리 생성
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)

    def get_db_path(self, crawler_name: str) -> str:
        """크롤러별 데이터베이스 경로를 반환합니다."""
        return os.path.join(self.state_dir, f"{crawler_name}_state.db")

    def get_state_file(self, crawler_name: str) -> str:
        """크롤러별 상태 파일 경로를 반환합니다."""
        return os.path.join(self.state_dir, f"{crawler_name}_state.pkl")

    def get_log_file(self, crawler_name: str) -> str:
        """크롤러별 로그 파일 경로를 반환합니다."""
        return f"{crawler_name}.log"


# ======================================================================================
# 통계 수집 클래스
# ======================================================================================

class CrawlStats:
    """크롤링 통계를 수집하고 관리하는 클래스"""

    def __init__(self):
        self.reset()

    def reset(self):
        """통계를 초기화합니다."""
        self.processed_count = 0
        self.skipped_duplicate_content = 0
        self.skipped_duplicate_url = 0
        self.skipped_low_quality = 0
        self.skipped_robots_blocked = 0
        self.failed_requests = 0
        self.start_time = time.time()

    def increment_processed(self):
        self.processed_count += 1

    def increment_skipped_duplicate_content(self):
        self.skipped_duplicate_content += 1

    def increment_skipped_duplicate_url(self):
        self.skipped_duplicate_url += 1

    def increment_skipped_low_quality(self):
        self.skipped_low_quality += 1

    def increment_skipped_robots_blocked(self):
        self.skipped_robots_blocked += 1

    def increment_failed_requests(self):
        self.failed_requests += 1

    def get_summary(self) -> dict:
        """통계 요약을 반환합니다."""
        elapsed_time = time.time() - self.start_time
        total_processed = (self.processed_count + self.skipped_duplicate_content +
                          self.skipped_duplicate_url + self.skipped_low_quality +
                          self.skipped_robots_blocked + self.failed_requests)

        return {
            "processed": self.processed_count,
            "skipped_duplicate_content": self.skipped_duplicate_content,
            "skipped_duplicate_url": self.skipped_duplicate_url,
            "skipped_low_quality": self.skipped_low_quality,
            "skipped_robots_blocked": self.skipped_robots_blocked,
            "failed_requests": self.failed_requests,
            "total_handled": total_processed,
            "elapsed_time_seconds": elapsed_time,
            "pages_per_second": total_processed / elapsed_time if elapsed_time > 0 else 0
        }

    def print_summary(self):
        """통계 요약을 출력합니다."""
        stats = self.get_summary()
        print("\n" + "="*50)
        print("크롤링 통계 요약")
        print("="*50)
        print(f"성공적으로 처리됨: {stats['processed']}")
        print(f"중복 콘텐츠로 스킵: {stats['skipped_duplicate_content']}")
        print(f"중복 URL로 스킵: {stats['skipped_duplicate_url']}")
        print(f"낮은 품질로 스킵: {stats['skipped_low_quality']}")
        print(f"robots.txt 차단: {stats['skipped_robots_blocked']}")
        print(f"요청 실패: {stats['failed_requests']}")
        print(f"총 처리량: {stats['total_handled']}")
        print(f"소요 시간: {stats['elapsed_time_seconds']:.2f}초")
        print(f"처리 속도: {stats['pages_per_second']:.2f}페이지/초")
        print("="*50)


# ======================================================================================
# 전시 상태 판단 유틸리티
# ======================================================================================

def parse_exhibition_date(date_text: str) -> tuple[date | None, date | None]:
    """전시 기간 텍스트를 파싱하여 시작일과 종료일을 반환합니다.

    Args:
        date_text: "2010년 3월 23일 ~ 2010년 7월 25일" 형태의 텍스트

    Returns:
        (시작일, 종료일) 튜플. 파싱 실패시 (None, None)
    """
    if not date_text:
        return None, None

    # 다양한 날짜 형식 패턴들
    patterns = [
        # 2010년 3월 23일 ~ 2010년 7월 25일
        r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*~\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
        # 2010.3.23 ~ 2010.7.25
        r'(\d{4})\.(\d{1,2})\.(\d{1,2})\s*~\s*(\d{4})\.(\d{1,2})\.(\d{1,2})',
        # 2010-03-23 ~ 2010-07-25
        r'(\d{4})-(\d{1,2})-(\d{1,2})\s*~\s*(\d{4})-(\d{1,2})-(\d{1,2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, date_text)
        if match:
            try:
                start_year, start_month, start_day, end_year, end_month, end_day = map(int, match.groups())
                start_date = date(start_year, start_month, start_day)
                end_date = date(end_year, end_month, end_day)
                return start_date, end_date
            except ValueError:
                continue

    # 단일 날짜 패턴 (시작일만 있는 경우)
    single_patterns = [
        r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
        r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
    ]

    for pattern in single_patterns:
        match = re.search(pattern, date_text)
        if match:
            try:
                year, month, day = map(int, match.groups())
                parsed_date = date(year, month, day)
                return parsed_date, None
            except ValueError:
                continue

    return None, None


def get_exhibition_status(start_date: date | None, end_date: date | None, today: date = None) -> str:
    """전시 상태를 판단합니다.

    Args:
        start_date: 전시 시작일
        end_date: 전시 종료일
        today: 기준 날짜 (기본: 오늘)

    Returns:
        'current', 'ended', 'upcoming', 'unknown' 중 하나
    """
    if today is None:
        today = date.today()

    if not start_date:
        return 'unknown'

    if not end_date:
        # 종료일이 없으면 시작일만으로 판단
        if start_date <= today:
            return 'current'
        else:
            return 'upcoming'

    # 시작일과 종료일 모두 있는 경우
    if today < start_date:
        return 'upcoming'
    elif start_date <= today <= end_date:
        return 'current'
    else:
        return 'ended'


def extract_exhibition_info(text: str) -> dict:
    """텍스트에서 전시 관련 정보를 추출합니다.

    Args:
        text: 웹페이지 텍스트

    Returns:
        전시 정보 딕셔너리
    """
    info = {
        'exhibition_period': None,
        'exhibition_status': 'unknown',
        'start_date': None,
        'end_date': None,
        'venue': None,
        'artifact_count': None
    }

    # 전시기간 추출
    period_patterns = [
        r'전시기간[:\s]*([^\n]+)',
        r'기간[:\s]*([^\n]+)',
        r'(\d{4}년[^~]+~[^년]+년[^\n]+)',
    ]

    for pattern in period_patterns:
        match = re.search(pattern, text)
        if match:
            period_text = match.group(1).strip()
            info['exhibition_period'] = period_text

            # 날짜 파싱
            start_date, end_date = parse_exhibition_date(period_text)
            info['start_date'] = start_date.isoformat() if start_date else None
            info['end_date'] = end_date.isoformat() if end_date else None
            info['exhibition_status'] = get_exhibition_status(start_date, end_date)
            break

    # 전시장소 추출
    venue_patterns = [
        r'전시장소[:\s]*([^\n]+)',
        r'장소[:\s]*([^\n]+)',
        r'진행장소[:\s]*([^\n]+)',
    ]

    for pattern in venue_patterns:
        match = re.search(pattern, text)
        if match:
            info['venue'] = match.group(1).strip()
            break

    # 전시유물 수 추출
    artifact_patterns = [
        r'전시유물[:\s]*(\d+)점',
        r'유물[:\s]*(\d+)점',
        r'작품[:\s]*(\d+)점',
    ]

    for pattern in artifact_patterns:
        match = re.search(pattern, text)
        if match:
            info['artifact_count'] = int(match.group(1))
            break

    return info