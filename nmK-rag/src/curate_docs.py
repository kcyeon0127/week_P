import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.schema import Doc

NAV_PREFIX = "국립중앙박물관>"

NOISE_LINES = {
    "국립중앙박물관>소장품>소장품 검색",
    "국립중앙박물관>전시>상설 전시>조각·공예관>도자공예-분청사기-백자실",
    "국립중앙박물관>전시>특별 전시>지난 전시",
    "관람 정보",
    "조각·공예관",
    "불교조각",
    "기증관",
    "기증2",
    "확대보기",
    "QR코드",
    "현재 페이지의 QR코드 입니다.",
    "주소복사",
    "스크랩",
    "인쇄",
    "공유",
    "X",
    "페이스북",
    "Home",
    "전시해설",
    "수어",
    "동영상 바로가기",
    "특별 전시",
    "현재 전시",
    "예정 전시",
    "특별전",
    "소장품",
    "소장품 검색",
    "전시",
    "상설 전시",
    "지난 전시",
    "테마전",
    "* 이미지에 마우스를 올려 확대해 보세요.",
    "이미지 1",
    "이미지 보기",
    "이전 이미지 보기",
    "다음 이미지 보기",
    "다음",
    "관심유물로 등록",
    "소장품 바로가기",
    "목록",
    "예약내역 확인",
    "페이지",
    "회차명",
    "진행일시",
    "신청단체수",
    "신청하기",
    "대기단체수",
    "주중(월~금) 프로그램 신청 안내",
    "TOP",
    "3D보기",
    "총",
}

NOISE_CONTAINS = (
    "현재 [",
    "현재 페이지의",
    "이미지 보기",
    "QR코드",
    "sns 공유",
    "공유하기",
    "슬라이드",
    "이전 이미지",
    "다음 이미지",
    "이미지입니다",
    "내려받기",
    "확대보기",
    "누리집",
    "전화 문의",
    "문의 :",
    "문의:",
    "회차정보를",
    "검색되었습니다",
)

BULLET_PREFIXES = ("ㅇ", "-", "•", "*", "ㆍ")

FIELD_LABELS = (
    "중요",
    "다른명칭",
    "전시명칭",
    "전시명",
    "전시명칭/주제",
    "주제/내용",
    "주제",
    "전시기간",
    "전시장소",
    "진행장소",
    "전시내용",
    "전시구성",
    "전시유물",
    "알아두기",
    "전시유물 소개",
    "전시소개",
    "국적/시대",
    "재질",
    "작가",
    "분류",
    "크기",
    "소장품번호",
    "전시위치",
    "운영기간",
    "관람정보",
    "대상",
    "담당부서",
    "담당자",
    "참가방법",
    "참가비",
    "예약내역",
    "관련자료",
)

LICENSE_PREFIX = "국립중앙박물관이(가)"
LICENSE_KEYWORDS = (
    "공공누리",
    "공공저작물 자유이용허락",
    "저작물은 공공누리",
    "조건에 따라 이용할 수 있습니다",
)


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _is_navigation(line: str) -> bool:
    if line.startswith(NAV_PREFIX) and line.count(">") >= 2:
        return True
    if "상설 전시" in line and ">" in line:
        return True
    return False


def _strip_bullet(line: str) -> Tuple[bool, str]:
    for prefix in BULLET_PREFIXES:
        if line.startswith(prefix):
            stripped = line[len(prefix) :].lstrip(" \t-")
            return True, stripped
    return False, line


def clean_doc_text(text: str) -> Tuple[str, Optional[str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\ufeff", "")

    cleaned: List[str] = []
    license_lines: List[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)

        for postfix in [" 내려받기"]:
            if line.endswith(postfix):
                line = line[: -len(postfix)].rstrip()

        if line == "/":
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", line):
            continue
        if re.match(r"^총\s*\d+\s*건", line):
            continue

        if _is_navigation(line):
            continue
        if line in NOISE_LINES:
            continue
        if any(token in line for token in NOISE_CONTAINS):
            continue
        if line.startswith(LICENSE_PREFIX) or any(key in line for key in LICENSE_KEYWORDS):
            license_lines.append(line)
            continue

        cleaned.append(line)

    normalized: List[str] = []
    pending_label: Optional[str] = None
    value_buffer: List[str] = []

    def flush_pending() -> None:
        nonlocal pending_label, value_buffer
        if pending_label is None:
            value_buffer.clear()
            return
        if value_buffer:
            if any(v.startswith("- ") for v in value_buffer):
                merged = "\n".join(value_buffer)
            else:
                merged = " ".join(value_buffer)
            normalized.append(f"{pending_label}: {merged}")
        else:
            normalized.append(pending_label)
        pending_label = None
        value_buffer = []

    for line in cleaned:
        bullet, bullet_value = _strip_bullet(line)
        if bullet:
            if pending_label is None:
                normalized.append(f"- {bullet_value}")
            else:
                value_buffer.append(f"- {bullet_value}")
            continue

        if pending_label and any(
            line.startswith(f"{lbl}:") for lbl in FIELD_LABELS
        ):
            flush_pending()
            normalized.append(line)
            continue

        label = next(
            (lbl for lbl in FIELD_LABELS if line == lbl or line.rstrip(":") == lbl),
            None,
        )
        if label and ":" not in line:
            flush_pending()
            pending_label = label
            continue

        if pending_label:
            value_buffer.append(line)
            continue

        normalized.append(line)

    flush_pending()
    normalized = _dedupe_keep_order(normalized)

    cleaned_text = "\n".join(normalized).strip()
    license_note = "\n".join(_dedupe_keep_order(license_lines)).strip() or None
    return cleaned_text, license_note


def clean_doc(doc: Doc) -> Doc:
    cleaned_text, license_note = clean_doc_text(doc.text)
    update = {"text": cleaned_text}
    if license_note:
        update["license_note"] = license_note
    return doc.model_copy(update=update)


def curate_folder(input_dir: Path, output_dir: Path) -> int:
    count = 0
    for src in sorted(input_dir.rglob("*.json")):
        data = json.loads(src.read_text(encoding="utf-8"))
        doc = Doc(**data)
        doc = clean_doc(doc)
        if not doc.text.strip():
            continue
        dest = output_dir / src.relative_to(input_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        count += 1
    return count


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "data_raw"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data_curated"


def main() -> None:
    parser = argparse.ArgumentParser(description="크롤링 문서를 정제해 data_curated에 저장합니다.")
    parser.add_argument("input_dir", nargs="?", default=str(DEFAULT_INPUT_DIR), help="원본 JSON 디렉터리 (예: data_raw)")
    parser.add_argument("output_dir", nargs="?", default=str(DEFAULT_OUTPUT_DIR), help="정제본을 저장할 디렉터리 (예: data_curated)")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = curate_folder(in_dir, out_dir)
    print(f"[done] wrote {total} curated docs to {out_dir}")


if __name__ == "__main__":
    main()
