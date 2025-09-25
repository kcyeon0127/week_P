#!/usr/bin/env python3
"""
학습 데이터 병합 및 관리 스크립트
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

def load_json_data(file_path: str) -> List[Dict]:
    """JSON 파일에서 데이터 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 파일 로드 실패 {file_path}: {e}")
        return []

def save_json_data(data: List[Dict], file_path: str):
    """JSON 파일로 데이터 저장"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 데이터가 {file_path}에 저장되었습니다.")

def merge_training_data(original_file: str, generated_file: str, output_file: str):
    """학습 데이터 병합"""
    print("🔄 학습 데이터 병합 중...")

    # 기존 수동 작성 데이터 로드
    original_data = load_json_data(original_file)
    print(f"📚 기존 데이터: {len(original_data)}개")

    # 자동 생성 데이터 로드
    generated_data = load_json_data(generated_file)
    print(f"🤖 생성 데이터: {len(generated_data)}개")

    # 데이터 형식 통일 (generated_data를 original 형식으로 변환)
    standardized_generated = []
    for item in generated_data:
        standardized_item = {
            "question": item.get("question", ""),
            "general_response": item.get("general_response", ""),
            "children_response": item.get("children_response", ""),
            "metadata": {
                "source": "auto_generated",
                "source_url": item.get("source_url", ""),
                "generated_at": item.get("generated_at", "")
            }
        }
        standardized_generated.append(standardized_item)

    # 기존 데이터에 메타데이터 추가
    standardized_original = []
    for item in original_data:
        if "metadata" not in item:
            item["metadata"] = {"source": "manual"}
        standardized_original.append(item)

    # 데이터 병합
    merged_data = standardized_original + standardized_generated

    # 중복 제거 (질문 기준)
    seen_questions = set()
    unique_data = []
    for item in merged_data:
        question = item.get("question", "").strip()
        if question and question not in seen_questions:
            seen_questions.add(question)
            unique_data.append(item)

    print(f"🎯 병합 완료: {len(unique_data)}개 (중복 제거: {len(merged_data) - len(unique_data)}개)")

    # 데이터 저장
    save_json_data(unique_data, output_file)

    return len(unique_data)

def analyze_training_data(file_path: str):
    """학습 데이터 분석"""
    data = load_json_data(file_path)
    if not data:
        return

    print("📊 학습 데이터 분석:")
    print("-" * 40)
    print(f"총 데이터 개수: {len(data)}개")

    # 소스별 분류
    sources = {}
    for item in data:
        source = item.get("metadata", {}).get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    print("\n📋 소스별 분류:")
    for source, count in sources.items():
        print(f"  {source:15}: {count:3}개")

    # 질문 길이 분석
    question_lengths = [len(item.get("question", "")) for item in data]
    avg_q_length = sum(question_lengths) / len(question_lengths) if question_lengths else 0

    # 응답 길이 분석 (일반)
    general_lengths = [len(item.get("general_response", "")) for item in data]
    avg_g_length = sum(general_lengths) / len(general_lengths) if general_lengths else 0

    # 응답 길이 분석 (어린이)
    children_lengths = [len(item.get("children_response", "")) for item in data]
    avg_c_length = sum(children_lengths) / len(children_lengths) if children_lengths else 0

    print(f"\n📏 평균 길이:")
    print(f"  질문:      {avg_q_length:.1f}자")
    print(f"  일반 응답: {avg_g_length:.1f}자")
    print(f"  어린이 응답: {avg_c_length:.1f}자")

    # 샘플 출력
    if data:
        print(f"\n📝 데이터 샘플:")
        sample = data[0]
        print(f"Q: {sample.get('question', '')}")
        print(f"A(일반): {sample.get('general_response', '')[:100]}...")
        print(f"A(어린이): {sample.get('children_response', '')[:100]}...")

def split_training_data(input_file: str, train_ratio: float = 0.8):
    """학습 데이터를 train/validation으로 분할"""
    data = load_json_data(input_file)
    if not data:
        return

    # 데이터 섞기
    import random
    random.shuffle(data)

    # 분할점 계산
    split_point = int(len(data) * train_ratio)

    # 분할
    train_data = data[:split_point]
    val_data = data[split_point:]

    # 저장
    base_path = Path(input_file).stem
    train_file = f"data/{base_path}_train.json"
    val_file = f"data/{base_path}_val.json"

    save_json_data(train_data, train_file)
    save_json_data(val_data, val_file)

    print(f"📊 데이터 분할 완료:")
    print(f"  훈련용: {len(train_data)}개 → {train_file}")
    print(f"  검증용: {len(val_data)}개 → {val_file}")

def main():
    parser = argparse.ArgumentParser(description="학습 데이터 관리 도구")

    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령")

    # merge 명령
    merge_parser = subparsers.add_parser("merge", help="학습 데이터 병합")
    merge_parser.add_argument("--original", default="data/training_data.json",
                            help="기존 학습 데이터 파일")
    merge_parser.add_argument("--generated", default="data/generated_training_data.json",
                            help="생성된 학습 데이터 파일")
    merge_parser.add_argument("--output", default="data/merged_training_data.json",
                            help="병합 결과 파일")

    # analyze 명령
    analyze_parser = subparsers.add_parser("analyze", help="학습 데이터 분석")
    analyze_parser.add_argument("file", help="분석할 데이터 파일")

    # split 명령
    split_parser = subparsers.add_parser("split", help="학습/검증 데이터 분할")
    split_parser.add_argument("file", help="분할할 데이터 파일")
    split_parser.add_argument("--ratio", type=float, default=0.8,
                            help="훈련 데이터 비율 (기본값: 0.8)")

    args = parser.parse_args()

    if args.command == "merge":
        merge_training_data(args.original, args.generated, args.output)
    elif args.command == "analyze":
        analyze_training_data(args.file)
    elif args.command == "split":
        split_training_data(args.file, args.ratio)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()