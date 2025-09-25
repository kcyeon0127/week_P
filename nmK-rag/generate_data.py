#!/usr/bin/env python3
"""
박물관 학습 데이터 자동 생성 스크립트

사용법:
    python generate_data.py --api ollama --count 50
    python generate_data.py --api groq --count 30
    python generate_data.py --api huggingface --count 20
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_generator import generate_museum_training_data

def main():
    parser = argparse.ArgumentParser(description="박물관 학습 데이터 자동 생성")

    parser.add_argument(
        "--api",
        choices=["ollama", "groq", "huggingface", "openrouter"],
        default="ollama",
        help="사용할 API 선택 (기본값: ollama)"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="생성할 데이터 개수 (기본값: 30)"
    )

    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="API 설정 상태 확인"
    )

    args = parser.parse_args()

    if args.check_setup:
        check_api_setup()
        return

    print("=" * 60)
    print("🏛️  국립중앙박물관 학습 데이터 생성기")
    print("=" * 60)

    # API별 설정 확인
    if not check_api_available(args.api):
        print(f"❌ {args.api.upper()} API 설정이 필요합니다.")
        show_api_setup_guide(args.api)
        return

    # 데이터 생성 실행
    generate_museum_training_data(
        max_examples=args.count,
        api_preference=args.api
    )

def check_api_available(api_name: str) -> bool:
    """API 사용 가능 여부 확인"""
    if api_name == "ollama":
        return check_ollama()
    elif api_name == "groq":
        return os.getenv("GROQ_API_KEY") is not None
    elif api_name == "huggingface":
        return os.getenv("HF_TOKEN") is not None
    elif api_name == "openrouter":
        return os.getenv("OPENROUTER_API_KEY") is not None
    return False

def check_ollama() -> bool:
    """Ollama 서비스 확인"""
    try:
        import requests
        url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        response = requests.get(f"{url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_api_setup():
    """전체 API 설정 상태 확인"""
    print("🔍 API 설정 상태 확인:")
    print("-" * 40)

    apis = {
        "Ollama": check_ollama(),
        "Groq": os.getenv("GROQ_API_KEY") is not None,
        "HuggingFace": os.getenv("HF_TOKEN") is not None,
        "OpenRouter": os.getenv("OPENROUTER_API_KEY") is not None
    }

    for api_name, is_available in apis.items():
        status = "✅ 사용 가능" if is_available else "❌ 설정 필요"
        print(f"{api_name:12}: {status}")

    print("\n📋 설정 가이드:")
    print("Ollama:     ollama serve && ollama pull llama3.2:3b")
    print("Groq:       export GROQ_API_KEY=your_key")
    print("HF:         export HF_TOKEN=your_token")
    print("OpenRouter: export OPENROUTER_API_KEY=your_key")

def show_api_setup_guide(api_name: str):
    """API별 설정 가이드 출력"""
    guides = {
        "ollama": """
🚀 Ollama 설정 방법:

1. Ollama 설치:
   - macOS: brew install ollama
   - Linux: curl -fsSL https://ollama.ai/install.sh | sh
   - Windows: https://ollama.ai/download

2. Ollama 서비스 시작:
   ollama serve

3. 모델 다운로드:
   ollama pull llama3.2:3b        # 추천 (3B, 빠름)
   ollama pull llama3.1:8b        # 고품질 (8B, 느림)
   ollama pull gemma2:2b          # 가벼움 (2B)

4. 환경변수 설정 (선택사항):
   export OLLAMA_MODEL=llama3.2:3b
   export OLLAMA_URL=http://localhost:11434
""",

        "groq": """
🚀 Groq API 설정 방법:

1. Groq Console 회원가입:
   https://console.groq.com/

2. API Key 발급 (무료):
   - Dashboard > API Keys > Create API Key

3. 환경변수 설정:
   export GROQ_API_KEY=gsk_...

4. 무료 제한:
   - 시간당 14,400 토큰
   - 분당 30 요청
   - llama-3.1-8b-instant 모델 사용 가능
""",

        "huggingface": """
🚀 HuggingFace API 설정 방법:

1. HuggingFace 회원가입:
   https://huggingface.co/

2. Access Token 발급:
   - Settings > Access Tokens > New token

3. 환경변수 설정:
   export HF_TOKEN=hf_...

4. 무료 모델들:
   - microsoft/DialoGPT-medium
   - EleutherAI/gpt-neo-1.3B
   - facebook/blenderbot-400M-distill
""",

        "openrouter": """
🚀 OpenRouter API 설정 방법:

1. OpenRouter 회원가입:
   https://openrouter.ai/

2. API Key 발급:
   - Keys 탭에서 새 키 생성

3. 환경변수 설정:
   export OPENROUTER_API_KEY=sk-or-...

4. 무료 모델들:
   - microsoft/phi-3-mini-128k-instruct:free
   - huggingface/zephyr-7b-beta:free
   - openchat/openchat-7b:free

5. 무료 크레딧: $1 제공 (약 1000번 호출 가능)
"""
    }

    print(guides.get(api_name, "설정 가이드를 찾을 수 없습니다."))

if __name__ == "__main__":
    main()