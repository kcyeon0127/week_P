#!/usr/bin/env python3
"""
서버 환경용 학습 데이터 생성 스크립트 (아나콘다 가상환경)
"""

import os
import time
import json
import subprocess
import requests
from pathlib import Path
from src.data_generator import generate_museum_training_data, MuseumDataGenerator

def check_server_resources():
    """서버 리소스 체크"""
    print("🖥️  서버 리소스 체크:")

    # 메모리 체크
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) // 1024  # MB
        mem_available = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) // 1024  # MB

        print(f"   💾 총 메모리: {mem_total//1024:.1f}GB")
        print(f"   💾 사용 가능: {mem_available//1024:.1f}GB")
    except:
        mem_total = 8192  # 기본값 8GB
        print(f"   💾 메모리: 감지 실패 (기본값 8GB 가정)")

    # CPU 체크
    try:
        cpu_count = os.cpu_count()
        print(f"   🚀 CPU 코어: {cpu_count}개")
    except:
        cpu_count = 4
        print(f"   🚀 CPU: 감지 실패 (기본값 4코어 가정)")

    # GPU 체크 (선택사항)
    has_gpu = False
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        has_gpu = result.returncode == 0
        if has_gpu:
            print("   🎮 GPU: ✅ NVIDIA GPU 감지됨")
        else:
            print("   🎮 GPU: ❌ 감지되지 않음")
    except:
        print("   🎮 GPU: ❌ 감지되지 않음")

    return mem_total, cpu_count, has_gpu

def optimize_settings(mem_mb, cpu_count, has_gpu):
    """리소스에 맞는 최적 설정 (GPU 서버 최적화)"""
    mem_gb = mem_mb // 1024

    print("\n⚙️ 최적화 설정:")

    # GPU 서버 특별 최적화
    if has_gpu and mem_gb >= 32:
        model = "llama3.1:8b"  # 고성능 모델
        batch_size = 20        # 대량 배치
        parallel = min(6, cpu_count // 2)
        print("   🎮 GPU 서버 고성능 모드 활성화!")
    elif has_gpu and mem_gb >= 16:
        model = "llama3.2:3b"
        batch_size = 15
        parallel = min(4, cpu_count // 2)
        print("   🎮 GPU 가속 모드 활성화!")
    elif mem_gb >= 12:
        model = "llama3.2:3b"
        batch_size = 10
        parallel = min(2, cpu_count // 4)
    else:
        model = "gemma2:2b"
        batch_size = 5
        parallel = 1

    print(f"   🤖 모델: {model}")
    print(f"   📦 배치 크기: {batch_size}")
    print(f"   🔄 병렬 처리: {parallel}")

    # 환경변수 설정
    os.environ['OLLAMA_MODEL'] = model
    os.environ['OLLAMA_NUM_PARALLEL'] = str(parallel)
    os.environ['OLLAMA_URL'] = 'http://localhost:11434'
    os.environ['OLLAMA_GPU_LAYERS'] = '40'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    return model, batch_size, parallel

def check_ollama_connection():
    """Ollama 서비스 연결 확인"""
    print("\n🔗 Ollama 연결 체크:")

    url = os.getenv("OLLAMA_URL", "http://ollama:11434")
    max_retries = 10

    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"   ✅ Ollama 연결 성공!")
                print(f"   📚 설치된 모델: {len(models)}개")
                for model in models:
                    print(f"      - {model.get('name', 'Unknown')}")
                return True
            else:
                print(f"   ⚠️ Ollama 응답 오류: {response.status_code}")
        except Exception as e:
            print(f"   ❌ 연결 시도 {i+1}/{max_retries} 실패: {e}")
            if i < max_retries - 1:
                print(f"   ⏳ 5초 후 재시도...")
                time.sleep(5)

    print("   ❌ Ollama 연결 실패!")
    print("   💡 해결 방법:")
    print("      1. Ollama 서비스가 실행 중인지 확인")
    print("      2. docker-compose up -d ollama")
    print("      3. docker-compose logs ollama")

    return False

def ensure_model_downloaded(model_name):
    """모델 다운로드 확인 및 설치"""
    print(f"\n📥 모델 다운로드 확인: {model_name}")

    url = os.getenv("OLLAMA_URL", "http://ollama:11434")

    try:
        # 설치된 모델 목록 확인
        response = requests.get(f"{url}/api/tags", timeout=30)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]

            if model_name in model_names:
                print(f"   ✅ 모델 {model_name} 이미 설치됨")
                return True
            else:
                print(f"   📥 모델 {model_name} 다운로드 시작...")

                # 모델 다운로드
                pull_response = requests.post(
                    f"{url}/api/pull",
                    json={"name": model_name},
                    timeout=1800  # 30분 타임아웃
                )

                if pull_response.status_code == 200:
                    print(f"   ✅ 모델 {model_name} 다운로드 완료!")
                    return True
                else:
                    print(f"   ❌ 모델 다운로드 실패: {pull_response.status_code}")
                    return False
    except Exception as e:
        print(f"   ❌ 모델 다운로드 중 오류: {e}")
        return False

def run_data_generation(total_count=50, batch_size=10):
    """배치 단위로 안전하게 데이터 생성"""
    print(f"\n🚀 학습 데이터 생성 시작:")
    print(f"   📊 목표: {total_count}개")
    print(f"   📦 배치 크기: {batch_size}개")

    generated_files = []
    total_generated = 0
    batch_num = 1

    while total_generated < total_count:
        remaining = total_count - total_generated
        current_batch = min(batch_size, remaining)

        print(f"\n📦 배치 {batch_num} 시작 ({current_batch}개)")
        print(f"   📈 진행률: {total_generated}/{total_count} ({total_generated/total_count*100:.1f}%)")

        start_time = time.time()

        try:
            # 고유한 파일명 생성
            timestamp = int(time.time())
            output_file = f"generated_data/batch_{batch_num}_{timestamp}.json"

            # 데이터 생성
            generator = MuseumDataGenerator()
            examples = generator.generate_training_examples(
                max_examples=current_batch,
                api_preference="ollama"
            )

            if examples:
                # 결과 저장
                output_path = generator.save_training_data(examples, output_file)
                generated_files.append(output_path)

                elapsed = time.time() - start_time
                total_generated += len(examples)

                print(f"   ✅ 배치 {batch_num} 완료!")
                print(f"   ⏱️ 소요시간: {elapsed:.1f}초")
                print(f"   📁 저장 위치: {output_path}")

                # 진행 상황 저장
                progress = {
                    "total_target": total_count,
                    "total_generated": total_generated,
                    "batches_completed": batch_num,
                    "generated_files": generated_files,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                with open("generated_data/progress.json", "w", encoding="utf-8") as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)

                # 배치 간 대기 (서버 부하 분산)
                if total_generated < total_count:
                    wait_time = 10
                    print(f"   ⏳ {wait_time}초 대기 중...")
                    time.sleep(wait_time)

            else:
                print(f"   ❌ 배치 {batch_num} 실패: 데이터가 생성되지 않음")

        except Exception as e:
            print(f"   ❌ 배치 {batch_num} 오류: {e}")
            print("   🔄 10초 후 재시도...")
            time.sleep(10)
            continue

        batch_num += 1

    print(f"\n🎉 데이터 생성 완료!")
    print(f"   📊 총 생성량: {total_generated}개")
    print(f"   📁 생성 파일: {len(generated_files)}개")

    # 최종 통합 파일 생성
    if generated_files:
        merge_all_files(generated_files)

    return total_generated

def merge_all_files(file_list):
    """생성된 모든 배치 파일을 하나로 통합"""
    print("\n🔄 배치 파일 통합 중...")

    all_data = []
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                all_data.extend(batch_data)
                print(f"   ✅ {file_path}: {len(batch_data)}개")
        except Exception as e:
            print(f"   ❌ {file_path}: 읽기 실패 - {e}")

    # 통합 파일 저장
    output_path = f"generated_data/all_generated_{int(time.time())}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"   📁 통합 파일: {output_path}")
    print(f"   📊 총 데이터: {len(all_data)}개")

    return output_path

def main():
    print("=" * 60)
    print("🏛️  국립중앙박물관 학습 데이터 생성 (서버용)")
    print("=" * 60)

    # 디렉토리 생성
    os.makedirs("generated_data", exist_ok=True)

    # 리소스 체크
    mem_mb, cpu_count, has_gpu = check_server_resources()

    # 설정 최적화
    model, batch_size, parallel = optimize_settings(mem_mb, cpu_count, has_gpu)

    # Ollama 연결 확인
    if not check_ollama_connection():
        return

    # 모델 다운로드 확인
    if not ensure_model_downloaded(model):
        return

    # 사용자 확인
    print(f"\n🎯 생성 계획:")
    print(f"   🤖 사용 모델: {model}")
    print(f"   📦 배치 크기: {batch_size}개")
    print(f"   📊 목표 수량: 원하는 개수를 입력하세요")

    try:
        target_count = int(input("\n생성할 데이터 개수를 입력하세요 (기본: 30): ") or "30")

        if target_count <= 0:
            print("❌ 올바른 개수를 입력하세요.")
            return

        print(f"\n🚀 {target_count}개 데이터 생성을 시작합니다!")

        # 데이터 생성 실행
        total_generated = run_data_generation(
            total_count=target_count,
            batch_size=batch_size
        )

        if total_generated > 0:
            print(f"\n✅ 성공적으로 {total_generated}개의 학습 데이터를 생성했습니다!")
            print("\n📋 다음 단계:")
            print("   1. generated_data/ 폴더에서 생성된 파일 확인")
            print("   2. python merge_training_data.py merge 명령으로 기존 데이터와 병합")
            print("   3. python src/fine_tuning.py 명령으로 모델 재훈련")
        else:
            print("\n❌ 데이터 생성에 실패했습니다.")

    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다.")
    except ValueError:
        print("❌ 올바른 숫자를 입력하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()