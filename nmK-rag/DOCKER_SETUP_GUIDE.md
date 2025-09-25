# 🐳 Docker 서버에서 학습 데이터 생성하기

## 🎯 진행해야 할 단계

### 1단계: 환경 준비 ✅ (완료됨)
```bash
# 이미 준비된 파일들:
# - docker-compose.yml (Ollama + 데이터 생성기)
# - Dockerfile (Python 환경)
# - server_generate.py (자동화 스크립트)
```

### 2단계: Docker Compose 실행 🚀

#### 2-1. 서비스 시작
```bash
# 서버에서 실행
cd /path/to/nmK-rag
docker-compose up -d

# 상태 확인
docker-compose ps
```

예상 출력:
```
NAME              COMMAND             SERVICE         STATUS    PORTS
nmk-ollama        "/bin/ollama serve" ollama          running   0.0.0.0:11434->11434/tcp
nmk-data-generator "/bin/bash"        data-generator   running
```

#### 2-2. Ollama 서비스 확인
```bash
# Ollama 연결 테스트
curl http://localhost:11434/api/tags

# 로그 확인
docker-compose logs ollama
```

### 3단계: 모델 다운로드 📥

#### 3-1. 권장 모델 선택
```bash
# 서버 사양에 따라 선택:

# 저사양 (8GB RAM): 가벼운 모델
docker-compose exec ollama ollama pull gemma2:2b

# 중간 사양 (16GB RAM): 추천 모델 ⭐
docker-compose exec ollama ollama pull llama3.2:3b

# 고사양 (32GB+ RAM): 고품질 모델
docker-compose exec ollama ollama pull llama3.1:8b
```

#### 3-2. 모델 다운로드 확인
```bash
# 설치된 모델 목록
docker-compose exec ollama ollama list

# 예상 출력:
# NAME              ID         SIZE    MODIFIED
# llama3.2:3b       a80c4f17acd5  2.0 GB   2 minutes ago
```

### 4단계: 학습 데이터 생성 🤖

#### 4-1. 데이터 생성기 접속
```bash
# 데이터 생성기 컨테이너 접속
docker-compose exec data-generator bash
```

#### 4-2. 자동 생성 실행
```bash
# 컨테이너 내부에서 실행
python server_generate.py
```

실행하면 다음과 같이 진행됩니다:
```
🐳 Docker 환경 리소스 체크:
   💾 총 메모리: 16.0GB
   💾 사용 가능: 14.2GB
   🚀 CPU 코어: 8개
   🎮 GPU: ❌ 감지되지 않음

⚙️ 최적화 설정:
   🤖 모델: llama3.2:3b
   📦 배치 크기: 10
   🔄 병렬 처리: 2

🔗 Ollama 연결 체크:
   ✅ Ollama 연결 성공!
   📚 설치된 모델: 1개
      - llama3.2:3b

📥 모델 다운로드 확인: llama3.2:3b
   ✅ 모델 llama3.2:3b 이미 설치됨

🎯 생성 계획:
   🤖 사용 모델: llama3.2:3b
   📦 배치 크기: 10개
   📊 목표 수량: 원하는 개수를 입력하세요

생성할 데이터 개수를 입력하세요 (기본: 30): 50
```

#### 4-3. 생성 과정 모니터링
```bash
# 별도 터미널에서 진행상황 확인
docker-compose exec data-generator ls -la generated_data/

# 실시간 로그 확인
docker-compose logs -f data-generator
```

### 5단계: 결과 확인 및 병합 📊

#### 5-1. 생성된 파일 확인
```bash
# 생성된 데이터 파일들
docker-compose exec data-generator ls -la generated_data/
```

예상 출력:
```
total 156
drwxr-xr-x 2 root root 4096 Jan 15 10:30 .
drwxr-xr-x 8 root root 4096 Jan 15 10:15 ..
-rw-r--r-- 1 root root 8432 Jan 15 10:18 batch_1_1736934518.json
-rw-r--r-- 1 root root 9124 Jan 15 10:22 batch_2_1736934768.json
-rw-r--r-- 1 root root 7892 Jan 15 10:26 batch_3_1736935012.json
-rw-r--r-- 1 root root 45234 Jan 15 10:30 all_generated_1736935234.json
-rw-r--r-- 1 root root 412 Jan 15 10:30 progress.json
```

#### 5-2. 데이터 품질 확인
```bash
# 컨테이너 내에서 분석
python merge_training_data.py analyze generated_data/all_generated_*.json
```

#### 5-3. 기존 데이터와 병합
```bash
# 기존 수동 작성 데이터와 병합
python merge_training_data.py merge \
  --original data/training_data.json \
  --generated generated_data/all_generated_*.json \
  --output data/final_training_data.json
```

### 6단계: 모델 재훈련 🔄

#### 6-1. 훈련/검증 데이터 분할
```bash
# 80% 훈련, 20% 검증
python merge_training_data.py split data/final_training_data.json --ratio 0.8
```

#### 6-2. LoRA 모델 재훈련
```bash
# 새로운 데이터로 모델 훈련
python src/fine_tuning.py --data_path data/final_training_data_train.json

# 완료까지 약 30분-2시간 소요 (데이터량과 하드웨어에 따라)
```

### 7단계: 배포 및 테스트 🎉

#### 7-1. 호스트로 파일 복사
```bash
# Docker 볼륨에서 호스트로 복사
docker cp nmk-data-generator:/app/data ./
docker cp nmk-data-generator:/app/models ./
```

#### 7-2. Streamlit 앱 테스트
```bash
# 호스트에서 앱 실행
streamlit run app.py
```

## 🚨 문제 해결

### Ollama 연결 실패
```bash
# Ollama 서비스 재시작
docker-compose restart ollama

# 로그 확인
docker-compose logs ollama

# 포트 확인
netstat -tlnp | grep 11434
```

### 메모리 부족
```bash
# 더 작은 배치 크기로 재시도
# server_generate.py에서 배치 크기가 자동 조정됨

# 또는 더 작은 모델 사용
docker-compose exec ollama ollama pull gemma2:2b
export OLLAMA_MODEL=gemma2:2b
```

### GPU 사용하려면
```yaml
# docker-compose.yml에서 주석 해제
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 📊 예상 성능

| 서버 사양 | 모델 | 배치 크기 | 50개 생성 시간 | 품질 |
|-----------|------|-----------|----------------|------|
| 8GB RAM | gemma2:2b | 5개 | 60-90분 | ⭐⭐⭐ |
| 16GB RAM | llama3.2:3b | 10개 | 45-60분 | ⭐⭐⭐⭐ |
| 32GB RAM + GPU | llama3.1:8b | 15개 | 30-40분 | ⭐⭐⭐⭐⭐ |

## 🎯 완료 체크리스트

- [ ] Docker Compose 실행
- [ ] Ollama 연결 확인
- [ ] 모델 다운로드
- [ ] 데이터 생성 실행
- [ ] 결과 파일 확인
- [ ] 데이터 병합
- [ ] 모델 재훈련
- [ ] Streamlit 앱 테스트

**모든 단계가 자동화되어 있으니 단계별로 실행하시면 됩니다!** 🚀