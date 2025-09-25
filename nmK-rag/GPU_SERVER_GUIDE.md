# 🎮 GPU 서버 실행 가이드

## ⚡ **GPU 서버 최적화 설정**

### 🔍 **1단계: GPU 환경 확인**
```bash
# GPU 확인
nvidia-smi

# NVIDIA Docker 런타임 확인
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Docker Compose 버전 확인 (v2.3+ 필요)
docker-compose version
```

### 🚀 **2단계: 최적화된 실행**

#### **GPU 서버용 Docker Compose 실행**
```bash
# 프로젝트 폴더로 이동
cd /path/to/nmK-rag

# GPU 지원으로 서비스 시작
docker-compose up -d

# GPU 할당 확인
docker-compose exec ollama nvidia-smi
```

#### **고성능 모델 다운로드**
```bash
# GPU 서버라면 큰 모델 추천! (4.7GB)
docker-compose exec ollama ollama pull llama3.1:8b

# 또는 더 큰 모델 (7.4GB) - 최고 품질
docker-compose exec ollama ollama pull llama3.1:70b

# 한국어 특화 모델도 가능 (4.4GB)
docker-compose exec ollama ollama pull qwen2.5:7b

# 다운로드 확인
docker-compose exec ollama ollama list
```

### ⚙️ **3단계: GPU 최적화 데이터 생성**

#### **GPU 가속 설정**
```bash
# 데이터 생성기 접속
docker-compose exec data-generator bash

# GPU 환경변수 설정 (컨테이너 내부에서)
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 모든 GPU 사용
export OLLAMA_MODEL=llama3.1:8b      # 고성능 모델
export OLLAMA_NUM_PARALLEL=4         # 병렬 처리 증가

# 최적화된 데이터 생성 실행
python server_generate.py
```

#### **예상 성능 (GPU 서버)**
```
🎮 GPU 서버 감지됨!
⚙️ 최적화 설정:
   🤖 모델: llama3.1:8b
   📦 배치 크기: 20개 (GPU 가속으로 증가)
   🔄 병렬 처리: 6개

생성할 데이터 개수를 입력하세요: 100
```

### 📊 **GPU 서버 성능 비교**

| GPU | RAM | 모델 | 배치크기 | 100개 생성시간 | 품질 |
|-----|-----|------|----------|----------------|------|
| RTX 4090 | 32GB | llama3.1:8b | 20개 | **15-20분** | ⭐⭐⭐⭐⭐ |
| RTX 3080 | 24GB | llama3.2:3b | 15개 | 25-30분 | ⭐⭐⭐⭐ |
| T4 | 16GB | llama3.2:3b | 10개 | 30-40분 | ⭐⭐⭐⭐ |

### 🔧 **GPU 서버 전용 명령어**

#### **실시간 GPU 모니터링**
```bash
# GPU 사용률 실시간 모니터링
watch -n 1 nvidia-smi

# Docker 컨테이너 GPU 사용량
docker stats

# Ollama GPU 사용 확인
docker-compose exec ollama nvidia-smi
```

#### **메모리 최적화**
```bash
# GPU 메모리 부족시 컨테이너 재시작
docker-compose restart ollama

# 더 작은 배치로 다시 실행
# server_generate.py에서 자동으로 조정됨
```

### 🎯 **GPU 서버 실행 순서**

#### **1. GPU 환경 확인**
```bash
nvidia-smi
# GPU가 보이는지 확인
```

#### **2. Docker GPU 지원 확인**
```bash
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
# Docker에서 GPU 접근 가능한지 확인
```

#### **3. 서비스 시작**
```bash
cd /path/to/nmK-rag
docker-compose up -d
```

#### **4. 고성능 모델 다운로드**
```bash
docker-compose exec ollama ollama pull llama3.1:8b
```

#### **5. 대량 데이터 생성**
```bash
docker-compose exec data-generator bash
python server_generate.py
# 입력: 100 (또는 원하는 개수)
```

### ⚠️ **GPU 서버 주의사항**

#### **NVIDIA Docker 런타임 필요**
```bash
# 설치되지 않은 경우:
curl -fsSL https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-runtime-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/nvidia-container-runtime-keyring.gpg] https://nvidia.github.io/nvidia-container-runtime/ubuntu20.04/$(ARCH) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

sudo apt-get update
sudo apt-get install -y nvidia-container-runtime

# Docker 재시작
sudo systemctl restart docker
```

#### **CUDA 버전 호환성**
```bash
# CUDA 버전 확인
nvcc --version

# 호환되는 Ollama 이미지 사용
# docker-compose.yml에서 이미 설정됨
```

### 🚀 **GPU 서버 최고 성능 설정**

#### **멀티 GPU 활용**
```yaml
# docker-compose.yml 고급 설정
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['0', '1', '2', '3']  # 특정 GPU 지정
          capabilities: [gpu]
```

#### **대량 생성 명령어**
```bash
# 500개 데이터를 빠르게 생성
python server_generate.py
# 입력: 500

# 예상 소요시간: 1-2시간 (GPU 서버)
# vs 8-10시간 (CPU만)
```

### ✅ **GPU 서버 완료 체크리스트**

- [ ] `nvidia-smi` 정상 작동
- [ ] Docker GPU 지원 확인
- [ ] `docker-compose up -d` 실행
- [ ] Ollama에서 GPU 인식 확인
- [ ] 고성능 모델 다운로드
- [ ] 대량 데이터 생성 실행
- [ ] GPU 사용률 모니터링

**GPU 서버에서는 10배 빠른 속도로 고품질 데이터를 대량 생성할 수 있습니다!** 🔥🚀