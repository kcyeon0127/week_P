# 🐍 아나콘다 가상환경으로 서버 실행 가이드

## 🚀 **훨씬 간단한 방법!**

Docker 없이 아나콘다 가상환경에서 바로 실행하는게 더 쉽고 빠릅니다.

### 📋 **1단계: 서버에서 Ollama 직접 설치**

#### **Ubuntu/Linux 서버에 Ollama 설치**
```bash
# Ollama 설치 (1분 소요)
curl -fsSL https://ollama.ai/install.sh | sh

# 백그라운드로 실행
ollama serve &

# 또는 systemd 서비스로 등록
sudo systemctl enable ollama
sudo systemctl start ollama

# 설치 확인
ollama --version
curl http://localhost:11434/api/tags
```

#### **GPU 서버라면 CUDA 자동 감지**
```bash
# GPU 확인
nvidia-smi

# Ollama가 자동으로 GPU 사용함
# 별도 설정 불필요!
```

### 🐍 **2단계: 아나콘다 가상환경 설정**

#### **새 가상환경 생성**
```bash
# Python 3.11 가상환경 생성
conda create -n nmk-rag python=3.11 -y

# 가상환경 활성화
conda activate nmk-rag

# 필요한 패키지 설치
pip install -r requirements.txt
```

#### **환경변수 설정**
```bash
# ~/.bashrc 또는 현재 세션에 추가
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.2:3b"
export OLLAMA_NUM_PARALLEL=4
```

### ⚡ **3단계: 모델 다운로드**

#### **공용 서버 고려한 모델 선택**
```bash
# 가벼운 모델부터 (1.6GB)
ollama pull gemma2:2b

# 중간 성능 모델 (2.0GB) - 추천
ollama pull llama3.2:3b

# 고성능 모델 (4.7GB) - GPU 서버 권장
ollama pull llama3.1:8b

# 설치된 모델 확인
ollama list
```

### 🚀 **4단계: 데이터 생성 실행**

#### **간단한 명령어들**
```bash
# 프로젝트 폴더로 이동
cd /path/to/nmK-rag

# 가상환경 활성화 (매번 필요)
conda activate nmk-rag

# 일반 데이터 생성 (30개)
python generate_data.py --api ollama --count 30

# 또는 서버 최적화 스크립트 사용
python server_generate.py
```

#### **실행 과정**
```
🐍 아나콘다 환경 감지됨
🖥️  서버 리소스 체크:
   💾 총 메모리: 32.0GB
   🚀 CPU 코어: 16개
   🎮 GPU: ✅ RTX 4090 감지됨

⚙️ 최적화 설정:
   🤖 모델: llama3.2:3b
   📦 배치 크기: 15개
   🔄 병렬 처리: 4개

생성할 데이터 개수를 입력하세요: 50
```

### 📊 **성능 비교: Docker vs 아나콘다**

| 방식 | 설정 복잡도 | 실행 속도 | 메모리 사용량 | 디스크 사용량 |
|------|-------------|-----------|---------------|---------------|
| **아나콘다** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 낮음 | 적음 |
| Docker | ⭐⭐⭐ | ⭐⭐⭐⭐ | 높음 | 많음 |

### ✅ **아나콘다 방식의 장점**

#### **1. 간단함**
- Docker 이미지 빌드 불필요
- 복잡한 설정 파일 불필요
- 바로 실행 가능

#### **2. 효율성**
- 컨테이너 오버헤드 없음
- 직접 GPU 접근
- 더 빠른 실행 속도

#### **3. 디버깅 용이**
- 직접적인 파일 접근
- 실시간 코드 수정 가능
- 로그 확인 간편

### 🔧 **공용 서버에서 매너있게 사용하기**

#### **실행 전 체크**
```bash
# 시스템 리소스 확인
htop                    # CPU/메모리 사용률
nvidia-smi             # GPU 사용률
df -h                   # 디스크 여유공간
ps aux | grep ollama    # Ollama 실행 중인지 확인
```

#### **백그라운드 실행**
```bash
# nohup으로 백그라운드 실행
nohup python server_generate.py > generation.log 2>&1 &

# 진행상황 실시간 확인
tail -f generation.log

# 프로세스 확인
ps aux | grep python
```

#### **완료 후 정리**
```bash
# 가상환경 비활성화
conda deactivate

# Ollama 중지 (필요시)
pkill ollama

# 생성된 파일 확인
ls -la data/generated_*
```

### 🎯 **추천 실행 순서**

#### **1. 최소 설치**
```bash
# Ollama만 설치 (Docker 없음)
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
```

#### **2. 가상환경 준비**
```bash
conda create -n nmk-rag python=3.11 -y
conda activate nmk-rag
pip install -r requirements.txt
```

#### **3. 작은 모델로 테스트**
```bash
ollama pull gemma2:2b
python generate_data.py --api ollama --count 10
```

#### **4. 성공하면 확장**
```bash
ollama pull llama3.2:3b
python server_generate.py
# 입력: 50
```

### 💡 **팀원 배려 팁**

#### **실행 시간 공지**
```bash
echo "AI 데이터 생성 시작: $(date)" >> /shared/notices.txt
python server_generate.py
echo "AI 데이터 생성 완료: $(date)" >> /shared/notices.txt
```

#### **리소스 사용 최소화**
```bash
# CPU 제한으로 실행
nice -n 19 python server_generate.py

# 메모리 제한
ulimit -v 8388608  # 8GB 제한
```

**결론: 아나콘다 가상환경이 훨씬 간단하고 효율적입니다!** 🐍✨