FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p /app/data /app/generated_data

# 로그 디렉토리 생성
RUN mkdir -p /app/logs

# 실행 권한 추가
RUN chmod +x generate_data.py merge_training_data.py

# 기본 명령어 (대화형 모드)
CMD ["/bin/bash"]