import os
import json
import time
import random
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrainingExample:
    question: str
    general_response: str
    children_response: str
    source_content: str
    source_url: str

class FreeAPIGenerator:
    """무료 API들을 활용한 학습 데이터 생성기"""

    def __init__(self):
        self.apis = {
            "ollama": self._use_ollama,
            "huggingface": self._use_huggingface,
            "groq": self._use_groq,
            "openrouter": self._use_openrouter,
        }

    def _use_ollama(self, prompt: str) -> Optional[str]:
        """Ollama API 사용 (로컬/서버/Docker 무료)"""
        try:
            # 환경 자동 감지
            if os.path.exists('/.dockerenv'):
                # Docker 컨테이너 내부인 경우
                url = os.getenv("OLLAMA_URL", "http://ollama:11434")
                logger.info("Docker 환경 감지됨")
            elif 'CONDA_DEFAULT_ENV' in os.environ:
                # 아나콘다 가상환경
                url = os.getenv("OLLAMA_URL", "http://localhost:11434")
                logger.info(f"아나콘다 환경 감지됨: {os.environ['CONDA_DEFAULT_ENV']}")
            else:
                # 일반 환경
                url = os.getenv("OLLAMA_URL", "http://localhost:11434")

            model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 512,  # 응답 길이 제한
                    "num_ctx": 2048      # 컨텍스트 길이
                }
            }

            # Docker 환경에서는 더 긴 타임아웃 설정
            timeout = 600 if os.path.exists('/.dockerenv') else 300

            response = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama 응답 오류: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama API 호출 실패: {e}")
        return None

    def _use_huggingface(self, prompt: str) -> Optional[str]:
        """HuggingFace Inference API 사용 (무료)"""
        try:
            token = os.getenv("HF_TOKEN")
            if not token:
                logger.warning("HF_TOKEN이 설정되지 않았습니다.")
                return None

            # 무료로 사용할 수 있는 한국어 모델들
            models = [
                "microsoft/DialoGPT-medium",
                "EleutherAI/gpt-neo-1.3B",
                "facebook/blenderbot-400M-distill"
            ]

            model = random.choice(models)
            url = f"https://api-inference.huggingface.co/models/{model}"

            headers = {"Authorization": f"Bearer {token}"}
            payload = {"inputs": prompt}

            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
        except Exception as e:
            logger.error(f"HuggingFace API 호출 실패: {e}")
        return None

    def _use_groq(self, prompt: str) -> Optional[str]:
        """Groq API 사용 (빠른 무료 API)"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY가 설정되지 않았습니다.")
                return None

            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "llama-3.1-8b-instant",  # 무료 모델
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000
            }

            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq API 호출 실패: {e}")
        return None

    def _use_openrouter(self, prompt: str) -> Optional[str]:
        """OpenRouter API 사용 (다양한 무료 모델 제공)"""
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.warning("OPENROUTER_API_KEY가 설정되지 않았습니다.")
                return None

            # 무료로 사용할 수 있는 모델들
            free_models = [
                "microsoft/phi-3-mini-128k-instruct:free",
                "huggingface/zephyr-7b-beta:free",
                "openchat/openchat-7b:free"
            ]

            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": random.choice(free_models),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }

            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenRouter API 호출 실패: {e}")
        return None

    def generate_response(self, prompt: str, preferred_api: str = "ollama") -> Optional[str]:
        """지정된 API로 응답 생성, 실패 시 다른 API들 시도"""
        apis_to_try = [preferred_api] + [api for api in self.apis.keys() if api != preferred_api]

        for api_name in apis_to_try:
            if api_name in self.apis:
                result = self.apis[api_name](prompt)
                if result:
                    logger.info(f"{api_name} API로 응답 생성 성공")
                    return result

                # API 호출 간격 조절
                time.sleep(1)

        logger.error("모든 API 호출이 실패했습니다.")
        return None

class MuseumDataGenerator:
    """박물관 크롤링 데이터 기반 학습 데이터 생성기"""

    def __init__(self, curated_data_dir: str = "data_curated"):
        self.curated_data_dir = Path(curated_data_dir)
        self.api_generator = FreeAPIGenerator()

    def load_curated_data(self) -> List[Dict]:
        """큐레이션된 데이터 로드"""
        data_files = list(self.curated_data_dir.glob("*.json"))
        all_data = []

        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.append(data)
            except Exception as e:
                logger.error(f"파일 로드 실패 {file_path}: {e}")

        logger.info(f"총 {len(all_data)}개의 데이터 로드 완료")
        return all_data

    def generate_questions_from_content(self, content: Dict) -> List[str]:
        """컨텐츠 기반 질문 생성"""
        title = content.get("title", "")
        text = content.get("text", "")

        # 컨텐츠 타입에 따른 질문 템플릿
        question_templates = {
            "소장품": [
                f"{self._extract_artifact_name(text)}에 대해 알려줘",
                f"{self._extract_artifact_name(text)}의 특징은 무엇인가요?",
                f"{self._extract_artifact_name(text)}는 언제 만들어졌나요?",
                f"{self._extract_artifact_name(text)}의 역사적 의미는?",
                f"{self._extract_artifact_name(text)} 작품 설명해줘"
            ],
            "전시": [
                "현재 전시 중인 특별전은?",
                "이번 달 추천 전시가 뭐예요?",
                "가족과 함께 볼 만한 전시 추천해줘",
                "어린이가 좋아할 전시는?",
                "이 전시의 주요 작품들은?"
            ],
            "관람정보": [
                "박물관 운영시간이 어떻게 되나요?",
                "입장료는 얼마인가요?",
                "지하철로 어떻게 가나요?",
                "주차장이 있나요?",
                "휠체어로도 관람 가능한가요?"
            ]
        }

        # 컨텐츠 타입 추정
        content_type = self._classify_content_type(content)
        templates = question_templates.get(content_type, question_templates["소장품"])

        # 일부 템플릿 선택 (3-5개)
        selected_templates = random.sample(templates, min(len(templates), random.randint(3, 5)))

        return selected_templates

    def _extract_artifact_name(self, text: str) -> str:
        """텍스트에서 유물명 추출"""
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['다른명칭:', '전시명칭:', '작품명:']):
                # 콜론 이후 내용 추출
                if ':' in line:
                    name = line.split(':', 1)[1].strip()
                    # 괄호 안의 내용 제거
                    name = re.sub(r'\([^)]*\)', '', name).strip()
                    if name:
                        return name

        # 첫 번째 줄에서 유물명 추정
        first_line = lines[0] if lines else ""
        # 특수문자나 숫자 제거하고 깔끔한 이름 반환
        import re
        clean_name = re.sub(r'[^\w\s가-힣]', '', first_line).strip()
        return clean_name[:20] if clean_name else "이 유물"

    def _classify_content_type(self, content: Dict) -> str:
        """컨텐츠 타입 분류"""
        title = content.get("title", "").lower()
        text = content.get("text", "").lower()
        url = content.get("url", "").lower()

        if "소장품" in title or "relicId" in url:
            return "소장품"
        elif any(keyword in title or keyword in text for keyword in ["전시", "exhibition", "특별전"]):
            return "전시"
        elif any(keyword in title or keyword in text for keyword in ["관람", "이용", "안내", "시간", "요금"]):
            return "관람정보"
        else:
            return "소장품"  # 기본값

    def create_response_prompts(self, question: str, content: Dict) -> Tuple[str, str]:
        """일반용/어린이용 응답 생성 프롬프트 생성"""
        context = f"""
제목: {content.get('title', '')}
내용: {content.get('text', '')[:1000]}  # 토큰 수 제한
URL: {content.get('url', '')}
"""

        general_prompt = f"""당신은 국립중앙박물관의 전문 도슨트입니다.
다음 질문에 대해 제공된 정보를 바탕으로 정확하고 전문적으로 답변해주세요.

질문: {question}

참고 정보:
{context}

답변 요구사항:
- 전문적이지만 이해하기 쉬운 설명
- 역사적 배경과 문화적 의미 포함
- 200-300자 내외
- 정확한 정보만 사용

답변:"""

        children_prompt = f"""당신은 어린이를 위한 친근한 박물관 선생님입니다.
다음 질문에 대해 어린이가 이해하기 쉽고 재미있게 답변해주세요.

질문: {question}

참고 정보:
{context}

답변 요구사항:
- 쉬운 단어 사용
- 재미있고 친근한 설명
- 적절한 이모지 포함 (2-3개)
- 100-150자 내외
- 어린이 호기심 자극하는 내용

답변:"""

        return general_prompt, children_prompt

    def generate_training_examples(self,
                                 max_examples: int = 50,
                                 api_preference: str = "ollama") -> List[TrainingExample]:
        """학습 데이터 예시 생성"""
        curated_data = self.load_curated_data()
        if not curated_data:
            logger.error("크롤링된 데이터가 없습니다.")
            return []

        examples = []
        processed_count = 0

        # 데이터를 셔플해서 다양성 확보
        random.shuffle(curated_data)

        for content in curated_data:
            if processed_count >= max_examples:
                break

            # 각 컨텐츠마다 1-2개의 질문 생성
            questions = self.generate_questions_from_content(content)

            for question in questions[:2]:  # 최대 2개만
                if processed_count >= max_examples:
                    break

                try:
                    # 프롬프트 생성
                    general_prompt, children_prompt = self.create_response_prompts(question, content)

                    # API 호출 (재시도 로직 포함)
                    general_response = self.api_generator.generate_response(
                        general_prompt, api_preference
                    )

                    if not general_response:
                        continue

                    # 어린이용 응답 생성 (약간의 지연 추가)
                    time.sleep(2)
                    children_response = self.api_generator.generate_response(
                        children_prompt, api_preference
                    )

                    if not children_response:
                        continue

                    # 응답 정제
                    general_response = self._clean_response(general_response)
                    children_response = self._clean_response(children_response)

                    example = TrainingExample(
                        question=question,
                        general_response=general_response,
                        children_response=children_response,
                        source_content=content.get('text', '')[:500],
                        source_url=content.get('url', '')
                    )

                    examples.append(example)
                    processed_count += 1

                    logger.info(f"진행률: {processed_count}/{max_examples} - '{question}'")

                    # API 호출 제한 방지를 위한 지연
                    time.sleep(3)

                except Exception as e:
                    logger.error(f"데이터 생성 오류: {e}")
                    continue

        logger.info(f"총 {len(examples)}개의 학습 데이터 생성 완료")
        return examples

    def _clean_response(self, response: str) -> str:
        """응답 텍스트 정제"""
        # 불필요한 프롬프트 부분 제거
        response = response.strip()

        # "답변:" 이후 부분만 추출
        if "답변:" in response:
            response = response.split("답변:")[-1].strip()

        # 너무 긴 응답 자르기 (한국어 기준 500자)
        if len(response) > 500:
            response = response[:500] + "..."

        return response

    def save_training_data(self, examples: List[TrainingExample],
                          filename: str = "generated_training_data.json"):
        """생성된 학습 데이터 저장"""
        training_data = []

        for example in examples:
            training_data.append({
                "question": example.question,
                "general_response": example.general_response,
                "children_response": example.children_response,
                "source_content": example.source_content,
                "source_url": example.source_url,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })

        output_path = Path("data") / filename
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        logger.info(f"학습 데이터가 {output_path}에 저장되었습니다.")
        return str(output_path)

# 실행 함수
def generate_museum_training_data(max_examples: int = 30,
                                api_preference: str = "ollama"):
    """박물관 학습 데이터 생성 실행"""
    generator = MuseumDataGenerator()

    print(f"🚀 박물관 학습 데이터 생성 시작 (목표: {max_examples}개)")
    print(f"📡 사용 API: {api_preference}")

    # API 설정 가이드 출력
    print("\n📋 API 설정 가이드:")
    print("Ollama: export OLLAMA_MODEL=llama3.2:3b")
    print("HuggingFace: export HF_TOKEN=your_token")
    print("Groq: export GROQ_API_KEY=your_key")
    print("OpenRouter: export OPENROUTER_API_KEY=your_key")

    examples = generator.generate_training_examples(
        max_examples=max_examples,
        api_preference=api_preference
    )

    if examples:
        output_path = generator.save_training_data(examples)
        print(f"✅ 성공! {len(examples)}개의 학습 데이터가 {output_path}에 저장되었습니다.")

        # 샘플 데이터 출력
        if examples:
            print("\n📝 생성 샘플:")
            sample = examples[0]
            print(f"Q: {sample.question}")
            print(f"A(일반): {sample.general_response[:100]}...")
            print(f"A(어린이): {sample.children_response[:100]}...")
    else:
        print("❌ 학습 데이터 생성 실패. API 설정을 확인하세요.")

if __name__ == "__main__":
    import re
    # 기본 실행
    generate_museum_training_data(max_examples=20, api_preference="ollama")