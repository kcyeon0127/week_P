import os, json, requests
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
import torch
from threading import Thread
import logging

logger = logging.getLogger(__name__)

class MultiTargetLLM:
    """멀티타겟 응답을 위한 LLM 클래스"""

    def __init__(self):
        self.base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_base_model(self) -> tuple:
        """베이스 모델 로드"""
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return tokenizer, model

    def load_lora_model(self, model_path: str, model_type: str):
        """특정 LoRA 모델 로드"""
        if model_type in self.models:
            logger.info(f"{model_type} 모델이 이미 로드되어 있습니다.")
            return

        try:
            # 베이스 모델 로드
            tokenizer, base_model = self.load_base_model()

            # LoRA 모델 로드
            if os.path.exists(model_path):
                model = PeftModel.from_pretrained(base_model, model_path)
                logger.info(f"{model_type} LoRA 모델 로드 성공: {model_path}")
            else:
                logger.warning(f"LoRA 모델 경로가 존재하지 않아 베이스 모델을 사용합니다: {model_path}")
                model = base_model

            self.models[model_type] = model
            self.tokenizers[model_type] = tokenizer

        except Exception as e:
            logger.error(f"{model_type} 모델 로드 실패: {e}")
            # 실패 시 베이스 모델 사용
            tokenizer, model = self.load_base_model()
            self.models[model_type] = model
            self.tokenizers[model_type] = tokenizer

    def load_all_models(self, models_dir: str = "models"):
        """모든 타겟 모델들 로드"""
        model_configs = {
            "general": os.path.join(models_dir, "lora-general"),
            "children": os.path.join(models_dir, "lora-children")
        }

        for model_type, model_path in model_configs.items():
            self.load_lora_model(model_path, model_type)

        # 일반 모델이 없으면 베이스 모델을 일반 모델로 로드 (fallback용)
        if "general" not in self.models:
            tokenizer, model = self.load_base_model()
            self.models["general"] = model
            self.tokenizers["general"] = tokenizer

        logger.info(f"로드된 모델들: {list(self.models.keys())}")

    def get_system_prompt(self, target_type: str) -> str:
        """타겟별 시스템 프롬프트"""
        if target_type == "children":
            return """너는 어린이를 위한 친절한 박물관 안내봇이야.
어려운 용어는 쉽게 설명하고, 재미있고 이해하기 쉽게 대답해줘.
이모지를 적절히 사용해서 더 친근하게 대화해줘."""
        else:
            return """너는 국립중앙박물관 전문 안내 도슨트이다.
정확하고 전문적인 정보를 제공하며, 교육적 가치가 있는 설명을 해줘."""

    def chat_with_model(self, target_type: str, user_query: str, context_snippets: List[Dict]) -> str:
        """특정 모델로 채팅"""
        # 모델이 로드되지 않은 경우 로드
        if target_type not in self.models:
            logger.warning(f"{target_type} 모델이 로드되지 않았습니다. 일반 모델을 사용합니다.")
            target_type = "general"

        # 일반 모델도 없다면 로드
        if target_type not in self.models:
            logger.info(f"{target_type} 모델을 로드합니다...")
            self.load_lora_model(f"models/lora-{target_type}", target_type)

        tokenizer = self.tokenizers[target_type]
        model = self.models[target_type]

        # 컨텍스트 포맷팅
        ctx = "\n\n".join([
            f"[{i+1}] {s['title']} — {s.get('url','')}\n{s['text']}"
            for i, s in enumerate(context_snippets)
        ])

        # 시스템 프롬프트
        system_prompt = self.get_system_prompt(target_type)

        # 메시지 구성
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"[컨텍스트]\n{ctx}\n\n[질문]\n{user_query}\n\n규칙: 컨텍스트 근거 없는 내용은 쓰지 말고, 답 뒤에 (출처: [번호]) 인용표기."}
        ]

        # 채팅 템플릿 적용
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 토크나이징
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 스트리밍 설정
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024,  # 512 → 1024로 늘림
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

        # 생성 실행
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 결과 수집
        output = []
        for token in streamer:
            output.append(token)

        thread.join()
        return "".join(output).strip()

def use_ollama() -> bool:
    """Ollama 사용 여부 확인"""
    return os.getenv("OLLAMA_MODEL") is not None

def chat_ollama(system: str, user: str, context_snippets: List[Dict]) -> str:
    """Ollama API 호출"""
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL")
    assert model, "Set OLLAMA_MODEL to use Ollama."

    ctx = "\n\n".join([f"[{i+1}] {s['title']} — {s.get('url','')}\n{s['text']}" for i, s in enumerate(context_snippets)])
    prompt = f"{system}\n\n[컨텍스트]\n{ctx}\n\n[질문]\n{user}\n\n위 컨텍스트만 근거로 한국어로 답하세요. 답 뒤에 (출처: [번호]) 형태로 인용표기."

    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}
    r = requests.post(f"{url}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["response"]

# 글로벌 인스턴스
_multitarget_llm = None

def get_multitarget_llm() -> MultiTargetLLM:
    """멀티타겟 LLM 싱글톤 인스턴스 반환"""
    global _multitarget_llm
    if _multitarget_llm is None:
        _multitarget_llm = MultiTargetLLM()
        _multitarget_llm.load_all_models()
    return _multitarget_llm

def chat(system: str, user: str, context_snippets: List[Dict], target_type: str = "general") -> str:
    """메인 채팅 함수 - 타겟 타입에 따라 적절한 모델 사용"""
    if use_ollama():
        return chat_ollama(system, user, context_snippets)

    # 멀티타겟 LLM 사용
    llm = get_multitarget_llm()
    return llm.chat_with_model(target_type, user, context_snippets)

# 기존 함수 호환성 유지
def chat_hf(system: str, user: str, context_snippets: List[Dict]) -> str:
    """기존 HuggingFace 채팅 함수 (호환성용)"""
    return chat(system, user, context_snippets, target_type="general")