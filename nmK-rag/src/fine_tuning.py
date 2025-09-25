import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MultiTargetTrainer:
    def __init__(self, base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """베이스 모델과 토크나이저 로드"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def setup_lora_config(self, target_type: str = "general"):
        """LoRA 설정"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, data_path: str, target_type: str = "general"):
        """데이터셋 준비"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        response_key = f"{target_type}_response"

        for item in data:
            prompt = self._format_prompt(item["question"], target_type)
            response = item[response_key]

            # 챗 템플릿 적용
            messages = [
                {"role": "system", "content": self._get_system_prompt(target_type)},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

        # 토크나이즈
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        # labels를 input_ids로 설정 (causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()

        dataset = Dataset.from_dict(tokenized)
        return dataset

    def _get_system_prompt(self, target_type: str) -> str:
        """타겟별 시스템 프롬프트"""
        if target_type == "children":
            return """너는 어린이를 위한 친절한 박물관 안내봇이야.
어려운 용어는 쉽게 설명하고, 재미있고 이해하기 쉽게 대답해줘.
이모지를 적절히 사용해서 더 친근하게 대화해줘."""
        else:
            return """너는 국립중앙박물관 전문 안내 도슨트이다.
정확하고 전문적인 정보를 제공하며, 교육적 가치가 있는 설명을 해줘."""

    def _format_prompt(self, question: str, target_type: str) -> str:
        """타겟별 프롬프트 포맷"""
        if target_type == "children":
            return f"어린이가 궁금해하는 질문이에요: {question}"
        else:
            return question

    def train_model(self, dataset: Dataset, output_dir: str, target_type: str):
        """모델 훈련"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            logging_steps=10,
            logging_dir=f"{output_dir}/logs",
            save_strategy="epoch",
            evaluation_strategy="no",
            load_best_model_at_end=False,
            dataloader_pin_memory=False,
            fp16=True,
            report_to=None,
            run_name=f"multitarget-{target_type}",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # 훈련 실행
        trainer.train()

        # 모델 저장
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"모델 훈련 완료 및 저장: {output_dir}")

def train_multitarget_models(data_path: str, base_output_dir: str = "models"):
    """일반용과 어린이용 모델 모두 훈련"""
    os.makedirs(base_output_dir, exist_ok=True)

    for target_type in ["general", "children"]:
        print(f"\n=== {target_type.upper()} 모델 훈련 시작 ===")

        trainer = MultiTargetTrainer()
        trainer.load_model()
        trainer.setup_lora_config(target_type)

        dataset = trainer.prepare_dataset(data_path, target_type)
        output_dir = os.path.join(base_output_dir, f"lora-{target_type}")

        trainer.train_model(dataset, output_dir, target_type)

        print(f"=== {target_type.upper()} 모델 훈련 완료 ===\n")

        # GPU 메모리 정리
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    data_path = "data/training_data.json"
    train_multitarget_models(data_path)