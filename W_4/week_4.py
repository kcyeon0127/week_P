
# 필요한 라이브러리들을 가져옵니다.
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify_sentence(text):
    """
    사전 학습된 BERT 모델을 사용하여 문장을 분류합니다.
    참고: 이 모델은 AG_NEWS 데이터셋에 파인튜닝되지 않은 'bert-base-uncased' 모델입니다.
          따라서 예측 결과는 무작위에 가깝습니다.
    """
    # 사용할 모델의 이름을 지정합니다.
    model_name = "bert-base-uncased"
    
    # 지정된 모델 이름으로 토크나이저를 불러옵니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # AG_NEWS 데이터셋은 4개의 레이블을 가지므로, 모델의 출력 레이블 수를 4로 설정합니다.
    # 'bert-base-uncased'의 사전 학습된 가중치는 분류 헤드(classification head)를 포함하지 않으므로,
    # Transformers 라이브러리가 무작위 가중치로 새로운 분류 헤드를 초기화합니다.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # 입력된 텍스트를 토크나이징합니다. PyTorch 텐서로 반환하도록 설정합니다.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 모델 추론을 수행합니다. 기울기 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높입니다.
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 로짓(logits) 값 중 가장 높은 값의 인덱스를 가져와 예측된 클래스 ID로 사용합니다.
    predicted_class_id = torch.argmax(logits, dim=1).item()

    # 예측된 ID를 실제 레이블 이름으로 변환하기 위해 AG_NEWS 데이터셋 정보를 로드합니다.
    dataset = load_dataset("ag_news")
    labels = dataset["test"].features["label"].names
    predicted_label = labels[predicted_class_id]

    # 예측된 레이블과 신뢰도 점수(softmax 확률)를 반환합니다.
    return predicted_label, torch.softmax(logits, dim=1).max().item()

# 이 스크립트가 직접 실행될 때만 아래 코드를 실행합니다.
if __name__ == "__main__":
    # AG_NEWS 데이터셋을 로드합니다.
    dataset = load_dataset("ag_news")

    # 테스트 데이터셋에서 샘플을 하나 가져옵니다. (예: 10번째 샘플)
    sample_index = 10
    sample = dataset["test"][sample_index]
    text_to_classify = sample["text"]
    actual_label_id = sample["label"]
    
    # 실제 레이블 ID를 문자열 레이블로 변환합니다.
    actual_label = dataset["test"].features["label"].int2str(actual_label_id)

    print(f"--- 사전 학습된 BERT를 이용한 AG NEWS 분류 ---")
    print(f"분류할 텍스트: \"{text_to_classify}\"")
    print(f"실제 레이블: {actual_label}")

    # 정의된 함수를 호출하여 문장을 분류합니다.
    predicted_label, confidence = classify_sentence(text_to_classify)

    print(f"예측된 레이블: {predicted_label}")
    print(f"신뢰도: {confidence:.4f}")
    print("\n참고: 이 모델은 AG_NEWS에 파인튜닝되지 않았습니다.")
    print("분류 헤드가 무작위로 초기화되었기 때문에, 예측은 사실상 무작위입니다.")
    print("이 스크립트는 사전 학습 모델을 추론에 사용하는 파이프라인을 보여주기 위함입니다.")
