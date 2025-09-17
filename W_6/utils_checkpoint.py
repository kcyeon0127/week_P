import os
import glob

def find_latest_checkpoint(model_dir: str):
    """최신 체크포인트 경로 자동 찾기"""
    checkpoint_dirs = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        # 체크포인트가 없으면 모델 디렉토리 자체 반환
        return model_dir

    # 숫자 기준으로 정렬해서 가장 큰 번호 반환
    checkpoint_nums = []
    for cp_dir in checkpoint_dirs:
        try:
            num = int(os.path.basename(cp_dir).split("-")[1])
            checkpoint_nums.append((num, cp_dir))
        except (IndexError, ValueError):
            continue

    if checkpoint_nums:
        latest = max(checkpoint_nums, key=lambda x: x[0])
        print(f"최신 체크포인트 사용: {latest[1]}")
        return latest[1]

    return model_dir