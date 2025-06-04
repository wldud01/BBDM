import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def convert_to_bbdm_structure_with_split(src_root, dst_root, val_ratio=0.1, test_ratio=0.1):
    # 쌍이 맞는 파일 리스트 구성
    pairs = []

    patient_dirs = sorted(Path(src_root).glob("S*/1.preprocess/png"))
    for patient_dir in tqdm(patient_dirs, desc="Searching patient folders"):
        nect_dir = patient_dir / "NECT"
        cect_dir = patient_dir / "CECT"

        if not nect_dir.exists() or not cect_dir.exists():
            continue

        for cect_file in sorted(cect_dir.glob("*.png")):
            file_idx = cect_file.stem.split("_")[-1]
            patient_id = cect_file.stem.split("_")[0]
            nect_filename = f"{patient_id}_NECT_{file_idx}.png"
            nect_file = nect_dir / nect_filename

            if nect_file.exists():
                pairs.append((nect_file, cect_file))

    print(f" 총 {len(pairs)} 쌍의 이미지가 발견되었습니다.")

    # 셔플 후 split
    random.seed(42) 
    random.shuffle(pairs)
    n_total = len(pairs)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    split_dict = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:]
    }

    # 복사
    for split_name, split_pairs in split_dict.items():
        a_dir = Path(dst_root) / split_name / "NECT"
        b_dir = Path(dst_root) / split_name / "CECT"
        a_dir.mkdir(parents=True, exist_ok=True)
        b_dir.mkdir(parents=True, exist_ok=True)

        for nect_file, cect_file in split_pairs:
            shutil.copy(nect_file, a_dir / nect_file.name)
            shutil.copy(cect_file, b_dir / cect_file.name)

        print(f" {split_name.upper()} 세트: {len(split_pairs)} 쌍 복사 완료")

    print(f"\n 변환 완료: {dst_root} 에 train/val/test 구조로 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    src_root = "/mnt/c/Users/0106y/PSHC/Bucheon_CMC_DR/Chest"
    dst_root = "/mnt/c/Users/0106y/PSHC/BBDM_input_split"
    convert_to_bbdm_structure_with_split(src_root, dst_root)
