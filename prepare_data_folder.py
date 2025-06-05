import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def convert_to_bbdm_structure_with_split(src_root, dst_root, val_ratio=0.1, test_ratio=0.1):
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

    random.seed(42) # 시드 고정
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

    for split_name, split_pairs in split_dict.items():
        a_dir = Path(dst_root) / split_name / "NECT"
        b_dir = Path(dst_root) / split_name / "CECT"
        a_dir.mkdir(parents=True, exist_ok=True)
        b_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for nect_file, cect_file in split_pairs:
            dst_nect = a_dir / nect_file.name
            dst_cect = b_dir / cect_file.name

            if dst_nect.exists() and dst_cect.exists():
                continue  # 이미 존재하는 경우 스킵

            shutil.copy(nect_file, dst_nect)
            shutil.copy(cect_file, dst_cect)
            copied += 1

        print(f" {split_name.upper()} 세트: {len(split_pairs)} 쌍 중 {copied} 쌍 복사 완료")

    print(f"\n 변환 완료: {dst_root} 에 train/val/test 구조로 저장되었습니다.")

# 실행 예시
if __name__ == "__main__":
    # 데이터가 저장된 위치
    src_root = "/mnt/c/Users/0106y/PSHC/Bucheon_CMC_DR/Chest"
    # 새롭게 데이터를 저장할 위치
    dst_root = "/mnt/c/Users/0106y/PSHC/BBDM_input_split"
    convert_to_bbdm_structure_with_split(src_root, dst_root)
