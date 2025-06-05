# BBDM 소개

* **논문:** BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models (CVPR 2023)

  * [CVPR 2023 논문 링크](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf)
  * [공식 GitHub 저장소](https://github.com/xuekt98/BBDM/tree/main)

* **Task:** 이미지 간 변환 (Source → Target)

* **기반 모델:** Diffusion Models

## 주요 특징

* Brownian Bridge diffusion 과정을 통해 두 확률 분포 사이의 경로를 학습

  * Forward 과정: Target 이미지에 노이즈를 추가하여 Source로 이동
  * Reverse 과정: Source에서 노이즈를 제거하며 Target 복원 학습
* **Paired 데이터** 기반으로 학습
* **Pixel-space** 및 **Latent-space** 모두 적용 가능

  * Latent-space의 경우 VQGAN의 latent 표현 사용

## 구현 상세

* **현재 구현:** Pixel-space 기반 버전만 포함

  * 원 저자 코드에는 Pixel/Latent-space 모두 포함

## 📁 프로젝트 구조 요약

### `main.py`

* 실행 entry point
* `config.yaml`을 불러와 환경 설정 (CPU / GPU / DDP) 및 학습 or 추론 수행

### `prepare_data_folder.py`

* 전처리된 이미지에서 학습용 폴더 구조 생성
* 생성되는 폴더 구조:

  * `train/`, `val/`, `test/` (비율: 0.8 / 0.1 / 0.1)

### `Register.py`

* 다양한 runner들을 설정 파일 기반으로 등록하는 유틸리티

### `utils.py`

* `DictConfig` ⇄ `Namespace` 변환
* 문자열 경로를 통해 Python 객체 동적 생성
* 설정된 `Runner` 인스턴스 반환

## 📁 `configs`

* `Template-BBDM.yaml`: Pixel-space BBDM 설정 파일
* 기타 YAML 파일: Latent-space 설정 포함

## 📁 `datasets`

### `custom.py`

* NECT → CECT 이미지 쌍을 로드하여 학습용 데이터셋 생성

* **필요한 폴더 구조:**

  ```
  dataset_path/
    train/
      NECT/
        S0001_NECT_0001.png
        ...
      CECT/
        S0001_CECT_0001.png
        ...
    val/
      NECT/
      CECT/
    test/
      NECT/
      CECT/
  ```

* 해당 구조는 `prepare_data_folder.py` 스크립트를 통해 자동 생성 가능

### `base.py`

* 이미지 크기 조정 및 flip을 통한 데이터 증강 수행
* \[-1, 1] 범위로 정규화된 텐서 반환

### `utils.py`

* 폴더 내 모든 이미지 경로를 재귀적으로 탐색하여 리스트 반환

## 📁 `model`

### `utils.py`

* `extract()`, `exists()`, `default()` 등: 스케줄 추출, 기본값 설정에 사용되는 유틸 함수

### `BrownianBridgeModel.py`

* Diffusion process의 핵심 로직 구현 (Forward, Reverse, Sampling, Training)

### `base/utils.py`

* 로깅, 이미지 체크, 파라미터 개수 세기, 동적 객체 생성 등 지원 기능 포함

### `base/modules`

* Attention, Transformer 모듈 정의

### `base/modules/diffusionmodules`

* UNet 기반 구조 구현 (예: `openaimodel.py`)

## 📁 `runners`

### `DiffusionBasedModel`

* `BBDMRunner.py`: BBDM 모델 학습 및 테스트 전용 runner
* `DiffusionBaseRunner.py`: Diffusion 기반 runner 공통 base class

### `base`

* `EMA.py`: Exponential Moving Average 적용을 위한 모듈
* `BaseRunner.py`, `utils.py`: 공통 학습 기능 및 보조 유틸리티 포함
