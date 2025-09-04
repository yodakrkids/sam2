# SAM-2 실행 방법

이 문서는 `SAM-2` 프로젝트를 설치하고 실행하는 방법을 안내합니다. 프로젝트는 두 가지 주요 방법으로 실행할 수 있습니다: Docker를 사용한 전체 웹 데모 또는 Python 라이브러리.

## 1. 웹 데모 실행 (Docker)

Docker Compose를 사용하여 프론트엔드와 백엔드 서비스를 포함한 전체 웹 데모를 실행하는 것을 권장합니다.

### 사전 요구사항
*   Docker 및 Docker Compose
*   NVIDIA GPU 및 NVIDIA Container Toolkit

### 실행 방법
1.  **Docker 서비스 빌드 및 실행**
    프로젝트 루트 디렉토리에서 다음 명령어를 실행하세요.
    ```bash
    docker-compose up --build
    ```
    *백그라운드에서 실행하려면 `-d` 플래그를 추가하세요: `docker-compose up --build -d`*

2.  **데모 접속**
    빌드가 완료되고 서비스가 실행되면 웹 브라우저에서 아래 주소로 접속할 수 있습니다.
    *   **프론트엔드:** `http://localhost:7262`
    *   **백엔드 API:** `http://localhost:7263`

## 2. Python 라이브러리로 사용

핵심 모델을 Python 라이브러리로 직접 사용할 수도 있습니다.

### 사전 요구사항
*   Python ≥ 3.10
*   PyTorch ≥ 2.5.1
*   CUDA Toolkit (PyTorch와 호환되는 버전)
> Windows 사용자는 WSL 사용을 강력히 권장합니다.

### 설치 및 준비
1.  **저장소 복제 (아직 안했다면)**
    ```bash
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    ```

2.  **모델 체크포인트 다운로드**
    ```bash
    ./checkpoints/download_ckpts.sh
    ```

3.  **의존성 설치**
    ```bash
    # 노트북 예제를 포함하여 설치
    pip install -e ".[notebooks]"
    ```
    > **참고**: 설치 관련 상세 내용 및 문제 해결은 `INSTALL.md`를 참고하세요.

### 실행 예제: 이미지 예측
```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 모델 설정 및 로드
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# 이미지(<your_image>)와 프롬프트(<input_prompts>)를 설정하여 예측 실행
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)

# 자세한 내용은 notebooks/image_predictor_example.ipynb 를 참고하세요.
```