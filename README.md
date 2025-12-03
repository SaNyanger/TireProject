# TireProject

차량 이미지에서 타이어를 검출하고 트레드(접지면)를 추출하여 마모 상태를 분석하는 파이프라인입니다.

---

## 주요 기능

- 차량 이미지에서 타이어 위치 자동 검출 (타원/허프 서클)
- 타이어 트레드 이미지 언랩 (펼치기)
- 트레드 마모 상태 분류 (danger / warning / safety)
- CAM 히트맵 시각화

---

## 파이프라인 흐름
```
차량 이미지 입력
    ↓
전처리 (Resize, CLAHE)
    ↓
타이어 후보 검출 (Ellipse, Hough Circle, NMS)
    ↓
RCNN 키포인트 검출 & 트레드 언랩
    ↓
Best 트레드 선택
    ↓
마모 상태 분류 (CNN)
    ↓
JSON 결과 출력
```

---

## 설치 방법

### 1. Git LFS 설치 (모델 파일 다운로드용)
```bash
git lfs install
```

### 2. 저장소 클론
```bash
git clone [저장소 URL]
cd TireProject
git lfs pull
```

### 3. 가상환경 생성 (권장)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4. 의존성 설치
```bash
pip install -r requirements.txt
```

---

## 실행 방법

### 단독 실행 (테스트용)
```bash
cd app
python tire_rcnn_pipeline.py
```

- `assets/` 폴더의 이미지를 분석하여 결과 출력
- `IMG_NAME` 변수에서 입력 이미지 파일명 변경 가능

### FastAPI 서버 연동

이 코드는 FastAPI 서버에서 `analyze_car_image()` 함수를 호출하여 사용합니다.
```python
from tire_rcnn_pipeline import analyze_car_image

result = analyze_car_image("이미지_경로.jpg")
# 결과: {"result": {"predict_result": "safety", "img_url": "cam_결과_경로.png"}}
```

---

## 프로젝트 구조
```
TireProject/
├── app/
│   ├── tire_rcnn_pipeline.py    # 메인 파이프라인
│   ├── tire_classifier.py       # 마모 분류 모델
│   ├── model/
│   │   └── model.py             # ResNet 모델 정의
│   ├── models/
│   │   ├── RCNN/
│   │   │   └── saved_model.pth  # RCNN 키포인트 모델
│   │   └── model_best_weights.pth  # 분류 모델
│   ├── utils.py                 # 유틸리티 함수
│   ├── assets/                  # 입력 이미지 폴더
│   └── outputs/                 # 출력 결과 폴더
├── requirements.txt
└── README.md
```

---

## 설정 파라미터

`tire_rcnn_pipeline.py` 상단에서 조정 가능:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `MAX_SIDE` | 1920 | 이미지 긴 변 최대 크기 (초과 시 축소) |
| `MIN_SIDE` | 800 | 이미지 긴 변 최소 크기 (미만 시 확대) |
| `USE_CUDA` | False | GPU 사용 여부 (True로 변경 시 GPU 사용) |
| `CROP_PAD` | 1.75 | 후보 박스 여유 배수 |
| `UPSCALE_BEFORE_UNWRAP` | 1.8 | RCNN 입력 전 업스케일 배수 |

---

## 출력 결과

### JSON 응답 형식
```json
{
  "result": {
    "predict_result": "safety",
    "img_url": "/path/to/cam_result.png"
  }
}
```

| 필드 | 설명 |
|------|------|
| `predict_result` | 분류 결과 (danger / warning / safety / null) |
| `img_url` | CAM 히트맵 이미지 경로 (실패 시 null) |

### 출력 파일

- `app/outputs/debug_candidates.jpg` - 후보 박스 시각화
- `app/outputs/tread_*.png` - 언랩된 트레드 이미지
- `app/outputs/results_summary.csv` - 결과 요약

---

## 요구사항

- Python 3.10
- CUDA (선택사항, GPU 사용 시)

---

## 라이선스

이 프로젝트는 학술 목적으로 개발되었습니다.