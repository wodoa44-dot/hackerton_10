# 프로젝트 복구 상태

이 채팅의 파일 작성·수정·읽기 기록과 로컬에 남은 파일에서 복구했습니다.

- 공통 전처리, 6개 예측 칸, XGBoost/LightGBM/Forest 학습, 품번 랭커,
  속성 보정, 매뉴얼 코드 조립, 후보 스펙트럼, Top-K 평가를 포함합니다.
- 웹 UI, CSV 빈 코드 채우기, 칸별 결과, K 그래프, 평가·테스트 코드를 포함합니다.
- 공통 인코더와 6개 출력 헤드의 학습·추론·평가 코드도 포함합니다.
- 원재료 전체·training·inference CSV와 매뉴얼 PDF를 원본 그대로 복사했습니다.
- `config/company_rules.json`은 이전 모델에 기록된 SHA-256과 일치합니다.

## 복구 한계

학습된 `.joblib`, `.pt`, `.safetensors` 가중치는 채팅에 바이너리 내용이
없어 복구하지 못했습니다. 현재 소스 복구는 학습 완료를 의미하지 않습니다.
과거 성적·검증 기록은 `recovery/historical_metadata`에 분리했습니다.

`train.py`는 기록에 남은 공통 함수와 현재 학습 진입점을 결합했습니다.
`train_material.py`는 현재 복구한 코드에서 필요한 호환 함수만 재작성했습니다.
옛 진단·리포트 스크립트 중 원문이 없는 항목은 `RECOVERY_MANIFEST.json`에 명시했습니다.

## 실행 환경

```sh
uv venv --python python3 .venv
UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 uv pip install --python .venv/bin/python -r requirements.txt
```

현재 회사 규칙 기반 학습은 `python -m erp_model.train_generation`, 인코더
학습은 `python -m erp_model.train_encoder`입니다. 인코더에는 PyTorch CUDA
설치와 training audit 결과가 추가로 필요합니다. 복구 검증과 학습 결과는
서로 구분해서 확인해야 합니다.
