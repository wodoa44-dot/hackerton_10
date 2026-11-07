# ERP 원재료 품목코드 모델

품목명·규격명을 입력으로 받아 **분류 / 특성 / 세분류·품번 / 형상 / Packing / Vendor**를
각각 출력하고, 품목별 매뉴얼 규칙에 따라 완성 코드와 후보 스펙트럼을 만듭니다.

- 공통 전처리와 XGBoost·LightGBM·Forest, 입력 후보 기반 품번 랭커
- 매뉴얼 코드표 검증·코드 조립, 칸별 결과, Top-K 평가와 그래프
- CSV의 빈 품목코드 채우기와 원본 열 보존
- 공통 인코더와 6개 출력 헤드의 학습·추론·평가 코드

## 현재 상태

이 저장소는 채팅에 남은 작성·수정·읽기 기록에서 복구한 소스 백업입니다.
회사 규칙표와 CSV의 해시는 이전 기록과 일치하며, 소스 검증 36개가 통과했습니다.
**학습 가중치는 복구되지 않았고, 인코더 학습도 아직 완료되지 않았습니다.**
이전 모델을 이용한 성능 평가와 실제 웹 예측은 재학습 이후 다시 검증해야 합니다.

[복구 범위와 한계](RECOVERY.md), [파일별 복구 기록](RECOVERY_MANIFEST.json),
[소스·데이터 검증](recovery/source_data_verification.json),
[테스트 결과](recovery/source_tests.json)를 확인할 수 있습니다.

## 시작

```sh
git clone https://github.com/wodoa44-dot/hackerton_10.git
cd hackerton_10
uv venv --python python3 .venv
UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 uv pip install --python .venv/bin/python -r requirements.txt
```

코드 조립에 쓰는 Node.js도 실행 경로에 있어야 합니다.

기본 학습·보정 과정은 다음과 같습니다. 학습 입력은 training CSV이며,
inference CSV의 정답은 모델 학습에 사용하지 않습니다.

```sh
ERP_XGB_DEVICE=cuda .venv/bin/python -m erp_model.train_generation
ERP_XGB_DEVICE=cuda .venv/bin/python -m erp_model.train_accuracy
ERP_XGB_DEVICE=cuda .venv/bin/python -m erp_model.train_attributes
.venv/bin/python -m erp_model.validate_accuracy
.venv/bin/python deploy/assemble_generation_v6.py
.venv/bin/python -m erp_model.training_audit
```

인코더 학습에는 `requirements-encoder.txt`와 GPU에 맞는 PyTorch가 필요합니다.
위 과정에서 만든 학습 라벨 감사 결과를 읽어 학습합니다.

```sh
.venv/bin/python -m erp_model.train_encoder
.venv/bin/python -m erp_model.evaluate_encoder
```

`encoder_config.json`과 `completion.json`이 생성됐는지 확인해야 실제 학습 완료입니다.
모델 가중치는 이 소스 저장소에 포함하지 않습니다.

과거 전체 사용 설명은 [복구된 설명서](docs/README_chat_recovered.md)에 보존했습니다.
