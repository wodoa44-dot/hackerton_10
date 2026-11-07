# ERP 원재료 품목코드 모델

품목명·규격명을 입력으로 받아 **분류 / 특성 / 세분류·품번 / 형상 / Packing / Vendor**를
각각 출력하고, 품목별 매뉴얼 규칙에 따라 완성 코드와 후보 스펙트럼을 만듭니다.

- 공통 전처리와 XGBoost·LightGBM·Forest, 입력 후보 기반 품번 랭커
- 매뉴얼 코드표 검증·코드 조립, 칸별 결과, Top-K 평가와 그래프
- CSV의 빈 품목코드 채우기와 원본 열 보존
- 공통 인코더와 6개 출력 헤드의 학습·추론·평가 코드

## 현재 상태 · 2026-11-07 재학습

복구한 CSV·매뉴얼로 두 방식을 실제 재학습하고 같은 inference 2,850행에서 비교했습니다.
인코더는 **Transformer 없는 문자 CNN, 402,003 파라미터, 256차원**입니다.
품목명 128바이트·규격명 384바이트를 별도로 인코딩하고 다섯 분류 보조 헤드와 입력 품번 후보 랭커로 표현을 학습합니다.
학습 그룹 검증으로 24 epoch를 선택한 뒤 전체 유지 학습 행으로 새로 학습했으며 GPU 학습은 약 95초 걸렸습니다.

운영 파이프라인은 두 방식 모두 XGBoost·LightGBM·Forest와 입력 후보 품번 랭커입니다.
인코더 사용 방식은 기존 TF-IDF·구조화 특징에 CNN 벡터만 추가합니다.
칸별 모델·가중치·트리 설정·학습 라벨·공통 규칙·조회·코드 조합은 고정했습니다.
인코더를 거친 전체 학습 행으로 트리를 다시 학습하므로, 이 방식에 별도 OOF 성능이나 보정된 신뢰도를 주장하지 않습니다.

| 전체 완성 코드 | 인코더 없음 | 학습 인코더 사용 |
|---|---:|---:|
| Top-1 | 275 / 2,850 · 9.65% | 277 / 2,850 · 9.72% |
| Top-3 | 310 / 2,850 · 10.88% | 310 / 2,850 · 10.88% |
| Top-5 | 332 / 2,850 · 11.65% | 332 / 2,850 · 11.65% |
| Top-10 | 349 / 2,850 · 12.25% | 350 / 2,850 · 12.28% |

현재 최종 코드 성능에서 인코더의 효과는 작습니다. 기본 운영 모델은 인코더 없는 방식이며 웹 단일 예측에서 두 방식을 선택할 수 있습니다.
웹의 모델 평가 화면은 여섯 칸마다 별도 패널·K=1~10 그래프·Top-1·3·5·10 표·개별 K 선택·CSV를 제공합니다.
기본 화면은 각 칸 모델의 독립 후보이며, 규칙을 반영한 칸별 후보와 완성 코드에서 해석한 칸의 점수를 구분해 전환할 수 있습니다.
완성 코드 전체의 조합 평가는 아래의 별도 펼침 영역에 보존합니다.
완성 코드는 전체 2,850행, 상세 칸은 매뉴얼로 의미를 확인할 수 있는 해당 행만 채점합니다.
매뉴얼 해석 가능 완성 코드 행은 473행입니다. 해당 범위와 전체 범위는 그래프에서 구분합니다.

학습 CSV 11,399행에서 inference와 입력이 겹치는 563행과 비코드 2행을 제외한 10,834행을 유지했습니다.
같은 입력의 상충 정답·품목군/코드 체계 불일치·수치 충돌은 칸별 라벨 마스크로 처리하며 원본 CSV를 바꾸지 않았습니다.
inference 정답은 학습·epoch 선택·트리 선택에 쓰지 않았습니다. 이 CSV는 이전에 살펴본 진단 데이터이므로 처음 보는 블라인드 시험으로 간주하지 않습니다.

[실제 비교 수치](artifacts/encoder_comparison/comparison.json),
[K별 수치](artifacts/encoder_comparison/top_k_comparison.csv),
[행별 두 결과](artifacts/encoder_comparison/paired_predictions.csv),
[학습 라벨 감사](artifacts/encoder_training_data/learning_data_audit.json),
[행별 결과 재검산 120항목](recovery/encoder_results_verified.json)을 확인할 수 있습니다.

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

동일 비교를 재현하려면 기존 출력 모델이 없는 새 작업 폴더에서 아래 과정을 실행합니다.
공통 전처리·라벨 감사 뒤 인코더와 기준 모델을 학습하고 동일 조건의 트리 모델을 다시 맞춥니다.

```sh
.venv/bin/python -m erp_model.prepare_encoder_data
.venv/bin/python -m erp_model.train_encoder
ERP_XGB_DEVICE=cuda .venv/bin/python -m erp_model.train_comparison_baseline
ERP_XGB_DEVICE=cuda .venv/bin/python -m erp_model.train_encoder_augmented --device cuda
.venv/bin/python -m erp_model.compare_encoders
.venv/bin/python recovery/verify_encoder_results.py
```

인코더에는 `requirements-encoder.txt`와 GPU에 맞는 PyTorch가 필요합니다. Transformer나 사전학습 모델 다운로드는 사용하지 않습니다.
장시간 실행은 `deploy/run_encoder_comparison.py`가 단계별 상태·로그를 남깁니다. CPU 2스레드와 배치 64, 인코더 GPU 메모리 상한 8GiB를 기본으로 사용합니다.
XGBoost는 CUDA, LightGBM·Forest는 CPU로 학습했습니다. 이 실행의 환경과 버전은 저장된 manifest와 encoder_config.json에 기록되어 있습니다.

```sh
.venv/bin/python -m erp_model serve --models artifacts/run1 --port 8501 --root-path /i/OsAdz8rQR7JSXeW6
```

`artifacts/generation_v6`, `artifacts/generation_v7`와 `artifacts/encoder_v7`는 실제 학습 결과입니다.
인코더 유무 비교에 필요한 가중치와 평가 CSV도 보관합니다. `RECOVERY_MANIFEST.json`은 최초 복구 시점의 기록이며 이후 의도한 변경에 대한 현재 해시 목록은 아닙니다.

과거 전체 사용 설명은 [복구된 설명서](docs/README_chat_recovered.md)에 보존했습니다.
