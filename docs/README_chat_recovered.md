# ERP 원재료 품목코드 생성

품목명과 규격명을 공통 전처리하고 분류, 특성, 세분류·품번, 형상, Packing, Vendor를 각각 반환합니다. 여섯 결과는 품목별 실제 위치에 배치하여 코드를 조립합니다. 기본 배포 모델은 매뉴얼·CSV 대조를 반영한 generation_v5입니다.

웹: https://devopen-gpu.jntp-ax.or.kr/i/OsAdz8rQR7JSXeW6/

## 매뉴얼과 CSV 대조

- 근거: 품목코드메뉴얼(정신전자) PDF 30쪽. 원본 SHA-256, 페이지별 근거, 문서의 내부 불일치와 처리 이유는 config/manual_source.json에 기록했습니다.
- 상세 학습 라벨은 코드표의 모든 값, 고정 구분자, 분해·재조립 일치를 통과한 행에서만 만듭니다. 원본 CSV는 변경하지 않습니다.
- JSD 등 고객사/기존 관리 코드를 J 상품의 생산지·상태로 분해하지 않습니다. 구간 의미를 확인할 수 없는 코드는 등록 원문과 출처를 표시합니다.
- 써미스터는 품번 7자리, 12번째 Packing, 13번째 제조사·버전으로 분리합니다.
- 퓨즈는 중분류 뒤 세분류 8자리와 버전 1자리입니다. 바리스터는 형상·세분류·Packing·Vendor가 있으며 CSV로 확인한 별도 배치를 사용합니다.
- 전해콘덴서의 pF/uF 혼용은 학습 참조 데이터의 근거 건수로 정한 품목별 정책을 저장합니다. 매뉴얼 uF 표기와 다른 기본값을 쓸 때 화면에 표시합니다.
- 매뉴얼 21쪽 S/C, 25쪽 17/18, 27쪽 19/18의 불일치를 기록했습니다. 반도체의 세부 배치와 바리스터의 세분류 폭은 실사용 확장임을 구분합니다.

대조 결과는 artifacts/generation_v5/MANUAL_COMPARISON.md, 행별 대조는 manual_comparison.csv, 집계는 manual_comparison.json입니다. 매뉴얼 규칙에 맞지 않는다는 이유로 원래 코드를 잘못된 코드로 단정하지 않습니다.

## 설치

Python 3.12, Node.js, LightGBM 실행용 libgomp1이 필요합니다. 기존 서버에는 .venv와 Node.js가 설치되어 있습니다.

~~~bash
sudo apt-get install libgomp1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
~~~

## GPU 학습과 평가

학습·검증은 training CSV의 입력 그룹을 공통 3폴드로 나눕니다. evaluation CSV와 품목명·규격명 키가 겹치는 학습 행 및 동일 입력의 상충 라벨은 제외합니다. 코드 길이와 하이픈은 문자열로 보존합니다. XGBoost는 CUDA, LightGBM과 Random Forest는 CPU를 사용합니다.

~~~bash
ERP_XGB_DEVICE=cuda:0 ERP_TRAIN_THREADS=16 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -m erp_model train \
  --train dataset/ERP_원재료_training.csv \
  --evaluation dataset/ERP_원재료_inference.csv \
  --out artifacts/generation_v5

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python -m erp_model.evaluate_generation --models artifacts/generation_v5
~~~

CPU 학습은 ERP_XGB_DEVICE=cpu로 실행합니다. 전체 CSV를 검증하려면 train 명령에 --full 경로를 추가합니다. XGBoost·LightGBM 앙상블을 비교하고, 분류와 Packing에는 Forest도 비교합니다. 선택은 학습 OOF Macro F1과 정확도로 결정합니다. 품번은 입력에서 만든 후보에 대한 두 랭커의 평균 순위입니다. 수치 용량·저항값은 규칙으로 계산합니다.

각 모델과 함께 코드표 스냅샷을 저장하고 해시를 검증합니다. 상세 슬롯은 다른 슬롯의 예측값을 피처로 사용하지 않습니다.

## 독립 추론

~~~bash
.venv/bin/python -m erp_model predict --models artifacts/generation_v5 \
  --name 'RES CHIP' --spec '1608 F 12.3K, REEL, YAGEO'

.venv/bin/python -m erp_model predict --models artifacts/generation_v5 \
  --slot packing --name 'THERMISTOR NTC' --spec 'NTX9234 BULK'
~~~

전체 실행은 context, 여섯 슬롯, assembly의 8개 JSONL 이벤트입니다. 단일 슬롯은 해당 슬롯 이벤트 1개와 context만 반환합니다. 공통 입력 정규화와 텍스트 변환은 한 번 수행하며, 각 슬롯 오류는 다른 슬롯의 반환을 막지 않습니다.

## 웹 실행

~~~bash
.venv/bin/python deploy/manage_web.py start
.venv/bin/python deploy/manage_web.py status
.venv/bin/python deploy/manage_web.py stop
~~~

배포는 artifacts/active_generation.json이 지정한 완성된 모델을 사용합니다. 명시적으로 --models 경로를 전달하면 해당 폴더를 사용합니다. 현재 플랫폼은 8501 포트를 /i/OsAdz8rQR7JSXeW6/ 경로로 연결합니다. 로그는 artifacts/web.log입니다. 컨테이너 재시작 후에는 관리 명령으로 다시 실행해야 합니다.

## 결과와 평가 의미

- registry_exact: 등록된 품목명·규격명과 정규화 후 일치하는 코드입니다.
- registry_candidate: 규격명과 품목군이 일치하는 등록 후보입니다. 품목명 차이를 검토합니다.
- reference_candidate: 학습 참조의 유사 코드입니다. 신규 발급 코드로 자동 확정하지 않습니다.
- model_assembled: 여섯 슬롯 값을 코드표와 실제 배치에 맞춰 조립한 예측 코드입니다.

매뉴얼 외 등록 코드도 원문과 길이를 보존해 복사할 수 있습니다. 해당 구간 의미를 확인하지 못하면 각 슬롯에 해석 보류를 표시합니다. 규칙상 없는 구간은 해당 없음이며, ERP에 실제 존재하는 "-"와 구분합니다. 일련번호를 임의로 발급하지 않습니다.

운영 등록 마스터 조회와 모델 평가는 분리합니다. 평가는 겹치는 입력을 제외한 학습 데이터만 검색 참조로 사용하며 운영 마스터를 사용하지 않습니다. 완성 코드 일치율과 코드 후보 제공률은 서로 다른 지표입니다. evaluation 데이터는 이미 점검한 적이 있으므로 결과는 진단 평가입니다.

## 확인

~~~bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/browser_generation.py
~~~

테스트는 매뉴얼 코드표·실제 배치, 고객 코드 오분해 방지, 슬롯 독립 실행과 오류 격리, 신규 품번, 웹 표시·복사와 모바일 동작을 확인합니다. 구버전 다이오드 전용 실험은 --legacy-diode 옵션과 artifacts/run1에 남아 있습니다.


## v6: one prediction pipeline and a bounded code spectrum

The default web/CLI predictor is SpectrumPredictor when the active artifact manifest is v6.
There is one shared normalization/text feature state and six independent logical output heads.
The existing XGBoost/LightGBM/Forest heads, identifier rankers, and conditional field
estimators feed the same candidate assembly pipeline. Registered-master retrieval is an
evidence source, not a second ERP/manual trained model or a user-selectable prediction mode.

Each prediction still streams context, six slot results, and one assembly event.
A single-slot request returns only that slot and its context.
The assembly event adds spectrum (up to 10 unique full codes), per-head part_spectra,
and physical parts with original code positions. Every candidate carries its own six
values and parts. Selecting a candidate in the UI updates every displayed part and the
copied code together. Unknown legacy bodies are preserved but never given invented meanings.

Evaluation:
- Policies are selected on grouped training validation; the final CSV is diagnostic.
- The evaluation retrieval index contains retained training rows only; target-code overlap is 0.
- Original-string Exact@1/3/5/10 and MRR remain separate from numeric encoding equivalence.
- Part coverage in the first K joint codes is separate from independent-head marginal coverage.
- Input numeric consistency is not correctness against ERP labels. Missing/ambiguous constraints
  are unscored. CE/CM mixed capacitance conventions are not silently treated as equivalent.
- Candidate set size is reported alongside coverage. Scores are relative ordering scores,
  not calibrated correctness probabilities.

Main files: erp_model/spectrum.py, accuracy_runtime.py, attribute_models.py,
accuracy_candidates.py, code_metrics.py, evaluate_spectrum.py.
Training refinements: python -m erp_model.train_accuracy and
python -m erp_model.train_attributes. Training-policy validation:
python -m erp_model.validate_accuracy.
Evaluate the frozen v6 snapshot: python -m erp_model.evaluate_spectrum.
Run regressions: python -m unittest discover -s tests -p 'test*.py'.
Run browser verification: python tests/browser_spectrum.py.

CSV downloads from the evaluation page:
- original_code_parts.csv: all 14,249 original rows, verified six-part decoding or unmapped body.
- prediction_parts.csv: 2,850 held-out top-1 predictions, six slot values and detailed parts.
- spectrum_parts_top5.csv: one row per candidate among the first five, six values and detailed parts.

Operational master matches are available in production. Their convenient lookup results
must never be reported as held-out model accuracy.

## CSV upload and evaluation isolation (2026-11-07)

The active v6 web and CLI lookup now contain only the 10,834 retained training records.
Evaluation input-key overlap and evaluation-code overlap are both zero. The previous
14,249-record master is retained only as an unserved backup. Uploaded CSV rows never
update model weights or the lookup index.

The workspace has a CSV blank-code filling panel. POST /batch-csv accepts raw CSV bytes
(UTF-8/BOM or CP949), then GET /batch-csv/{id} reports progress and
GET /batch-csv/{id}/download returns the completed CSV. Nonempty codes, original columns,
leading zeros and row order are preserved. Blank codes receive the same top hypothesis
as the single-item predictor, with six logical values, detailed parts and review flags.
Unresolvable or invalid rows remain blank with reasons. Limit: 20 MB / 30,000 records.
The worker scores learned heads in chunks and caches duplicate inputs; existing code
values from the upload are never used to predict other rows.

Evaluation counts the selected full-code candidate's displayed fields, including unknown
fields as misses where truth is available. It no longer gives credit from an unused head
when the selected code has an unmapped field. The table shows label coverage, designated
(non-hyphen) accuracy, Macro F1 and a retained-training-majority baseline. These are final
pipeline results including rules, not standalone learned-model scores. The deprecated
D-only metric payload is no longer exposed as the current /model-info metrics field.

Verification: python -m unittest discover -s tests -p 'test*.py';
python tests/browser_csv_audit.py; python tests/browser_spectrum.py.

## Expanded unverified candidates and K coverage

Unverified layouts now have a separate exploratory result under the primary result.
It reuses the same frozen six learned heads and training-only reference. Whole-code
candidates combine the primary candidate, up to 40 similar training references, and
same-width input-identifier substitutions at spans actually observed in training codes.
When no reference matches, a training-frequency prior is explicitly labeled weak evidence.
No serial number is invented from the held-out answer. All exploratory applications remain
unverified, even when their strings happen to satisfy an existing code table.

Six exploratory head lists are reference estimates, filtered to family-observed labels
where available; they are never silently asserted to be the physical fields of a legacy code.
CSV output adds these candidates and estimates. A strict assembly failure can receive an
exploratory code with review required and the explicit unverified status.

candidate_k_metrics.json and candidate_k_curve.csv contain measured Hit@K for every K=1..10,
with fixed denominators: all, manual-parseable, and unmapped truth codes. Candidate selection
uses only prediction-side applicability, never the truth's group. effective_codes chooses the
expanded pool for an unverified prediction and the existing spectrum otherwise. The web chart
includes counts, average set size, series toggles and SVG/CSV export. Existing detailed slot
metrics and strict spectrum exports retain their own definitions and labels.

The current main heads select RF for classification/packing, XGBoost for trait/case/vendor,
and average XGBRanker/LGBMRanker ranks for identifiers. Shared text dimensions are 9310+10000.
AttributeBundle adds a 32-class schema router and 24 conditional field groups. The 724-row
identifier target pool produces 546 actual positive/negative ranking groups.
