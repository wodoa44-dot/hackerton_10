# 매뉴얼과 CSV 대조

원본 14,249행을 보존한 채 코드표와 자리 배치를 대조했습니다.

| 구분 | 행 수 |
|---|---:|
| outside_manual | 10,313 |
| layout_conflict | 1,529 |
| matched | 2,407 |

## 적용 원칙

matched는 코드표의 모든 필드와 고정 구분자를 통과하고 분해 후 원래 코드로 재조립되는 행입니다.
layout_conflict는 매뉴얼 규칙과 맞지 않는 행입니다. 고객사/기존 관리 코드일 수 있으므로 원본을 고치거나 상세 학습 라벨로 사용하지 않습니다.
outside_manual은 매뉴얼에 확인된 자리 규칙이 없는 형식입니다. 등록 코드 조회와 코드 분류에 사용할 수 있습니다.
학습 선택에는 원본 training 중 평가 입력 중복을 제외한 데이터만 사용합니다. 전체 CSV 대조 결과를 검증 성능으로 사용하지 않습니다.

## 문서 내부 불일치

- 21쪽: Semiconductor diagram S versus prose C / Fixed Capacitor. Use diagram S; subtype SS/ST/SF and remaining layout are CSV-supported extensions, not definitions from this page
- 25쪽: Variable resistor diagram 17 versus prose 18; switch entries are mixed with thermistors. Use 17 for 75/76 thermistors, supported by CSV; explicitly support observed 1701 legacy switch but route new switches to 18
- 27쪽: Connector diagram 19 versus prose 18. Use diagram 19, supported by CSV connector rows

## 전해콘덴서 용량 표기

{"CE": {"pf": 150, "uf": 8, "value_conflict": 2}, "CM": {"uf": 18, "pf": 3, "value_conflict": 7}}
매뉴얼은 uF 기준입니다. 실사용 코드에는 pF와 uF 기준이 함께 존재하여, retained training에서 근거 건수를 집계한 기본 정책과 함께 출력합니다.
