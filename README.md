# Theme Leadership OS

Theme Leadership OS v0.1은 무료 공개 데이터와 시점 기준(point-in-time)
원칙으로 테마 주도권을 연구하는 로컬 우선 Python 패키지다. 연도별 상승
테마, 초입 상승 레이더, 현재 주도·신흥·12개월 후보 신호를 같은 데이터
계약과 감사 가능한 결과로 제공한다.

## 설계 원칙

- **질문을 분리한다.** `Current Leader`, `Emerging Radar`, `12M Hold Candidate`는 서로 다른 화면과 게이트를 가진다. 초입 신호가 곧 매수 추천은 아니다.
- **무료 데이터만 사용한다.** 공개 출처의 이용 조건을 확인하고, 유료 피드를 요구하거나 원시 가격 파일을 재배포하지 않는다.
- **시점 기준을 지킨다.** 이벤트·구성종목·가격의 발생일과 이용 가능일을 분리해 미래 정보가 과거 결정에 들어가지 않게 한다.
- **M0를 기준으로 삼는다.** 26주 SPY 상대강도 순위를 고정 기준선으로 두고, 새 규칙은 M0·도전 사례·음성 대조군과 비교한다.
- **검증 전에는 관찰만 한다.** 모든 결과에 자료 충족률·경고·검증 상태를 남기며 자동 주문을 만들지 않는다.

## 핵심 기능

- `annual_theme_leaders`: 연도별 테마 수익률, SPY 수익률, 초과수익률, 자료 충족률, 순위를 계산한다.
- `early_rise_signals`: 4·8·13주 상대강도 가속, 26주 기준선, breadth, 참여율, 3주 중 2주 지속성을 이용해 `초입 관찰`·`초입 확인`을 표시한다.
- `current_leader`, `emerging_radar`, `hold_candidate_12m`: 현재 주도·신흥 레이더·12개월 지속 후보를 독립적으로 계산한다.
- 검증 도구: 무결성 감사, rank-IC·top-bottom 스프레드, 조기 적중창, 에피소드 라벨, purged walk-forward 분할, 도전 사례·음성 대조군.
- 로컬 CLI: `doctor`, `catalog`, `score`, `validate`, `annual`, `dashboard`.

초입 규칙의 상세 설명과 탐색 결과는 [`docs/ANNUAL_LEADERSHIP.md`](docs/ANNUAL_LEADERSHIP.md)에 있다. 2018~2026년 표는 조정 전 Nasdaq 공개 가격과 현재 확인 가능한 바스켓을 사용한 탐색 결과이므로 확정 성과로 해석하면 안 된다. 기존 탐색 로그는 [`research/EXPLORATORY_BASELINE.md`](research/EXPLORATORY_BASELINE.md)에 보관한다.

## 입력 형식

가격 패널은 다음 네 열을 갖는 CSV/JSON이어야 한다. `SPY` 행이 시장 기준선이며, 나머지 행은 테마 구성종목이다.

```text
date,theme,security,value
2025-01-03,__market__,SPY,590.12
2025-01-03,solar,TAN,35.80
```

실행 예시는 다음과 같다.

```bash
theme-leadership doctor
theme-leadership catalog
theme-leadership score --demo
theme-leadership annual --demo
theme-leadership annual --input prices.csv --as-of 2025-06-30 --json
theme-leadership validate --input prices.csv
```

`annual`의 JSON 결과에는 `annual_leaders`와 `latest_early_signals`가 포함된다. 신호의 추천 필드는 항상 “관찰 전용 · 매수 추천 아님”이다.

## 개발

Python 3.11 이상이 필요하다.

```bash
python -m pip install -e '.[dev]'
ruff check src tests
python -m compileall -q src
python -m pytest -q
```

Typer·Streamlit은 선택 의존성이다. Typer가 없으면 argparse로 CLI를 실행하고,
Streamlit이 없으면 대시보드가 선택 의존성 설치 방법을 안내한다. 기본 CLI는
네트워크를 호출하지 않으며, yfinance 어댑터도 개인 연구용 명시적 opt-in이
필요하고 완전한 PIT 데이터가 아니다.

## 라이선스와 데이터 주의

소프트웨어는 Apache License 2.0으로 배포한다([`LICENSE`](LICENSE)). 출처별
데이터 이용 조건은 소프트웨어 라이선스와 별개이므로 [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md)를 확인한 뒤 새 출처나 산출물을 추가한다.
