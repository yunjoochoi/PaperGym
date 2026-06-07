# VRN: Verifiable Recombination Novelty — 설계 문서

- **Date:** 2026-06-06
- **Status:** ⚠️ SUPERSEDED by [`2026-06-07-execution-gym-design.md`](2026-06-07-execution-gym-design.md). VRN/provenance는 폐기가 아니라 실행 Gym의 *reward novelty 가드 부품*으로 재배치됨. 이 문서는 진단(§1 null 결과)·VRN 정의(§4)·Si et al. 데이터 인벤토리(§5)의 참고용으로 보존.
- **Target venue:** ICLR / NeurIPS / ICML 계열 (measurement + reward-modeling 트랙)
- **Repo:** PaperGym (이 레포). 외부 앵커 데이터: `../AI-Researcher` (NoviScl/AI-Researcher, Si et al.)

---

## 1. 문제 정의 (왜 지금 PaperGym은 탑티어가 아닌가)

현재 Stage 3 결과는 headline claim을 *데이터로 반증*하고 있다. 자체 릴리스 런
([`data/eval/layer12_20260503T172939/summary.json`](../../../data/eval/layer12_20260503T172939/summary.json),
[`data/eval/ideation_20260503T131152/summary.json`](../../../data/eval/ideation_20260503T131152/summary.json)) 기준:

| 신호 | 값 | 해석 |
|---|---|---|
| per-condition novelty | A 3.97 / B 4.07 / **C 4.13 / D 4.13** | targeted retrieval(C)이 random control(D)과 동일 |
| pairwise novelty C-vs-D | **C 14 / D 16 (46.7%)** | C가 random control에 *짐* |
| pairwise novelty C-vs-A / C-vs-B | 60% / 67% | C는 no-retrieval/same-domain은 이김 |
| validity C-vs-A | A 15 / C 6 / tie 9 | C가 baseline보다 *덜* valid |
| novelty hist | 거의 전부 "4" (stdev ~0.4), validity는 거의 전부 "5" (A는 stdev 0.0) | **LLM-judge 축이 포화되어 해상도 없음** |
| inspired_by incorporation (McNemar) | no-incorporation 문제 C 1/30 vs D 8/30, p≈0.016 | **C는 인용 메커니즘을 실제로 더 잘 녹임 (유일한 견고한 우위)** |

**진단:** "노벨티 정량화가 힘들다"는 방법론 한계가 아니라 *측정 도구(LLM-judge Likert)가 포화*된 결과다.
효과를 내는 것은 seed *diversity*이지 *targeting*이 아니며(D도 cross-domain 다양성을 받음),
C의 유일한 견고한 우위는 **provenance fidelity**(인용한 메커니즘의 실제 구현)다.

## 2. 반박 대상 (정면)

NoviScl/AI-Researcher (Stanford NLP, Si et al.) — 두 논문:

1. **"Can LLMs Generate Novel Research Ideas?" (arXiv:2409.04109)** — 79명 전문가 리뷰로 LLM 아이디어가 인간보다 *더 novel*하다고 결론.
2. **"The Ideation–Execution Gap"** — 같은 아이디어 43개를 *실제 실행*한 뒤 재평가하니 LLM 아이디어 점수가 인간보다 훨씬 더 떨어지고 **순위가 뒤집힘**(인간 > LLM).

핵심 통찰: **Si et al. 본인들이 이미 "ideation 단계 novelty 점수는 신뢰 불가"를 증명**했다 —
단, *사후에, 실행을 다 하고 나서야*. 우리의 무기: **provenance는 ideation 시점에 자동으로 계산되는 검증가능한 신호**이며, 실행 생존을 *사전에* 예측한다.

## 3. 한 줄 주장 (thesis)

> LLM 아이디어 novelty는 평가자 Likert 점수(포화·gaming가능·실행에서 붕괴)가 아니라
> **VRN(Verifiable Recombination Novelty)** — *반증가능한* mechanism provenance — 로 측정해야 한다.
> 이 렌즈에서 Si et al.의 "LLM이 더 novel" 결론은 상당 부분 측정 아티팩트이며,
> VRN은 ideation 시점에 실행 생존(execution survival)을 예측한다.

기여(contribution)는 *시스템*이 아니라 *측정론(measurement)*이다. 이게 "그래서 뭐가 새롭냐"
공격을 막는 핵심이다.

## 4. VRN 지표 정의

방법 제안을 다음으로 분해한다: 타깃 문제 P (home domain dom(P)), 차용 메커니즘 집합
{(Mᵢ, Dᵢ)} — Mᵢ는 메커니즘, Dᵢ는 그 출처 도메인. 세 축 모두 자동 계산이며 *반증가능*하다.

- **T — Transfer distance.** dom(P)와 Dᵢ 사이 거리(도메인 그래프 / 인용그래프 / 임베딩).
  same-domain ≈ 0, cross-domain 클수록 큼. recombination 거리.
- **A — Literature absence.** "dom(P)의 선행연구에 Mᵢ→P 적용이 *부재*한가"를
  Semantic Scholar 검색 + entailment 판정으로 검사. **이것이 반증가능 핵심**:
  리뷰어/후속연구/자동검사가 반례 논문 하나로 깰 수 있다.
- **F — Provenance fidelity.** 인용한 Mᵢ가 본문에 *실제로 구현*됐는가.
  PaperGym 기존 `inspired_by` grounding(incorporation 1–3) 재활용. citation-stuffing 차단.

**합성:** VRN은 *멀고(T↑) ∧ 새롭고(A↑) ∧ 진짜 녹인(F↑)* transfer에만 점수를 준다.
(예: 거리만 있고 absence 없음 = 이미 존재하는 재조합; absence만 있고 fidelity 없음 = 빈 name-dropping.)
정확한 합성 함수(곱/가중합/lexicographic)는 implementation plan에서 ablation으로 결정.

**불변식(advertise할 속성):** 모든 VRN 점수는 falsifiable하다 — Likert vibe의 대척점.

## 5. 데이터셋

| 출처 | 규모 | 위치 | 용도 |
|---|---|---|---|
| PaperGym A/B/C/D 생성물 | 30 query × 4 condition | `data/eval/ideation_*` | E4 (자기 결과 재해석), E5 (RL) |
| Si et al. ideation 리뷰 | 337 리뷰 (novelty_score + **novelty_rationale 텍스트** + 4축 + condition + idea_id) | `../AI-Researcher/reviews_ideation/data_points_all_anonymized.json` | E1 앵커 |
| Si et al. 아이디어 본문 | Human / AI / AI+Rerank 전체 제안서 | Google Drive (README L157–163), `gdown`으로 취득 | E1/E2의 VRN 계산 입력 |
| Si et al. execution 리뷰 | 181 리뷰 / 43 아이디어 (pre/post + faithfulness + codebase_quality) | `../AI-Researcher/reviews_execution/data_points_all_execution.json` | E2 센터피스 |
| Si et al. 실행 산출물 | 편집 아이디어 + 실행 논문 + 코드베이스 | Google Drive (README L169) | E2 fidelity 보강 |

조인 키: `idea_id` (예: `Safety_2_AI`). `reviews_execution/compare_change.py`가 study1↔study2를
idea_id로 조인해 Δ(post−pre)를 계산하는 방식이 이미 구현돼 있음 — 그대로 재사용.

## 6. 실험 (검증 백본, human 0명)

### E2 — 실행 생존 예측 [센터피스]
- **가설:** ideation-시점 VRN이 Δ(post−pre execution score)를 예측한다. 전문가 `novelty_score`는 못 한다(또는 더 약하다).
- **방법:** 43개 공통 아이디어에 대해 ideation 제안서로 VRN 계산 → Δscore에 회귀.
  VRN-기반 예측 vs novelty_score-기반 예측을 상관/효과크기/AIC로 비교. metrics = novelty/excitement/effectiveness/overall.
- **성공 기준:** VRN의 Δscore 예측력이 Likert-novelty보다 유의하게 높음 (단측, 효과크기 보고).
- **리스크:** n=43 작음 → 무거운 모델링 금지, 상관·효과크기 중심, E1(n=337)과 결합 해석. 부트스트랩 CI.

### E1 — 구성타당도 @ ideation (n=337)
- **E1a:** VRN vs `novelty_score` 상관 (전체 / condition별).
- **E1b — grounding gap:** AI 아이디어가 Likert-novel하지만 VRN은 낮은가? (AI vs Human vs AI_Rerank의 VRN 분포). Si et al.의 "AI가 더 novel" 주장을 VRN으로 재검.
- **E1c — rationale 마이닝:** `novelty_rationale` 텍스트에서 전문가가 novelty를 깎을 때 "literature-absence"식 추론("이미 X하는 연구가 많다")을 하는 비율을 LLM으로 분류. → **VRN이 전문가가 실제로 쓰는 구성개념임**을 입증.

### E3 — Adversarial gaming battery
- **방법:** baseline 아이디어에 "진짜 novelty는 안 올라야 하는" 변형: (a) jargon 주입, (b) 안 쓴 `inspired_by` 인용 stuffing, (c) 도메인 buzzword 끼얹기.
- **성공 기준:** LLM-judge novelty는 유의하게 ↑(=hack됨), VRN은 평탄/↓. → 기존 지표가 깨졌다는 직접 증거.

### E4 — PaperGym 자기 결과 재해석
- **방법:** 기존 A/B/C/D 생성물을 VRN으로 재채점.
- **가설:** Likert에서 C≈D(현재 null)지만 **VRN에서 C>D** (C는 F가 높음 — McNemar p=0.016이 이미 뒷받침; D는 random이라 absence/fit 약함).
- **효과:** *지금의 약점(null)이 헤드라인*이 됨 — "Likert가 못 잡는 걸 VRN이 잡는다"의 자체 증명.

### E5 — RL Goodhart stress-test
- **방법:** VRN을 reward로 synthesizer를 RL 학습(기존 RL 시도 편입). 최적화 곡선 기록.
- **판정:** 학습 중 held-out 독립 지표(+E1 human 앵커)도 같이 오르면 "VRN은 hack 안 되는 좋은 target"(positive); 발산하면 "여기가 VRN의 breaking point"(이것도 finding).
- **효과:** reward-modeling 커뮤니티(ICLR/NeurIPS 핵심)에 직격. RL이 자랑이 아니라 *지표 검증 실험*이 됨.

## 7. PaperGym 레포에 필요한 변경 (개요)

- `eval/ideation/`에 VRN 채점기 추가 (T/A/F 서브모듈). A는 기존 S2 클라이언트(`eval/ideation/evaluate.py`의 `_s2_search`) 재사용, F는 `inspired_by_grounding` 재사용.
- `scripts/`에 외부 앵커 파이프라인: AI-Researcher 데이터 로드 + 아이디어 본문 `gdown` 취득 + idea_id 조인.
- E1–E4 분석 스크립트 + E5 RL 학습 루프(별도 모듈, 분리 가능).
- 새 결과는 기존 패턴대로 `data/eval/`에 timestamped run으로.

세부 단계·파일·인터페이스는 implementation plan에서 확정.

## 8. 리스크 / 미티게이션

| 리스크 | 수준 | 미티게이션 |
|---|---|---|
| A(literature-absence) 자동검사 recall 부족 → false "absence"가 VRN을 띄움 | 상 | precision/recall 보고, 인간 spot-check 소량, adversarial 검증(E3), absence 임계 ablation |
| E2 n=43 작음 | 중 | 상관·효과크기·부트스트랩 CI, E1과 결합, 과적합 모델 금지 |
| Drive 본문 텍스트 포맷/매칭 | 중 | id_title_mapping.csv로 idea_id 매칭 검증, 누락 보고 |
| E5 RL이 수렴 안 함/컴퓨트 | 중 | E5는 분리 가능한 모듈 — 안 되면 E1–E4만으로도 논문 성립 |
| VRN이 결국 또 하나의 LLM-judge 아니냐는 비판 | 상 | A는 검색-근거(외부 문헌), F는 deterministic 구현체크 — *반증가능성*을 전면에. judge는 보조로만 |

## 9. 한계 (논문에 명시)

- 새 human study 없음 → "전문가와 일치"는 *재활용 라벨*로만 주장. (단 Si et al. 데이터가 정확히 반박 대상이라 오히려 강점.)
- VRN은 *mechanism-level* recombination novelty에 한정 — 문제설정 novelty·empirical novelty는 범위 밖.
- ML 도메인(PaperGym 7개) 한정.

## 10. 재현성 / 산출물

- 모든 런: `data/eval/<name>_<ts>/`. 클레임-스크립트-필드 매핑은 기존 [`docs/REPRODUCE.md`](../../REPRODUCE.md) 패턴 확장.
- 외부 데이터 취득 스크립트 + 해시 고정.

## 11. Open questions (plan 단계에서 해소)

1. VRN 합성 함수 형태 (곱 vs 가중합 vs lexicographic) — ablation으로.
2. A의 도메인 그래프 정의 (S2 field 기반 vs 임베딩 클러스터).
3. E5 RL 알고리즘 (GRPO/PPO/best-of-n reranking) 및 reward 정규화.
4. E2에서 Δscore를 어느 metric(novelty/overall)으로 잡을지, control 변수(topic, familiarity).
