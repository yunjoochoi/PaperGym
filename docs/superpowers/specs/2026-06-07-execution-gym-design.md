# PaperGym: 연구 아이디어를 위한 검증된 실행 Gym — 설계 문서

- **Date:** 2026-06-07
- **Status:** Design approved (brainstorming) → 다음 단계: implementation plan (writing-plans)
- **Supersedes:** `2026-06-06-vrn-provenance-novelty-design.md` (provenance/VRN은 폐기가 아니라 *reward의 novelty 가드 부품*으로 재배치됨)
- **Target venue:** ICLR / NeurIPS / ICML 계열 (evaluation environment + agents 트랙)
- **Repo:** PaperGym. 외부 검증 데이터: `/home/shaush/AI-Researcher` (NoviScl/AI-Researcher, Si et al. 두 논문)

---

## 1. 문제 정의

연구 아이디어의 질을 *ideation 시점*에 평가하는 것은 신뢰 불가능하다. Si et al. 두 논문이 이를 증명:

1. **"Can LLMs Generate Novel Research Ideas?" (2409.04109):** 79명 전문가가 LLM 아이디어를 인간보다 *더 novel*하다고 평가. 단 LLM-as-judge novelty는 휴먼과 칼리브레이션 안 됨(저자 본인 인정).
2. **"The Ideation–Execution Gap" (2506.20803):** **43명 전문가**가 같은 아이디어를 *실제 실행*(1인 평균 103시간, 3개월) → AI 아이디어 점수가 인간보다 훨씬 더 떨어지고 **순위가 뒤집힘**.
   - Table 3(실행 후): AI vs Human이 effectiveness/soundness/excitement/overall에서 p=0.01\*\*로 갈림. **novelty는 p=0.21(유의 아님)** — ideation의 AI novelty 우위가 실행 후 소멸.
   - Table 5(drop): AI novelty −1.049 vs Human −0.010 (p=0.025\*). execution은 *거짓 novelty를 깎는* 신호.
   - Appendix D: ideation↔execution 점수 상관 약함, AI excitement는 음의 상관.
   - §5.2/Fig 3: execution 리뷰어는 empirical performance(93%)·실험설계·baseline·ablation을 봄 — ideation 텍스트엔 없는 정보.

**핵심 병목:** 진짜 신호(execution)는 *사람이 실행해야 해서 비싸고 느려 43개가 한계*다.

## 2. 한 줄 주장 (thesis)

> 연구 아이디어의 질은 ideation 평가(휴먼-비칼리브 LLM-judge·실행에서 붕괴)가 아니라
> **실행 결과**로 측정해야 한다. 우리는 그 실행을 사람 대신 자동화하는 **검증된 실행 Gym**을 만든다 —
> inference-time으로 제약해 API call만으로 실험이 돌아가게 하고, **Si et al.의 사람-실행 결과를
> 재현함을 보여 gym을 신뢰가능한 proxy로 검증**한 뒤, 사람이 못 한 규모(N≫43)로 확장한다.
> 나아가 *실행 결과로 다듬은(test-time feedback) 아이디어가 ideation-execution gap을 좁힘*을 보인다.

기여는 *시스템 자랑*이 아니라 **evaluation environment + 그것이 드러내는 reward 설계 통찰**이다.

## 3. 기여 (contributions)

- **C1 — 검증된 실행 Gym [스파인]:** inference-time NLP 아이디어를 LLM이 자동 실행하는 환경. **Si et al. 43개 아이디어에서 사람-실행 ideation-execution gap을 재현**함을 보여 proxy로 검증.
- **C2 — dual-reward 통찰:** execution은 *effectiveness*만 양의 신호로 주고 novelty엔 안 줌 → execution-only 최적화는 incrementalism으로 Goodhart. 따라서 **provenance(VRN)를 reward의 novelty 가드**로 결합해야 함. (이게 측정/리워드 설계 기여.)
- **C3 — test-time → RL 최적화:** test-time execution-feedback이 gap을 좁힘(effectiveness↑, novelty 붕괴 없이)을 N≫43에서 입증. dual reward로 RL 스케일.

## 4. 아키텍처

```
PaperGym Execution Gym
├── Ideation (기존 자산)          : seed library + paraphraser + synthesizer → 아이디어 제안서
│                                   (Si et al. 입력 아이디어도 그대로 주입 가능)
├── Execution Engine (신규)       : inference-only. 표준 하니스에서 LLM이 실험 코드/프롬프트를
│                                   조립·실행. 여러 LLM에 API call만 열어둠(학습 없음).
├── Evaluation (신규)             : 실행 산출물의 empirical 결과 측정 (정해진 task metric)
│                                   + faithfulness 점검(제안서↔실행 일치)
├── Dual Reward                   : R = effectiveness(execution, 객관)  ∧  not-derivative(VRN, 검증가능)
└── Optimizer
    ├── test-time feedback [메인] : propose → execute → 결과 읽기 → 재생성 (best-of-N/self-refine)
    └── RL [스케일/2차]           : dual reward로 정책 학습 (예산 허용 시)
```

## 5. Execution Engine (핵심 신규 컴포넌트)

- **Action space = inference-time NLP만:** 프롬프팅·디코딩 전략·retrieval·test-time compute·compound LLM system. **GPU 학습 없음.** Si et al. 7토픽(Bias/Coding/Safety/Multilingual/Factuality/Math/Uncertainty) 재사용. (그들의 예시 8개가 전부 여기 해당.)
- **표준 실행 하니스:** 고정된 데이터셋/스플릿/metric/baseline을 환경이 제공 → 아이디어 간 비교가능. 에이전트는 *방법*만 조립.
- **다중 LLM API:** 환경이 model-pool(여러 LLM)에 호출만 열어둠. 실험 = inference 다발.
- **faithfulness 제어 [최우선]:** "결과가 나쁜 게 아이디어 탓인가 실행 탓인가" 교란 차단.
  - 제안서 ↔ 실행 코드 일치 자동 점검(Si et al.의 faithfulness_score와 같은 척도)
  - 제약된 action space + 실행 로그 보존 + 동일 하니스로 분산 축소
  - 검증(E1)에서 **LLM-실행 faithfulness를 사람-실행(6.4/10)과 같은 척도로 비교**
  - **[①에서 발견] 사람-실행 논문들도 이미 LLM-as-judge로 평가함** — 제안서의 (optional) human eval을 조용히 LLM-judge로 대체하고 심지어 "human-aligned"로 표기(H2/H3/H6에서 확인). → gym이 LLM-judge를 쓰는 건 *executor 관행과 동일*이라 faithfulness 논거가 오히려 강해짐. 단 execution 결과가 순수 객관이 아니라 *객관 metric + LLM-judge 혼합*임을 인지해야 함.

## 5.1 실행가능성 사전검증 결과 (①, 2026-06-07)

부록 showcased 8개 아이디어(H1–H8)를 각각 분류 + 적대적 검증(숨은 학습/휴먼/비공개데이터/LLM-judge 대체 색출)으로 점검한 결과:

- **8/8 모두 `inference_only`로 자동실행 가능** — 적대적 검증에서 단 하나도 뒤집히지 않음(블로커 0). 사용자가 물은 Factuality(H2)·Uncertainty(H4) 포함.
- 반복 패턴: 메서드는 전부 프롬프팅+retrieval, "학습"으로 보이는 부분은 (i) 프롬프트 최적화(DSPy, 가중치 미변경), (ii) post-hoc calibration(로지스틱/temperature), (iii) frozen 임베딩 사용 — 전부 inference-friendly. 제안서의 human eval/fine-tune은 대부분 *optional fallback*이라 미실행.
- 핵심 caveat 2개: ① **model 치환**(GPT-4o/Gemini 하드코딩 → gym이 자기 LLM으로 치환, 절대수치 달라짐 → E1은 절대값 아닌 *상대 패턴/상관*을 타깃) ② **inference throughput**(H1 4역할×4000+MCQ, H6 ~20k 프롬프트 등 — 학습은 없지만 호출량 큼 → 대규모/RL 비용 예산 필요).
- 한계: 이 점검은 *showcased 8개*만. 전체 43개(Drive)에 동일 점검 필요. corpus가 "3개월·1인 실행가능"으로 scoping되어 커버리지 높을 것으로 기대하나 Math/Coding 일부에 학습 케이스 가능.

## 6. Dual Reward

- **effectiveness (execution):** 실행 결과가 제공된 baseline을 이긴 정도. **객관 metric(accuracy/ROUGE/ECE/ASR 등)을 우선**하고 LLM-judge 기반 측정은 별도 취급(①에서 사람-실행도 객관+LLM-judge 혼합임이 확인됨). 객관 부분은 휴먼-칼리브 회피.
- **novelty 가드 (VRN, 검증가능):** 차용 메커니즘이 (T) 멀고 (A) dom(P)에 부재하고 (F) 실제 구현됐는가. *진짜 novelty를 양으로 보상*하기보다 **incrementalism Goodhart를 차단**하는 게이트 역할.
- **합성:** effective ∧ not-derivative. 정확한 형태(곱/임계 게이트/가중합)는 plan에서 ablation.

## 7. 실험

### E1 — Gym이 사람-실행 gap을 재현하는가 [센터피스/검증]
- **데이터:** Si et al. 43개 아이디어 제안서(Drive 공개) + 사람-실행 점수(`reviews_execution`).
- **방법:** inference-time 실행 가능한 부분집합을 gym으로 실행·평가 → 사람-실행 결과와 비교.
- **성공 기준:**
  - (a) per-idea gym-execution 점수 ↔ human-execution 점수 상관 (양·유의)
  - (b) gym이 gap 부호/크기 재현 (Table 5: AI gap ≈ −1~−2, Human ≈ 0; 순위 flip)
  - (c) gym-실행 faithfulness ≈ 사람-실행 수준
- **커버리지 [①에서 사전검증]:** showcased 8/8이 inference-only 자동실행 가능 확인(§5.1). 전체 43개(Drive)에 동일 점검 후 비율 확정(정직). corpus scoping상 커버리지 높을 것으로 기대.
- **리스크:** 일부 아이디어는 학습 필요 가능(특히 Math/Coding) → 부분집합만. 커버리지를 한계로 명시.

### E2 — 사람이 못 한 규모로 확장
- 검증된 gym으로 N≫43 아이디어 실행·평가. ideation-execution gap을 대규모로 정량화(조건·모델·토픽별).

### E3 — dual-reward 통찰 입증
- execution-only로 선택/최적화하면 incrementalism으로 수렴(novelty 가드 없을 때 VRN↓·전형성↑)을 측정.
- VRN 가드 추가 시 effectiveness 유지하며 novelty 붕괴 방지됨을 대조. → C2의 직접 증거.

### E4 — test-time execution-feedback이 gap을 좁히는가 [메인 최적화]
- **대조:** (i) ideation-judge로만 다듬기 vs (ii) **실행 결과로 다듬기**(propose→execute→revise).
- **가설:** (ii)가 실행 후 effectiveness↑, 그리고 (VRN 가드로) novelty 붕괴 없음. ← Si et al. 실패모드 정조준.

### E5 — RL 스케일 (staged)
- E4가 "루프가 효과 있다"를 싸게 증명한 뒤, dual reward로 정책 RL 학습(예산 허용분).
- **판정:** RL 전/후 gym-effectiveness delta(자동) + VRN 유지(자동). "휴먼-급 novelty 확인"은 limitation으로 명시.
- **비용 주의:** reward = 실험 1회 → rollout 수 = 실험 수. 예산 상한·캐싱·저비용 task 우선.

## 8. PaperGym 레포 변경 (개요)

- 신규 `execution/` 모듈: 표준 하니스, model-pool API 래퍼, 실행 러너(샌드박스 재사용 가능 — 기존 `src/papergym/env/` Docker), faithfulness 점검기.
- 신규 `eval/execution/`: empirical metric 채점 + gap 분석.
- `eval/ideation/`에 VRN 채점기(novelty 가드). A=기존 `_s2_search` 재사용, F=`inspired_by` grounding 재사용.
- `scripts/`: Si et al. 데이터 `gdown` 취득 + idea_id 조인 + E1 재현 러너; test-time feedback 러너; (옵션) RL 러너 = 기존 RL 코드 재배치.
- 결과는 기존 패턴대로 `data/eval/<name>_<ts>/`.

## 9. 리스크 / 미티게이션

| 리스크 | 수준 | 미티게이션 |
|---|---|---|
| **execution faithfulness 교란** (결과↔아이디어 vs 실행 탓) | **상** | 제약 action space, faithfulness 자동점검, E1에서 사람-실행과 동일척도 비교, 실행로그 공개 |
| reward = 실험 1회 → RL 비용 | 상 | test-time 먼저(E4), RL은 staged·예산상한·저비용 task 우선 |
| inference-time 제약이 아이디어 공간 축소 | 중 | 잘 동기화된 scope(test-time compute는 현 NLP 주류); 커버리지 명시 |
| gym 자체 타당도 ("진짜 실험 맞냐") | 상 | C1 검증이 곧 이 답 — Si et al. 사람 데이터 재현이 통과 못 하면 논문 보류 |
| execution-only Goodhart(incrementalism) | 중 | VRN novelty 가드(C2/E3) |
| 43개 중 실행 가능 부분집합이 너무 작음 | 하 (①에서 완화) | showcased 8/8 inference-executable 확인(§5.1) → 부분집합 충분 기대; 전체 43 점검 후 확정 |
| model 치환으로 절대수치 불일치 | 중 | E1 성공기준을 절대값 아닌 *상대 패턴/상관*으로(이미 그렇게 설계); model을 통제변수로 보고 |

## 10. 한계 (논문 명시)

- inference-time NLP에 한정(학습 필요 아이디어 제외).
- "휴먼이 보기에 RL 후 아이디어가 진짜 더 좋다"의 완전한 주장은 휴먼 필요 → 자동(execution+VRN)으로 대체하고 한계로 명시. (단 gym 자체가 사람-데이터에 검증되어 effectiveness 신호는 휴먼-정렬.)
- gym 실행이 사람 실행만큼 창의적 디버깅·우회를 못 할 수 있음(faithfulness로 측정·보고).

## 11. 재현성 / 산출물

- 모든 런 `data/eval/<name>_<ts>/`. 실행 로그·코드·결과 공개. 클레임-스크립트-필드 매핑은 기존 `docs/REPRODUCE.md` 확장.
- Si et al. 데이터 취득 스크립트 + 해시 고정. 표준 하니스(데이터/스플릿/metric/baseline) 버전 고정.

## 12. Open questions (plan에서 해소)

1. 표준 하니스의 task/데이터셋/metric 셋 확정 (7토픽별 최소 1개씩).
2. faithfulness 자동점검 구현(제안서↔코드 정합 판정 방식).
3. dual reward 합성 형태(곱/게이트/가중합) ablation.
4. E1 커버리지: 43개 중 실행대상 선정 기준.
5. test-time feedback 루프 설계(best-of-N vs 반복 self-refine, 예산).
6. RL 알고리즘·reward 정규화·rollout 예산(staged).
