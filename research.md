# PaperGym 심층 분석 보고서

> 분석 일자: 2026-06-07 · 대상 경로: `/home/shaush/__research/PaperGym`
> 본 보고서는 PaperGym 코드베이스 전체(소스, 평가 하니스, 스크립트, 데이터 산출물, 설계 문서)를 코드 레벨에서 분석한 결과를 정리한 것이다.

---

## 0. 한 문장 요약

**PaperGym은 "각 ML 논문을 LLM 에이전트가 탐사하는 인터랙티브 환경(gym)으로 취급"하여 논문에서 *메커니즘 시드(mechanism seed)*를 추출·축적하고, 연구 질의를 7개 ML 도메인으로 패러프레이즈하여 도메인을 가로지르는 시드를 검색한 뒤, 어디서 무엇을 빌려왔는지를 명시적으로 인용(attribution)하는 새 연구 방법을 합성하는 시스템이다.** 여기에 더해, 합성된 아이디어가 실제로 "실행되어 효과가 있는지"까지 자동 측정하려는 **Execution Gym** 서브시스템이 최근 추가되고 있다.

핵심 학술적 주장은 두 가지다.
1. **도구 증강(tool-augmented) 추출**이 단순 프롬프팅보다 더 구체적인(specific) 시드를 만든다 (grounding은 유지하면서 specificity 향상).
2. **크로스 도메인 패러프레이즈 검색**이 단일 질의 검색보다 더 넓은 도메인의 메커니즘을 끌어와(domain coverage 증가), 더 참신하고 출처가 명확한 방법 합성을 가능하게 한다.

---

## 1. 동기와 연구 맥락

기존 ML 아이디어 합성 시스템은 보통 **질의와 같은 하위 분야(subfield)에서만** 선행 연구를 검색한다. PaperGym은 다른 입장을 취한다. 질의를 **7개 ML 도메인 각각의 언어로 패러프레이즈**한 뒤, 각 패러프레이즈로 **전체 라이브러리에서** 메커니즘 시드를 검색하고, "이 메커니즘을 어느 도메인에서 빌려왔다"를 명시하는 방법을 합성한다. 즉, 분야 간 메커니즘 *전이(transfer)*를 의도적으로 유도하는 것이 핵심 차별점이다.

Execution Gym 서브시스템은 Si et al.의 두 논문(2409.04109 "Can LLMs Generate Novel Research Ideas?", 2506.20803 "The Ideation–Execution Gap")에 대한 직접적 응답이다. 이들 논문은 (1) LLM 아이디어가 *생성 시점*에는 인간보다 참신하다고 평가되지만, (2) 전문가가 실제로 *실행*하고 나면 그 우위가 사라지고 순위가 뒤집힌다는 것을 보였다. 다만 인간 실행은 아이디어당 평균 ~103시간, 3개월이 들어 N=43에 그쳤다. PaperGym의 Execution Gym은 **이 비싼 인간 실행을 추론 시점(inference-time) API 호출만으로 자동화**하여 대규모(N≫43)로 idea-execution gap을 재현·측정하려는 시도다.

---

## 2. 전체 아키텍처 개관

시스템은 4개의 단계로 구성된다(앞 3개가 논문의 메인 파이프라인, 4번째가 신규 서브시스템).

```
[Stage 1: ACCUMULATOR]   논문 → Docker 샌드박스(gym) → read/grep/bash로 탐사 → 1~3개 시드 추출
                                       │
                                       ▼
                          data/library/ (도메인 분할 FAISS, 샤딩됨; 1,167 시드 / 446 논문 / 7 도메인)
                                       │
[Stage 2: PARAPHRASER]   질의 → 7개 도메인 문제-패러프레이즈 → retrieve_cross_domain (전역 top-k)
                                       │
                                       ▼  (seed, origin_domain, lens_text) 튜플
[Stage 3: SYNTHESIZER]   질의 + 시드들 + 렌즈 → method + rationale + inspired_by(출처 명시)
                                       │
                                       ▼
[EVAL]                   판정자(judge) LLM이 specificity/grounding/relevance/novelty/validity/
                         coherence/incorporation을 채점 (생성자와 다른 모델 패밀리 = self-bias 방지)

[Stage 4 (신규): EXECUTION GYM]
                         아이디어 제안서 → 샌드박스에서 method.py 작성·실행 → predictions.json
                         → 효과성(effectiveness = method_metric - baseline) + faithfulness 채점
```

### 조건 A/B/C/D (실험 설계의 중심)

Stage 3는 4개의 조건으로 ablation을 수행한다. 모두 동일한 `IdeationOutput` 자료구조를 반환하여 판정자가 균일하게 처리한다. B/C/D는 **시드 예산 21개(= k_per_domain 3 × 7 도메인)를 공유**하여 입력 규모를 맞춘다(A는 시드 0개).

| 조건 | 검색 방식 | 합성기에 전달되는 렌즈(lens) | 시드 예산 | 분리(isolate)하는 신호 |
|---|---|---|---|---|
| **A** | 없음(파라메트릭 지식만) | 없음 | 0 | 베이스라인, 시드 자체의 효과 |
| **B** | 임베딩 검색, **자연 도메인 내부만** | 원본 질의 (×N) | 21 | 동일 도메인 검색 |
| **C** | 임베딩 검색, **패러프레이즈 기반 크로스 도메인** | 각 시드를 찾아낸 패러프레이즈/원본 질의 텍스트 | ~21 | **제안 방법(핵심)** |
| **D** | **무작위**(임베딩 없음, seed=0) | `"(uniform random sample of the library)"` | 21 | 합성기만의 기여(널 검색) |

- **C vs D**: 임베딩 검색 신호를 분리 ("C가 D를 못 이기면 검색이 품질 향상의 원천이 아니다").
- **C vs B**: 크로스 도메인 패러프레이즈 신호를 분리.
- **B/C/D vs A**: 시드 검색 전체의 효과를 분리.

---

## 3. 코드 레이아웃

```
PaperGym/
├── src/papergym/                    # 코어 라이브러리 (pip 패키지)
│   ├── domain.py                    # 7개 ML 도메인 enum + S2 필드 매핑
│   ├── llm.py                       # litellm 기반 provider-무관 LLM 래퍼
│   ├── log.py                       # JSONL 이벤트 로거 (RunLogger)
│   ├── env/                         # 논문-as-환경 (PaperEnv, preparer, sampler)
│   ├── library/                     # 시드 저장소 (seed.py, store.py: 샤딩 FAISS)
│   ├── agents/                      # base / tool_loop + accumulator/paraphraser/synthesizer
│   ├── tools/                       # read / grep / bash (accumulator용 도구)
│   └── execution/                   # ★ 신규: Execution Gym (12개 모듈 + 프롬프트)
│
├── eval/                            # 단계별 루브릭 + 판정자
│   ├── common.py                    # 공유 판정 기계 (AxisScore, judge_axis, cost_summary)
│   ├── seed_quality/                # Stage 1: specificity, grounding
│   ├── retrieval/                   # Stage 2: relevance, lens-aware relevance
│   ├── ideation/                    # Stage 3: 조건 A/B/C/D + novelty/validity/coherence + novelty loop
│   └── execution/                   # Stage 4: 효과성 + faithfulness 오케스트레이션
│
├── scripts/                        # 진입점 (reproduce_paper.sh가 마스터)
├── data/                           # library/ events/ papers_cache/ datasets/ eval/ queries.yaml si_ideas/
├── docker/                         # Dockerfile(accumulator) + Dockerfile.exec(execution gym)
├── docs/                           # REPRODUCE.md(주장 원장) + superpowers/{specs,plans}
├── tests/                          # pytest 유닛 테스트 (~45개 파일, 전부 mock)
│
└── papergym_notool/                # Stage 1 무도구(no-tool) 추출 베이스라인 (별도 프로젝트)
```

### 환경 변수 / 모델 구성 (`.env.examples`)

```
LITELLM_MODEL   = gpt-5                          # 생성자(generator)
JUDGE_MODEL     = anthropic/claude-sonnet-4-6    # 판정자 (생성자와 다른 패밀리 = self-bias 방지)
EMBEDDING_MODEL = text-embedding-3-small         # 임베딩 (1536차원)
S2_API_KEY      = ...                            # Semantic Scholar bulk search (선택)
```

생성자=GPT-5, 판정자=Sonnet 4.6 조합은 모든 오케스트레이션 스크립트에서 **"두 모델이 같으면 즉시 종료"** 하는 가드로 강제된다. LLM이 자기 패밀리를 채점하면 점수가 부풀려지는 알려진 편향을 막기 위함이다.

---

## 4. 코어 인프라

### 4.1 도메인 (`domain.py`)

7개 ML 도메인 enum: `LLM_NLP, MULTIMODAL, CV, RL, IR_REC, SPEECH, ROBOTICS`. 각 도메인은 `DOMAIN_FIELDS`에 Semantic Scholar 검색용 분야 문구 리스트(예: LLM_NLP → "large language model", "instruction tuning", "chain of thought", ...)를 가진다. `DOMAIN_OVERRIDES`는 일부 S2 분야 문자열을 강제 매핑한다(예: "Vision and Language" → MULTIMODAL).

### 4.2 LLM 래퍼 (`llm.py`)

`LLMClient` — litellm 기반 provider-무관 채팅+임베딩 래퍼.
- `litellm.drop_params = True`: 지원하지 않는 파라미터(예: gpt-5의 temperature)를 에러 대신 자동 제거. 하나의 클라이언트로 여러 provider를 다룬다.
- `chat(messages, temperature, response_format)`: 일반 채팅. `response_format={"type":"json_object"}`로 JSON 모드 강제 가능.
- `chat_with_tools(messages, tools, tool_choice)`: 네이티브 tool calling. `ChatReply(content, tool_calls, raw_message, usage)` 반환. `raw_message`는 provider-무관 dict로 히스토리에 round-trip 가능.
- **재시도**: `tenacity`로 `(RateLimitError, APIError, Timeout)`에 대해 6회, exponential backoff(2배, min 10s ~ max 300s).
- **컨텍스트 초과 자가 치유**: `ContextWindowExceededError` 발생 시 시스템 메시지를 제외한 가장 긴 메시지의 중간을 `...[truncated]...`로 잘라 1회 재시도(기본 `truncate_to=10,000`자).
- 누적 토큰 카운터(`total_prompt_tokens` 등)를 유지하여 비용 계산에 사용.

### 4.3 이벤트 로거 (`log.py`)

`RunLogger` — `<run_dir>/events.jsonl`에 append하는 JSONL 로거. `event(type, **f)`와 `message(stage, iter, **f)` 두 메서드를 제공하며 각 줄에 `ts=time.time()`을 박고 즉시 flush(라인 단위 내구성). `default=str`로 enum/Path 등 비직렬화 값도 안전하게 기록. 컨텍스트 매니저 지원.

### 4.4 에이전트 베이스 (`agents/base.py`, `agents/tool_loop.py`)

코드베이스에는 **두 종류의 에이전트 형태**가 있다.

**(a) `PromptLoader`** — 프롬프트 로딩.
- 생성자가 **파일시스템 Path 또는 점 표기 패키지명**(예: `"papergym.agents.accumulator"`)을 모두 받는다. 컨테이너 안/밖에서 같은 코드가 동작하도록 한 설계. 패키지 경로일 때는 `importlib.resources`로 패키지 데이터에서 YAML을 읽는다.
- `render(name, **fields)`: YAML을 파싱한 뒤 고정 역할 순서 `("system", "user")`로 각 값을 **Jinja2 템플릿**으로 렌더링하여 `[{role, content}, ...]` 메시지 리스트를 만든다. Jinja2이므로 `{% for %}` 루프와 `{{ var }}` 치환을 쓸 수 있다(합성기 프롬프트가 시드 리스트를 루프로 렌더링).

**(b) `BaseAgent`** — 단일 호출 JSON 에이전트 (render → chat → json.loads).
- Paraphraser(temp 0.4), Synthesizer(temp 0.5)가 사용.
- `call(**fields)`: 프롬프트 렌더 → `chat(..., response_format={"type":"json_object"})` → `json.loads`. JSON 모드가 보장하므로 펜스 블록 추출 없이 bare object를 파싱. 실패 시 `ValueError`.
- `on_message_hook`으로 각 메시지를 로거에 흘려보낼 수 있다(try/except로 보호 — 로깅이 에이전트를 죽이지 못함).

**(c) `run_tool_loop`** — 공유 ReAct tool-calling 루프 (Accumulator, Execution Gym 에이전트가 사용).
- `dispatch(name, args) -> str`를 호출자가 주입하므로 재사용 가능.
- 매 스텝: `chat_with_tools(tool_choice="auto")` → `raw_message`를 히스토리에 append → tool_calls가 없으면 **자연 종료(natural_end)**, 있으면 각 호출을 dispatch하고 결과를 `{"role":"tool", ...}`로 히스토리에 push.
- **종료 조건**: 자연 종료 / `max_steps` 초과 / **루프 탐지**(동일한 `(tool, args)` 시그니처가 3회 연속이면 `status="error"`). tool 예외는 관측값으로 변환되어 모델이 복구하도록 함(루프를 죽이지 않음).
- `LoopResult(status, final_content, steps, trace, reason)` 반환. status ∈ {`natural_end`, `max_steps_exceeded`, `error`}.

### 4.5 도구 (`tools/{read,grep,bash}.py`)

세 함수 모두 `.schema` 속성(OpenAI function-tool JSON 스키마)을 갖고, 대문자 이름(`Read`, `Grep`, `Bash`)으로 디스패치된다.
- **Read(file_path, offset=0, limit=4000)**: 절대 경로 파일을 읽어 **줄 번호가 붙은** 내용 반환. 잘라내기가 아니라 페이지네이션(`Read(offset=...)`로 이어 읽기).
- **Grep(pattern, path)**: `grep -rn -- <pat> <path> | head -200` (200줄 상한). `set -o pipefail` + `shlex.quote`로 안전화. `cwd=paper_dir`.
- **Bash(command, timeout=120, description)**: `bash -c`를 `cwd=paper_dir`로 실행. `description`은 실행에 쓰이지 않지만 트레이스에 기록. 출력은 `exit=<rc>\nstdout:...\nstderr:...` 형식.
- **샌드박싱의 실체**: 이 레벨의 "샌드박스"는 *디렉터리 스코핑*일 뿐이며 OS 격리는 아니다. 진짜 격리는 바깥의 Docker 컨테이너(Accumulator) 또는 별도의 `execution/sandbox.py`(Execution Gym)가 담당한다.

---

## 5. Stage 1 — Accumulator: 논문에서 시드 추출

### 5.1 "논문-as-환경" (`env/base.py`)

`PaperEnv`는 **34줄짜리 얇은 per-paper 상태 객체**이지 Docker 샌드박스 자체가 아니다(흔한 오해). 컨테이너가 샌드박스이고, `PaperEnv`는 그 안에서 동작한다.
- 필드: `arxiv_id`, `work_root`, `paper_dir = work_root/arxiv_id`, 선택적 `cache_root`.
- `reset()`: `paper_dir` 생성 후 `fetch_paper_to_disk(...)`로 `paper.md`를 만든다(유일한 셋업).
- `close()`: no-op. 컨테이너 폐기는 외부의 `docker run --rm`이 담당.
- **`step()` 메서드는 없다.** 스테핑은 에이전트의 tool loop가 수행한다.
- 코드 레포는 `PaperEnv`/`preparer`가 클론하지 않는다. **에이전트**가 논문에서 "명백히 공식적인" GitHub URL을 발견했을 때만 `Bash`로 `git clone`한다(기본은 논문 단독).

### 5.2 논문 준비 (`env/preparer.py`)

`fetch_paper_to_disk(arxiv_id, root, cache_root)`:
1. `paper.md`가 이미 있으면 그대로 반환(idempotent, resume-safe).
2. `cache_root/<id>/paper.md`가 있으면 **심링크** 후 반환(다운로드·변환 생략 — 프로덕션의 고속 경로).
3. 아니면 임시 디렉터리에서 PDF 다운로드 → 마크다운 변환 → `paper.md` 기록.
- **다운로드**(`_download_pdf`): `arxiv.org/pdf/{id}.pdf`, content-type/length 검증, 429/5xx/타임아웃에 지수 백오프+지터로 재시도(기본 6회).
- **PDF→MD 변환**: 1차로 **docling**(OCR off, 테이블 구조 추출 on), 실패 시 **pymupdf4llm**로 폴백.

### 5.3 논문 샘플링 (`env/sampler.py`, `scripts/sample_envs.py`)

Semantic Scholar bulk search로 도메인별 논문을 샘플링.
- `PaperSearchClient`: bulk 엔드포인트(`/paper/search/bulk`), 429/5xx 백오프, `Retry-After` 준수.
- **도메인별 예산** `DEFAULT_BUDGET`: LLM_NLP=107, MULTIMODAL=89, CV=77, RL=60, IR_REC=60, SPEECH=60, ROBOTICS=47 (합계 500 목표).
- 도메인 필드 문구들을 S2 불리언(`|`)으로 OR 결합, `year_range=(2017,2025)`, 기본 `publicationTypes=Conference`로 서버 측 필터.
- **벤치마크 venue allowlist**(NeurIPS/ICML/ICLR/CVPR/.../ICASSP 등)를 단어 경계 lookaround 정규식으로 매칭하여 필터.
- **층화 그리드 샘플링**(`sample_paper_grid`): (연도 코호트 3 × 인용 티어 3) 그리드로 분할 후 셀별 무작위 추출, 부족분은 인용 수 내림차순으로 채움. `random.Random(seed)`로 재현 가능.
- 산출물: `data/arxiv_ids.jsonl`(446줄) — `{arxiv_id, source_query_domain, year, citations, title, venue}`. PDF는 이 시점에 받지 않는다(컨테이너/preconvert로 지연).

### 5.4 Accumulator 에이전트 (`agents/accumulator/agent.py` + `accumulator.yaml`)

코드베이스에서 유일한 **에이전틱(tool-loop) 에이전트**. 컨테이너 안에서 논문 하나를 탐사하여 1~3개 시드를 추출한다.
- 도구: `Read`/`Grep`/`Bash` (`Grep`/`Bash`는 `functools.partial`로 `paper_dir` 주입).
- 최종 답변은 자유 텍스트이므로 `_extract_json`이 ```json 펜스 블록 또는 최외곽 중괄호에서 JSON을 추출.
- 매우 방어적인 검증: `natural_end`가 아니거나 JSON 파싱 실패 시 빈 형태 `{"title":"", "domain":None, "seeds":[]}` 반환. 도메인이 7개 enum이 아니면 `None`(→ 상위에서 skipped 처리).
- 기본 `max_steps=100`.

**`accumulator.yaml` 시스템 프롬프트의 핵심(verbatim)** — *시드의 정의*:
> You are the Accumulator. Given one paper (markdown + optionally a cloned GitHub repo), do TWO things: 1. Assign the paper to ONE of the 7 ML domains below. 2. Extract 1–3 distinct main contributions as seeds. Each seed captures (a) the underlying problem the paper addresses, and (b) the method that solves it, including what is special / novel about the approach. Avoid splitting one idea into multiple sub-seeds; prefer fewer, higher-quality seeds.

**시드 포맷**:
> problem: 1–2 sentences. The challenge the paper addresses.
> method: 3–5 sentences. The solution AND what is special about it.

**핵심 설계 — 강제 grounding 단계(6번)**: 최종 JSON을 내기 전, 각 시드의 method가 주장하는 구체적 claim 2~3개를 `Grep`/`Read`로 논문에서 *반드시* 검증하고, 근거를 못 찾으면 그 claim/시드를 삭제하도록 강제한다("This step is MANDATORY"). 이것이 grounding 점수를 높게 유지하는 메커니즘이다.

**실제 시드 예시**(`data/library/shard_1`):
```json
{"seed_id": "d3fd461905b7",
 "problem": "Recursive neural networks for sentence encoding typically require supervised parse trees ... learn task-specific, discrete tree compositions directly from plain text while remaining trainable end-to-end.",
 "method": "Introduce Gumbel Tree-LSTM, a bottom-up Tree-LSTM that ... Straight-Through Gumbel-Softmax is used to sample a discrete parent during training (argmax at test) ...",
 "domain": "LLM_NLP",
 "paper_title": "Learning to Compose Task-Specific Tree Structures",
 "paper_id": "1707.02786"}
```

### 5.5 시드 라이브러리 (`library/seed.py`, `library/store.py`)

`Seed` 자료구조: `seed_id`(12-hex uuid), `problem`, `method`, `domain`(enum), `paper_title`, `paper_id`(arxiv id).

`LibraryStore` — **도메인 분할 + 샤딩** 시드 저장소.
- `seeds.jsonl`(메타데이터) + 도메인별 `faiss/<DOMAIN>.index` (`IndexFlatIP`, 1536차원). 임베딩을 L2 정규화하므로 inner product = cosine.
- **`add(seed, embedding)`**: `fcntl.flock(LOCK_EX)`로 advisory 락 + 락 하에서 디스크 재로드(느린 워커가 최신 인덱스를 덮어쓰지 못하게) + 원자적 인덱스 쓰기(`*.tmp` → `os.replace`). 부분 쓰기 크래시는 `_load()`가 FAISS/jsonl 중 더 긴 쪽을 잘라 정합화.
- **`open_merged(root)`**: `shard_*/` 서브디렉터리들을 단일 read-only 인-메모리 뷰로 병합(다운스트림 검색/합성이 사용). 시드/FAISS 개수가 어긋난 도메인은 건너뜀.
- **`retrieve(domain, emb, k)`**: 단일 도메인 파티션 내 top-k, `paper_id` 중복 제거 위해 `k*5` over-fetch.
- **`retrieve_cross_domain(...)` — 시스템의 심장(store.py 197–248)**: 7개 도메인의 모든 시드 벡터를 **하나의 병합 풀**로 합친 뒤, 각 도메인 슬롯마다 (자연 도메인이면 원본 질의, 아니면 그 도메인 패러프레이즈)를 임베딩하여 **전체 라이브러리에 대한** cosine top-k를 뽑는다. **CV 어휘로 쓴 패러프레이즈가 어느 도메인에 사는 시드든 끌어올 수 있다.** 반환은 `(seed, origin_domain=렌즈 슬롯, lens_text=실제 임베딩된 텍스트)` 튜플. 렌즈 텍스트는 합성기/판정자에 전달되어 패러프레이즈 프레임이 검색~사용 사이에서 소실되지 않게 한다.

### 5.6 부트스트랩 오케스트레이션 (`run_accumulator.py` + `accumulate_one.py`)

- **`run_accumulator.py`(호스트)**: `arxiv_ids.jsonl`을 순회하며 논문당 `docker run --rm papergym-accumulator` 컨테이너를 `ThreadPoolExecutor`로 병렬 기동. `--spawn-delay-s`(기본 10s)로 arXiv 버스트 방지.
  - **샤딩**: `_shard_for(id) = int(sha1(id),16) % 4` (고정 `N_SHARDS=4`). 같은 id는 항상 같은 샤드 → 병렬 FAISS 쓰기 충돌 방지.
  - **resume**: `accumulator_log.jsonl`에서 `ok`/`skipped`는 완료로 보고 error는 재시도.
  - 호스트 `src`/`scripts`를 read-only 바인드 마운트하여 리빌드 없이 코드 수정 반영. `papers_cache`(ro), HF 캐시(ro, `HF_HUB_OFFLINE=1`) 마운트.
- **`accumulate_one.py`(컨테이너 엔트리포인트)**: `PaperEnv.reset()` → `Accumulator.run()` → 유효 시드(problem+method 비어있지 않음) 최대 3개를 `Seed`로 만들고 `emb = llm.embed(problem)`(1536d, L2 정규화) 후 `library.add()`. `events.jsonl`을 `/events/<id>.jsonl`로 복사 후 컨테이너 폐기.

### 5.7 두 개의 Dockerfile

| | `docker/Dockerfile` (accumulator) | `docker/Dockerfile.exec` (execution gym) |
|---|---|---|
| 베이스 | python:3.11-slim | python:3.11-slim |
| 의존성 | 전체 `pip install -e .` (docling/litellm/datasets/faiss) | `--no-deps` + sklearn/numpy만 |
| apt | git, curl, GL/X11 라이브러리 | 없음 |
| docling 가중치 | 이미지에 사전 워밍 | 없음 |
| 비밀키 | provider 키 전달 | provider 키 **제거**; `GYM_LLM_URL`만 전달 |
| 엔트리포인트 | `accumulate_one.py` | 없음(ad hoc `python <file>`) |
| 신뢰 모델 | *우리* 에이전트 실행 | *신뢰 불가* 에이전트 작성 코드 실행 |

### 5.8 무도구 베이스라인 (`papergym_notool/`)

Stage 1 ablation을 위한 **별도의 자기완결 프로젝트**. 개념적으로 **Accumulator 에이전트만** 다르다: 전체 `paper.md`를 user 메시지에 넣고 단일 completion으로 `{title, domain, seeds}`를 받는다(ReAct 루프·Docker 없음). 출력 라이브러리 형태가 부모 프로젝트와 동일하여 같은 판정자가 채점 가능. 부모의 `seed_quality_eval.py`가 `A=무도구` vs `C=도구증강` 두 라이브러리를 비교한다.

---

## 6. Stage 2 — Paraphraser & 크로스 도메인 검색

### 6.1 Paraphraser (`agents/paraphraser/paraphraser.yaml`)

연구 **문제 진술**을 7개 도메인의 **문제 진술(해법 아님)**로 재구성한다. `BaseAgent`(temp 0.4) 단일 호출.

**핵심 제약(verbatim)**:
> Use PROBLEM vocabulary: "X is slow when ...", "Y fails when ...". DO NOT use SOLUTION vocabulary: NO "use ... to ...", NO "build ... that ...". DO NOT name a specific algorithmic *solution* ... The paraphrase MUST end as a problem statement, not a solution prescription.

2단계 절차: (1) 질의에서 **실패 모드(failure mode)**를 추출하여 한 문장 "essence"로 정제, (2) 각 도메인의 자연 어휘(3~5개 기술 용어)로 유사한 실패 모드를 기술. 도메인별 어휘 앵커(예: IR_REC → candidate pool, reranking, recall@k; SPEECH → phoneme, spectrogram, WER)를 제공. 진짜 유사물이 없는 도메인은 `null`을 낼 수 있다. 프롬프트에는 speculative decoding을 예로 든 GOOD/BAD 워크드 예시가 포함된다.

출력 스키마: `{essence, paraphrases: {LLM_NLP: ..., ..., ROBOTICS: null}}`.

### 6.2 Stage 2 평가 (`eval/retrieval/`)

동일한 시드 예산으로 **패러프레이즈 ON vs OFF** 두 검색을 비교.
- **ON**: `retrieve_cross_domain` (조건 C와 동일). 도메인별 임베딩으로 전역 top-k.
- **OFF**: 원본 질의를 1회 임베딩하여 전체에서 전역 top-(k×7). 도메인 분할·패러프레이즈 없음.

각 시드에 두 relevance 점수: **naive**(질의+시드만, 도메인 라벨 의도적 은닉)와 **lens_aware**(검색에 쓰인 렌즈 텍스트 추가). 렌즈가 원본 질의와 같으면 naive를 재사용(중복 호출 절약); 흥미로운 케이스는 ON 시드가 *패러프레이즈* 렌즈로 검색됐을 때 두 점수가 갈리는 경우다.

**핵심 지표 `seed_home_coverage`**: 검색된 시드가 실제로 몇 개의 도메인에서 왔는지(7개 중). 크로스 도메인 검색은 여러 도메인에 퍼지고, 단일 질의 검색은 1~2개에 집중된다는 주장을 측정 가능하게 만든 것.

**실제 수치**(`retrieval_20260514T190118`, 3 질의): ON seed_home_coverage 평균 **7**(모든 질의 7/7), OFF 평균 **5** → ON이 도메인 커버리지를 넓힘을 확인.

---

## 7. Stage 3 — Synthesizer & 조건 A/B/C/D

### 7.1 Synthesizer (`agents/synthesizer/synthesize.yaml`)

여러 검색 시드로부터 질의를 푸는 새 방법을 합성하되, 각 차용을 출처 명시한다. `BaseAgent`(temp 0.5). 각 시드는 **"Retrieved via" 렌즈**(원본 질의 또는 도메인 패러프레이즈)와 함께 들어온다.

**시스템 프롬프트 핵심(verbatim)**:
> ... it arrives with a "Retrieved via" lens ... Use the lens as a hint about WHY this seed was considered; the paraphrase lens often reveals an analogy ... the raw query alone would not have surfaced.

출력 스키마:
```json
{"method": "<3–5 문단>",
 "rationale": "<1–3 문단>",
 "inspired_by": [{"seed_id": "...", "domain": "...", "borrowed_aspect": "<무엇을 빌렸는지 짧은 명사구>"}]}
```

**출처 규칙**: 메커니즘이 실제로 기여한 시드만 `inspired_by`에 넣고, 시드가 뒷받침하지 않는 단계(글루 로직, 문제별 적응, 새 추가)도 필요하면 포함하되 패딩 금지.

**조건별 차이는 코드가 아니라 전달되는 `lenses` 텍스트만으로 결정된다.** 합성기 자체는 조건을 분기하지 않는다. (evaluate.py 78행 = B의 `[query]*N`, 109–110행 = C의 패러프레이즈 렌즈, 148–151행 = D의 `"(uniform random sample of the library)"`).

### 7.2 조건 구현 (`eval/ideation/evaluate.py`)

- `run_condition_a`: `ideation_direct` 프롬프트로 시드 없이 직접 제안. `inspired_by=[]`.
- `run_condition_b`: 자연 도메인 없으면 `None` 반환. 원본 질의 임베딩으로 자연 도메인 내부 top-21, 렌즈=원본 질의.
- `run_condition_c`: `Paraphraser.run` → `retrieve_cross_domain` → 렌즈=각 시드를 찾은 텍스트 → `Synthesizer.run`.
- `run_condition_d`: 전체 풀에서 `random.Random(0)`으로 21개 무작위 추출. 렌즈로 "무작위 표본임"을 합성기에 알림(C-vs-D 대비가 "출처 공개" 차이로 환원되지 않도록).

---

## 8. 평가 시스템 (eval) — 과학적 핵심

### 8.1 공유 판정 기계 (`eval/common.py`)

- **가격 상수**(1M 토큰당): `GPT5_PRICING=(1.25, 10.0)`(생성), `SONNET46_PRICING=(3.0, 15.0)`(판정).
- **`CostSnapshot`/`cost_summary`**: LLM 누적 토큰 카운터의 델타로 비용 계산. 판정 전용 스크립트는 gen 필드를 0으로 둠. 모든 `summary.json`에 `cost_total` 블록 임베드.
- **`_SCORE_RE`**: `^\s*\*{0,2}\s*Score\s*...[:：]\s*...([1-5])` — **줄 앵커(MULTILINE)**라서 추론 산문 속 "Score 4 앵커" 인라인 언급은 잡지 않고 *줄 시작* `Score:`만 점수로 인정. **[1-5]로 제한**되어 범위 밖 출력은 파싱 실패(점수 0 sentinel)로 떨어진다. 마크다운 볼드(`**`)와 전각 콜론(`：`) 허용.
- **`parse_axis`**: 매치 없으면 `AxisScore(0, ...)`(0 = 파싱 실패 sentinel, 집계 시 결측 처리). 여러 개면 **마지막** Score가 이김.
- **`judge_axis`**: 프롬프트 렌더 → `chat(temperature=0.0)`(결정론적 판정) → `parse_axis`. 대부분의 단일 축 판정자가 이 얇은 래퍼.

### 8.2 Stage 1 판정자 (`eval/seed_quality/`)

각 시드를 두 축으로 **별도 completion**에서 채점.
- **specificity** (시드만 봄): method 텍스트의 **구체적 기술 엔티티 밀도**. 6개 카테고리(명명된 알고리즘/데이터셋/하이퍼파라미터+값/아키텍처 컴포넌트/수식·기호/절차 단계). 점수 5는 "8+ 엔티티 AND ≥2 distinct 카테고리, 그중 명명 알고리즘·하이퍼파라미터값·수식 중 하나 이상 포함; 절차 동사만으로는 불가". 동점 시 기본 하향.
- **grounding** (전체 논문 봄): 시드의 명명 엔티티가 (다른 표현이라도) 논문에 *존재하는가*. 환각(어디에도 없음) vs 발명된 표현을 구분. "대부분의 잘 만들어진 시드는 4점에 안착", claim-fidelity 체크는 **4-vs-5 경계에서만** 적용.
- **shuffled 음성 대조**(`seed_shuffled.py`): 시드를 *다른* 논문과 짝지어 grounding만 재판정. 판정자가 정말 논문을 읽는다면 ~5에서 1-2로 붕괴해야 함. **실제**: A true 4.849 → shuffled **1.0**(100% 시드가 하락), C true 4.858 → shuffled **1.0**. 판정자가 실제로 읽고 있음을 검증.

**실제 Stage 1 수치**(`20260502T230414`, n=30 paper-level mean): specificity **A 4.217 → C 4.756**(도구 증강이 더 구체적), grounding **A 4.822 / C 4.817**(둘 다 높게 유지). 이것이 논문의 Stage 1 핵심 주장.

### 8.3 Stage 3 판정자 (`eval/ideation/`)

7개 판정 함수. novelty/validity/coherence는 조건마다, 나머지는 전용 스크립트로.

- **novelty** — **ReAct 멀티라운드 판정자**. 각 라운드에서 Semantic Scholar 검색(`Query:`) 또는 점수 확정(`Score:`) 중 하나. 정적 컨텍스트는 시스템 메시지에, 라운드별 검색 결과만 user 메시지에 → 토큰 비용 선형. 같은 응답에 Query와 Score가 섞이면 **검색 라운드로 처리**(1라운드 셀프 확정 방지). 판정자는 ideation 자신의 검색 시드도 봐서 "입력을 그대로 재탕한 제안"을 탐지. "**HARSH critic**" 페르소나, 점수 5는 "검색에 직접 클론이 없는 새 메커니즘". 기본 `max_rounds=10`.
- **validity** — 단일 호출. method의 메커니즘이 질의의 실패 모드를 **추적 가능하게(traceable)** 다루는가. 5 = 문제→방법 인과 경로를 빈틈없이 추적 가능.
- **coherence** — 단일 호출. 방법의 단계들이 내부적으로 일관되고, 메커니즘이 호환되며, 가정이 달성 가능하고, 실무자가 빠진 컴포넌트를 발명하지 않고 구현 가능한가.
- **pairwise (novelty/validity/coherence)** — 두 방법을 나란히 비교, **위치 무작위화**(swap)로 위치 편향 방지. `Winner: A|B|tie` 파싱, 파싱 실패는 tie 기본. `condition_for_a/b`로 원래 조건으로 역매핑.
- **method_specificity** — Stage 1 specificity와 동일 구성(6 카테고리)이나 긴 ideation 텍스트용으로 컷오프 상향.
- **inspired_by_grounding** — **유일한 1–3 척도**. 합성 방법이 인용 시드의 `borrowed_aspect`를 실제로 통합했는가(1=장식적, 2=부분, 3=완전 통합으로 제거 시 방법이 바뀜). 자기 편향 방지 위해 held-out 판정자.

**실제 Stage 3 수치 발췌**:
- per-condition novelty: A 3.97 / B 4.07 / C 4.13 / D 4.13 (REPRODUCE.md 원장 기준)
- coherence(`coherence_per_cond_20260515T130228`, 30 질의): A 3.633 / B 3.300 / C 3.467 / D 3.367
- pairwise coherence(`coherence_20260515T125751`): C_win 18 / D_win 12 → **C 0.6 vs D 0.4**
- inspired_by grounding(`grounding_20260515T131746`): C n=195 mean 2.769 fraction_full 0.779; D n=196 mean 2.643 fraction_full 0.679 → **C 인용이 D(무작위)보다 더 잘 통합됨**

### 8.4 Novelty Loop (`eval/ideation/evaluate_novelty_loop.py`) — 부록 전용

조건 C(또는 D)를 **반복적 참신성 정련 루프**로 감싼다(메인 파이프라인은 건드리지 않음).
- 라운드 1 = 표준 `run_condition_c`. 이후 라운드는 같은 시드 재사용.
- 판정자(`judge_novelty_with_priors`)는 확정 시 **`Prior works:` 블록**(S2가 surface한 충돌 선행연구 목록)을 추가로 낸다.
- 이 목록을 피드백 메시지로 합성기에 되먹임("이 메커니즘 조합을 피해 더 참신하게 다시 합성하라").
- **수렴 기준**: novelty score ≥ 4(기본)면 break. 최대 10라운드.
- 즉, **참신성 판정자가 실제 문헌을 검색 → 충돌 선행연구 surface → 그 조합을 피하라고 합성기에 지시 → 재합성 → 재판정**의 피드백 사이클.

**실제 loop 수치**(`loop_benchmark_20260506T193707`, n=30/30): C/D 모두 **100% 수렴**, 거의 항상 1라운드(C 평균 1.07, D 1.13). 최종 평균 점수 4.1(둘 다). 차별점은 **도메인 커버리지**: C 평균 3.87(min2 max6) vs D 평균 3.33(min1 max6). 루프 후 pairwise: novelty 50/50, validity C0.40/D0.30/tie0.30, coherence C0.467/D0.533. 즉 루프가 둘 다 참신성 임계로 끌어올리면 raw novelty는 비슷해지고, 검색의 우위는 **grounding/커버리지**에서 드러난다.

---

## 9. Stage 4 — Execution Gym (신규 서브시스템)

`src/papergym/execution/`(12개 모듈) + `eval/execution/` + 3개 스크립트. 설계 문서(`docs/superpowers/specs/2026-06-07-execution-gym-design.md`)와 3단계 plan(foundation, hardening, hardening-p16)을 따라 구축되었다.

### 9.1 개념과 동기

Si et al.이 보인 idea-execution gap을 **추론 시점 실행 자동화**로 대규모 재현하려는 것. ideation 파이프라인(시드 라이브러리+합성)이 *아이디어를 만들고*, 게임은 *그것을 실행·채점*한다. 기존 PaperGym 프리미티브(`LLMClient`, `run_tool_loop`, `judge_axis`, `cost_summary`, 타임스탬프 run-dir 관례)를 재사용한다.

### 9.2 자료 모델 (`types.py`, `task.py`)

- **`IdeaSpec`**: 실행할 아이디어 제안서(`idea_id`, `condition` ∈ AI/Human/AI_Rerank, `topic`, `title`, `proposal_text`, `human_exec_scores`).
- **`RunArtifact`**: 한 에이전트 실행 산출(`status`, `code`, `stdout`, `predictions`, `steps`, `trace`, `error`).
- **`ExecResult`**: 최종 결과(`baseline_metric`, `method_metric`, `effectiveness`, `faithfulness_score`, `cost`, `leakage_flags`, `sandbox`, `trustworthy`). **불변식: 실패한 실행은 `None`이지 `0.0`이 아니다**("유출된 100%/실패는 결측이지 0점이 아니다").
- **`Task`**: 고정 데이터 + 베이스라인 + 객관적 지표(서로 다른 아이디어를 비교 가능하게). split-aware(dev/test), `inputs(split)`은 라벨 없는 뷰만 에이전트에 전달.
  - `GSM8KAccuracyTask`: GSM8K 최종 정수 정확도. dev/test 50줄씩 disjoint.
  - `GeneratedArithmeticTask`(**기본 task**): 호스트 생성 라벨의 합성 산술 문제 → "공개 데이터셋 조회로 정답 복원 불가"(라벨 유출 차단).

### 9.3 실행 에이전트 (`agent.py`)

`run_tool_loop`을 4개 도구로 구동: `WriteFile`, `RunPython`(샌드박스에서 실험 실행), `ReadFile`, `Finish`. 작성된 모든 `.py`를 수집(`import cheat`에 숨긴 유출 탐지용). 시스템 프롬프트가 모든 하드닝 제약을 인코딩: "LLM은 오직 `gym_client.metered_llm_call`로만 호출, datasets/openai/anthropic/litellm/papergym.llm 임포트 금지, 데이터셋 다운로드/테스트 정답 읽기 금지, dev.json으로 개발하고 test_inputs.json을 예측, 모델 훈련 금지". 에이전트의 플래너 호출도 `BudgetedLLM`을 통해 metering된다.

### 9.4 샌드박스 (`sandbox.py`)

`Sandbox` Protocol. 공유 **경로 감옥(path jail)** `_resolve_in`이 절대 경로·`../` 탈출·심링크 탈출을 거부.
- **`LocalSandbox`**: dev/test 전용, OS 격리 아님. provider 키를 strip한 환경(`_sandboxed_env`)에서 subprocess 실행. **`trustworthy=False`로 표시**.
- **`DockerSandbox`**: 진짜 격리. `--network=bridge`, provider 키 **절대 미전달**(`GYM_LLM_URL`/`GYM_JOB_TOKEN`만). 컨테이너 내부 `metered_llm_call`이 호스트 프록시에 닿도록 `host.docker.internal` 재작성 + `--add-host`. **`trustworthy=True`**.

### 9.5 비용·키 격리 (`gym_client.py`, `llm_proxy.py`, `metering.py`)

원칙: **provider 키는 호스트에만, 샌드박스는 HTTP URL만 본다.**
- `UsageMeter`: 토큰/비용 누적, `budget_usd` 강제(`BudgetExceeded`), source별 귀속(`by_source`), thread-safe(프록시가 `ThreadingHTTPServer`).
- `BudgetedLLM`: 에이전트 플래너 호출도 같은 예산에 부과.
- `llm_proxy.run_proxy`: 호스트 측 HTTP 서버. 200(성공)/429(예산초과)/500(에러).
- `gym_client.metered_llm_call`: 샌드박스 내 유일한 LLM 경로. `GYM_LLM_URL`에 POST.

### 9.6 채점 (`scorer.py`, `integrity.py`, `executability.py`, `faithfulness.py`)

- **`score_effectiveness`**: `(method_metric, effectiveness, leakage_flags)` 반환. 유출 플래그나 형식 오류나 예측 없음이면 `(None, None, flags)` — **0점이 아니라 결측/의심으로 표시**. 정상이면 `effectiveness = method_metric - baseline_metric`.
- **`integrity.scan_for_leakage`**: 정적 정규식 스캔(load_dataset/openai 임포트/raw 네트워크/dynamic-import/subprocess/pip install 등 금지). **방어선 보강용일 뿐 회피 가능**임을 plan이 명시; 진짜 방어는 "샌드박스에 라벨 없음" + Docker egress 차단(수동 OPS).
- **`executability`**: 아이디어가 추론 전용(GPU 훈련·인간 평가 없이 공개 데이터+자동 지표)으로 실행 가능한지 분류.
- **`faithfulness`**: 실행 코드가 제안 방법을 충실히 구현했는가(1~5). "나쁜 결과가 아이디어 탓인가 실행 탓인가"라는 핵심 교란요인을 다룸.

### 9.7 현재 상태와 한계

- **Docker 실행만 `trustworthy=True`.** Local은 dev 전용.
- **네트워크 egress 차단(R5)은 코드가 아닌 수동 OPS 단계.** 임의의 "100%"는 의심 대상.
- **정적 가드는 회피 가능**(방어선 보강용).
- **VRN(Verifiable Recombination Novelty, 출처 기반 참신성 가드), dual-reward, gap 재현 분석, test-time feedback, RL은 모두 후속 단계로 연기됨.** 현재 게임은 effectiveness + faithfulness만 채점한다.
- `data/si_ideas/`는 채워져 있으나 아직 전체-43 파이프라인의 정규 형태는 아님.

---

## 10. 스크립트 & 재현

### 10.1 마스터 스크립트 (`scripts/reproduce_paper.sh`)

논문 결과 전체(Stage 2/3 + loop)를 순서대로 재실행. `set -euo pipefail`, `.env` 소스, `JUDGE_MODEL`/`LITELLM_MODEL`/`EMBEDDING_MODEL` 필수.
1. **Stage 2**: `retrieval_eval.py`
2. **Stage 3**: `ideation_eval.py --conditions A,B,C,D` → 그 산출 `evaluations.jsonl`을 4개 후속 판정자(`ideation_layers_eval`, `coherence_pairwise_eval`, `coherence_per_condition_eval`, `inspired_by_grounding_eval`)가 재사용(재생성 없음).
3. **부록 C**: `loop_benchmark.py`
- 끝에 인라인 Python으로 모든 `summary.json`/`aggregate.json`의 `cost_total`을 합산하여 총비용·총시간 출력.
- Stage 1(`seed_quality_eval`, `seed_shuffled`)은 무도구 라이브러리와 함께 **수동 실행**(마스터에 미포함).

### 10.2 주요 스크립트

- `ideation_eval.py`: Stage 3 메인. 조건별 novelty/validity/coherence 채점. `summary.json`에 `{cond:{axis:_stat}}` + 히스토그램.
- `retrieval_eval.py`: Stage 2. ON vs OFF, `seed_home_coverage`.
- `seed_quality_eval.py` / `seed_shuffled.py`: Stage 1 비교 + 음성 대조. paper-level 집계(다작 논문 과대가중 방지).
- `ideation_layers_eval.py`: pairwise 승률 + method specificity.
- `coherence_pairwise_eval.py` / `coherence_per_condition_eval.py`: coherence pairwise / per-condition.
- `inspired_by_grounding_eval.py`: 인용 통합도(1–3), seed_id 중복 제거.
- `loop_benchmark.py`: novelty 반복 루프 ablation(C vs D), 수렴 히스토그램, pairwise. 하드코딩: `NOVELTY_THRESHOLD=4`, `MAX_ROUNDS=10`, `PAIRWISE_RNG_SEED=42`.
- `run_synthesis.py`: 단일 질의 조건-C 빠른 시작(판정 없음).
- `fetch_si_ideas.py`: Si et al. 아이디어를 gdown으로 받아 인간 실행 점수를 조인 → `data/si_ideas/<id>.json`(Execution Gym 입력).

### 10.3 `data/queries.yaml` — 30개 벤치마크 질의

각 항목: `id`(Q01–Q30), `type`, `natural_domain`, `expected_paraphrase_domains`, `text`. 3개 타입:
- **primary(5)**: Q01–Q05, 추론 효율(speculative decoding, visual-token pruning, KV-cache 압축 등).
- **domain_specific(7)**: Q06–Q12, 단일 도메인 베이스라인(패러프레이즈 이득 미기대).
- **cross_domain(14)**: Q13–Q30, 패러프레이즈 스트레스 테스트(다도메인 전이 기대).

자연 도메인 분포는 LLM_NLP 우세(~20), 그 다음 MULTIMODAL/CV/RL 순. N=30은 paired ablation으로 d≥0.5를 α=0.05, power 0.8에서 탐지하도록 설계.

---

## 11. 데이터 산출물 & 실제 통계

- **시드 라이브러리**(`data/library/`): **총 1,167 시드 / 446 논문**, 4개 샤드(269/288/327/283).
  - 도메인 분포: LLM_NLP 245, MULTIMODAL 219, ROBOTICS 215, CV 183, IR_REC 149, RL 89, SPEECH 67.
- `data/events/`: 논문당 1개 Accumulator 트레이스(446개).
- `data/papers_cache/`: 논문당 `paper.md`(447 디렉터리, 메타데이터 파일 없음).
- `data/datasets/gsm8k_accuracy/`: dev/test 50줄씩.
- `data/eval/`: 타임스탬프 run 디렉터리들(Stage별 + smoke).
- `data/si_ideas/`: Si et al. 아이디어 폼(Human/AI/AI_Rerank).

비용 감각(실제 run): shuffled grounding이 가장 비쌈($17, 판정 토큰 5.46M), loop benchmark $14+14+1, 총 wall ~3.2h.

---

## 12. 설계 원칙 / 엔지니어링 하이라이트

1. **Self-bias 가드**: 모든 오케스트레이터가 판정자=생성자면 종료. 결정론적 판정(temp 0.0).
2. **파싱 실패 = 점수 0 sentinel**, 집계에서 결측 처리. 모든 곳에 정수 빈 히스토그램(평균이 숨기는 분포 인플레이션 노출).
3. **동점 시 기본 하향** + "장황함/인상적임에 보상 금지" + 축 간 격리 — 일관된 anti-inflation 입장.
4. **시드 예산 패리티**(B/C/D 21개 공유)로 조건 비교 공정성.
5. **렌즈를 검색~합성~판정까지 보존**하여 패러프레이즈 프레임 소실 방지.
6. **크래시·동시성 안전**: `fcntl` 락 + 디스크 재로드 + 원자적 FAISS 교체, 고정 해시 샤딩.
7. **provider-무관 LLM**(litellm `drop_params`) + 컨텍스트 초과 자가 절단 + tenacity 재시도.
8. **Execution Gym의 다층 신뢰 모델**: 키 격리(프록시), 예산 강제(metering), 라벨 호스트 전용, 정적 유출 스캔, Docker 격리, `trustworthy` 플래그.
9. **재현성 시드 고정**: D 풀 seed=0, pairwise 위치 무작위화 seed=42.
10. **resume-safe** 전반: `paper.md` 멱등, accumulator_log 기반 재시도, 증분 JSONL 쓰기.

---

## 13. 핵심 파일 빠른 참조

| 관심사 | 파일 |
|---|---|
| 도메인 정의 | [src/papergym/domain.py](src/papergym/domain.py) |
| LLM 래퍼 | [src/papergym/llm.py](src/papergym/llm.py) |
| 크로스 도메인 검색(심장) | [src/papergym/library/store.py](src/papergym/library/store.py) (`retrieve_cross_domain`, 197–248행) |
| Accumulator 프롬프트(시드 정의) | [src/papergym/agents/accumulator/accumulator.yaml](src/papergym/agents/accumulator/accumulator.yaml) |
| Paraphraser 프롬프트 | [src/papergym/agents/paraphraser/paraphraser.yaml](src/papergym/agents/paraphraser/paraphraser.yaml) |
| Synthesizer 프롬프트(렌즈) | [src/papergym/agents/synthesizer/synthesize.yaml](src/papergym/agents/synthesizer/synthesize.yaml) |
| 조건 A/B/C/D | [eval/ideation/evaluate.py](eval/ideation/evaluate.py) (78, 109–110, 148–151행) |
| 공유 판정 기계 | [eval/common.py](eval/common.py) |
| Novelty loop | [eval/ideation/evaluate_novelty_loop.py](eval/ideation/evaluate_novelty_loop.py) |
| Execution Gym 설계 | [docs/superpowers/specs/2026-06-07-execution-gym-design.md](docs/superpowers/specs/2026-06-07-execution-gym-design.md) |
| 마스터 재현 스크립트 | [scripts/reproduce_paper.sh](scripts/reproduce_paper.sh) |
| 주장 원장 | [docs/REPRODUCE.md](docs/REPRODUCE.md) |

---

## 14. 종합 평가

PaperGym은 잘 짜여진 **연구용 리서치-아이디에이션 벤치마크/시스템**이다. 가장 독창적인 기여는 (1) **"논문 = 에이전트가 도구로 탐사하는 환경"** 추상화와 (2) **"질의를 7개 도메인으로 패러프레이즈하여 전역 검색"** 하는 크로스 도메인 메커니즘 전이다. 평가 설계가 매우 견고하다: 음성 대조(shuffled grounding), self-bias 가드, 위치 무작위화, 시드 예산 패리티, ReAct 기반 라이브 문헌 검색 novelty 판정 등 LLM-judge 신뢰성 위협을 체계적으로 통제한다.

Execution Gym은 가장 야심차고 미완성인 부분으로, "ideation 점수는 실행 앞에서 무너진다"는 Si et al.의 발견에 대한 직접적 대응으로 **아이디어를 실제로 실행해 효과를 측정**하려 한다. 키 격리·예산 강제·라벨 격리 등 보안/무결성 설계는 성숙하나, 네트워크 egress 차단이 아직 수동이고 novelty 가드(VRN)·dual-reward·RL은 미구현 상태다. 즉 메인 3단계 파이프라인은 완성·평가 완료, Execution Gym은 P1 기반(effectiveness+faithfulness)까지 동작하는 진행형 단계다.
