# 연구 계획: 온라인 VAD 예측을 위한 Causal Speech Representation 학습

> 최종 수정: 2026-08-08
> 관련 문헌: [docs/references/](docs/references/) 3편 (MSP-Conversation, Transcription Policy/CrisperWhisper 2.0, VibeVoice-ASR)

## 1. 연구 목표

MSP-Conversation 데이터셋에서 **모든 타임스텝(10Hz)에 대해 valence/arousal/dominance(VAD)를 온라인으로 예측**하는 실시간 모델을 개발한다. 모델은 각 시점 t에서 과거 오디오만 사용하는 causal 구조여야 한다.

## 2. 논문 전개 (내러티브)

논문의 논리 사슬은 실험 순서와 일치한다:

1. **[실험 1]** WavLM을 causal로 변환해 VAD 단일 태스크로 학습하면 MSP-Conversation 논문의 bidirectional 베이스라인 대비 성능이 크게 하락한다. 이는 미래 컨텍스트 상실에 따른 구조적 비용이다.
2. **[가설]** 이 하락의 상당 부분은 VAD 단일 로스만으로는 causal representation이 감정 예측에 필요한 정보를 충분히 인코딩하지 못하기 때문이다. → Joint loss로 representation을 보강해야 한다.
3. **[실험 2 · 근거]** frozen 백본 3종(VibeVoice-ASR, WavLM, TDNN 화자모델) feature 결합 파일럿에서 bidirectional 베이스라인을 상회하는 결과를 이미 확보 → 감정 예측이 **content 정보와 화자 정보에 의존**함을 보이는 진단 실험으로 재배치한다. (백본 융합은 최종 시스템이 아님)
4. **[실험 3 · 본실험]** 실험 1의 causal 모델에 **ASR + 감정(paralinguistic) 태그 verbatim ASR + 화자 예측 + VAD 조인트 로스**를 적용하면 실험 1 대비 성능이 회복/향상됨을 보인다.

핵심 주장: *"인과성 제약이 잃게 만드는 정보를, 보조 태스크 supervision이 representation 안에 되돌려 넣는다."*

성공 기준 (2단계):
- **최소**: 조인트 로스 causal 모델 > VAD 단일 causal 모델 (실험 1), 전 속성에서 유의미한 차이
- **목표**: 조인트 로스 causal 모델 ≥ bidirectional 베이스라인 (Test1 CCC: valence 0.676 / arousal 0.669 / dominance 0.517, LSTM ST 기준)

## 3. 사전 조사 결과 (2026-08-08, HuggingFace/GitHub)

### 3.1 Causal WavLM 기존 변환 모델 — **없음, 직접 변환 필요**

| 후보 | 판정 |
|---|---|
| `ZhuoyanTao/causal-wavlm-turn-taking` | ESPnet 코드 덤프(문서/가중치 없음, 0 dl). 단, `espnet2/asr/frontend/causal_s3prl.py`는 S3PRL 계열 causal 변환 **코드 참고자료**로 활용 가능 |
| `codekansas/causal-hubert` | 빈 카드, 개인 실험 수준. 사용 불가 |
| `Adnan256/streaming-target-stno-wavlm-base` | Target-speaker voice activity 전용, CC-BY-NC. 목적 불일치 |

→ **결론: causal 변환은 직접 수행한다** (4.2절). 공개된 변환 모델이 없다는 점 자체가 기여의 여지.

### 3.2 Verbatim(감정 태그) 데이터/모델 — **모두 공개되어 있음**

- **모델**: `nyralabs/CrisperWhisper2.0_{large,medium,small,turbo}` — verbatim/intended 모드 전환, verbatimize, 단어 타이밍, longform(conditional continuation) 지원. 의사 라벨 생성기로 사용.
- **데이터**: `nyralabs/disfluency_speech_english` — 4,957 발화 / 9.4h, **train 4,458 / val 250 / test 249**, `verbatim_transcript` + `intended_transcript` 페어 제공. **실제 라벨 데이터이므로 보조 태스크 학습에 디스틸 없이 직접 사용 가능.**
- `nyralabs/disfluency_speech_german` — 독일어 test-only (본 연구에는 불필요, 참고용)

→ **결론: 하이브리드 전략.** (a) disfluency_speech_english train split을 verbatim ASR 보조 태스크의 실라벨 데이터로 사용 + (b) MSP-Conversation 오디오 자체에는 CrisperWhisper 2.0으로 verbatim 의사 라벨 생성(동일 오디오에 멀티 로스를 걸기 위해 필요). (a)만으로는 도메인(단일화자 스튜디오 vs 다자 팟캐스트)이 달라 (b)가 본선이고 (a)는 보강재.

### 3.3 베이스라인 재현 자산

- `tiantiaf/wavlm-large-msp-podcast-emotion-dim` — WavLM-large를 MSP-Podcast로 V/A/D 파인튜닝(Vox-Profile/SAILER 계열). MSP-Conversation 논문 베이스라인의 1단계 인코더와 유사 → **베이스라인 재현 시 초기화 후보**. 논문 저자의 정확한 체크포인트가 아니므로 재현 수치가 다를 수 있음을 명시할 것.
- `microsoft/wavlm-large` — 원본 백본.

### 3.4 데이터셋 접근

- **MSP-Conversation v2.0**: lab-msp.com에서 학술 라이선스로 배포. Train 41.4h / Dev 12.6h / Test1 14.3h / Test2 9.2h(라벨 비공개, 리더보드 제출). 10Hz 감정 트레이스(반응지연 3초 보정 완료), 수작업 diarization 포함.
- **MSP-Podcast v2.0**: 베이스라인 인코더 파인튜닝 재현에 필요할 수 있음(동일 라이선스 절차).

## 4. 실험 설계

### 4.0 실험 0 (준비): 베이스라인 재현

- 논문 레시피 그대로: WavLM-large(MSP-Podcast 파인튜닝) → 120ms/100ms mean-pool(10Hz) → FFN(512×3, ReLU, dropout 0.5) → BiLSTM(2층, hidden 256) 헤드, CCC loss, 1분 세그먼트(5초 오버랩), Adam lr 1e-4, batch 64.
- 목적: 우리 구현으로 논문 수치(Test1 LSTM ST: V 0.676 / A 0.669 / D 0.517)에 근접함을 확인 → 이후 모든 비교의 신뢰 기반.

### 4.1 실험 1: Causal 변환과 인과성 비용의 격리

**변환 대상 (WavLM-large 내부의 미래 정보 누수 지점 3곳):**

1. **Self-attention**: causal mask 적용. WavLM의 gated relative position bias는 mask와 호환됨.
2. **Convolutional positional embedding** (kernel 128, symmetric padding): 대칭 패딩이 미래를 봄 → **causal padding으로 교체**(왼쪽 전체 패딩) 후 파인튜닝으로 적응.
3. CNN feature extractor: 수용장이 ~25ms로 사실상 무시 가능(문서에 명시만).

**격리 원칙 (가장 중요):** 실험 1의 하락이 "인과성 때문"이라는 주장이 성립하려면, causal 모델과 bidirectional 재현 모델이 **attention mask/conv padding 외에 모든 것이 동일**해야 한다 — 동일 초기화, 동일 파인튜닝 데이터(MSP-Podcast), 동일 VAD 헤드(단, BiLSTM→uni-LSTM 또는 causal Transformer로 교체), 동일 하이퍼파라미터, 동일 평가 프로토콜. 비교표에 "변경점: causality만"을 명시한다.

**예측 헤드**: causal Transformer decoder(예: 6층, 8헤드, causal mask) — 최종 모델과 동일 구조를 실험 1부터 사용해 헤드 구조 confound 제거.

**추가 분석(옵션)**: lookahead 허용량 sweep(0s/0.5s/1s/2s/∞)으로 인과성 비용 곡선 제시.

### 4.2 실험 2: 파일럿 재정리 (진단 실험)

이미 확보된 결과를 논문용으로 재구성:

- frozen VibeVoice-ASR / WavLM / TDNN feature의 **단독 및 조합 ablation** (7가지 조합) — "feature 수 증가" confound에 대한 방어.
- **속성별 기여 분해**: content feature(VibeVoice-ASR)가 valence에, 화자 feature(TDNN)가 dominance에 특히 기여하는지 확인. 이 대응이 나오면 3→4 단계(어떤 보조 로스를 쓸 것인가)의 논리가 완결된다.
- 결론 문장: "감정 트레이스 예측에 필요한 정보 중 VAD 단일 로스가 포착하지 못하는 성분이 content·화자 표현에 존재한다."

### 4.3 실험 3 (본실험): Joint Loss 학습

**모델**: 실험 1의 Causal WavLM + 멀티 헤드.

```
CausalWavLM (50Hz features)
 ├── [VAD head]     10Hz pool → causal Transformer → 3dim 회귀, CCC loss
 ├── [ASR head]     CTC, intended 전사 타깃 (content supervision)
 ├── [vASR head]    CTC, verbatim 전사 타깃 — [UH],[UM],[laughter] 등
 │                  paralinguistic 태그를 단일 토큰으로 포함 (CrisperWhisper 2.0 규약)
 └── [SPK head]     frame-level 화자 supervision (diarization 기반)
```

**로스**: `L = w_vad·L_CCC + w_asr·L_CTC + w_vasr·L_CTC + w_spk·L_spk`, 가중치는 Dev에서 탐색(uncertainty weighting 등 자동화 기법은 옵션).

**타깃 소스**:

| 태스크 | 타깃 | 소스 |
|---|---|---|
| VAD | 10Hz V/A/D 트레이스 | MSP-Conversation 정답 라벨 |
| ASR (intended) | 정제 전사 | CrisperWhisper 2.0 intended 모드 의사 라벨 |
| vASR (verbatim) | 태그 포함 verbatim 전사 | CrisperWhisper 2.0 verbatim 모드 의사 라벨 (+ `nyralabs/disfluency_speech_english` train 실라벨 병행) |
| SPK | frame별 화자 활동/전환 | MSP-Conversation 수작업 diarization |

**SPK 태스크 구체화 (택1, Dev에서 결정)**:
- (a) frame-level 화자 전환(change point) 감지 — diarization 라벨만으로 가능, 단순
- (b) TDNN 화자 임베딩의 frame-level 디스틸(cosine loss) — 파일럿과의 연결이 직접적
- 권장: (b)를 기본, (a)를 ablation으로.

**CTC 어휘**: 문자 단위 + paralinguistic 태그를 원자 토큰으로 추가(`[UH]`, `[UM]`, `[laughter]`, `[cough]`, ...). intended/verbatim 헤드는 어휘 공유, 헤드 분리.

**핵심 ablation 매트릭스** (모두 causal, 동일 레시피):

| # | VAD | +ASR | +vASR | +SPK | 검증하는 주장 |
|---|-----|------|-------|------|----|
| 1 | ✓ | | | | 실험 1 (하한) |
| 2 | ✓ | ✓ | | | content 정보의 기여 |
| 3 | ✓ | | ✓ | | paralinguistic 정보의 추가 기여 (2와 비교) |
| 4 | ✓ | | | ✓ | 화자 정보의 기여 (특히 dominance) |
| 5 | ✓ | ✓ | ✓ | ✓ | 최종 모델 |

기대 스토리: #2~#4의 속성별 이득 패턴이 실험 2의 feature 의존성 분석과 일치 → 내러티브 완결.

### 4.4 평가 프로토콜

- **지표**: conversation part 단위 CCC (V/A/D 각각), Test1 중심 보고 + Test2 리더보드 제출.
- **온라인 추론 검증**: 학습 시 causal mask 사용만으로는 부족 — 실제 스트리밍 조건(chunked inference + KV 캐시, 과거만 접근)으로 Test1을 재평가해 수치 일치를 확인. 지연(latency)·RTF 보고.
- **세그먼트 처리**: 베이스라인과 동일하게 1분 세그먼트로 학습하되, 평가 시 part 전체를 스트리밍으로 처리(causal 모델은 컨텍스트 연장이 자유로움 — 부수적 이점으로 보고 가능).

## 5. 의사 라벨링 파이프라인 (선행 작업)

1. MSP-Conversation 오디오 확보 → conversation part 단위 정리
2. CrisperWhisper 2.0 (large)로 각 part를 2회 전사: verbatim 모드 / intended 모드 (longform conditional continuation 사용, 단어 타임스탬프 포함 저장)
3. 품질 필터: verbatim/intended 간 content 일치 검증(intended ⊆ verbatim 서브시퀀스 체크), 이상 샘플 제외
4. diarization 라벨과 시간 정렬 → frame-level 화자 타깃 생성
5. `datasets` 포맷으로 패키징 (audio, vad_trace@10Hz, verbatim_text, intended_text, speaker_frames)

## 6. 코드 설계 (transformers 표준 준수)

```
vad_detection/
  __init__.py
  configs.py                  # 실험 config (dataclass, yaml 로딩)
  datasets/
    base.py                   # 공통 Dataset 인터페이스
    msp_conversation.py       # 원본 → datasets 변환
    pseudo_label.py           # CrisperWhisper 2.0 의사 라벨 파이프라인
    collator.py               # 멀티태스크 collator (텍스트 패딩, 트레이스 정렬)
  models/
    WavLM/                    # 베이스라인 재현용 bidirectional 래퍼 (실험 0)
    CausalWavLM/              # CausalWavLMConfig(WavLMConfig 상속, is_causal 필드)
                              # CausalWavLMModel(WavLMPreTrainedModel 상속)
                              #  - causal attention mask, causal conv pos-emb
    <MODEL_NAME>/             # 최종 조인트 모델 (PreTrainedModel): 백본 + 4개 헤드
                              #  - forward()가 loss dict 반환 (Trainer 호환)
  utils/
    losses.py                 # CCCLoss, 조인트 가중 합산
    metrics.py                # part 단위 CCC compute_metrics
    streaming.py              # chunked causal inference (KV cache), latency 측정
  training/
    train.py                  # HfArgumentParser + Trainer 진입점
    baseline.py               # 실험 0 재현 (BiLSTM 헤드)
  scripts/
    run_exp0.sh ... run_exp3_ablation.sh
tests/
  test_causality.py           # 미래 프레임 변조 시 과거 출력 불변 검증 (필수 테스트)
  test_ccc.py, test_collator.py
```

설계 원칙:
- `PreTrainedModel`/`PretrainedConfig` 상속으로 `save_pretrained`/`from_pretrained` 왕복 보장, 체크포인트는 HF 포맷.
- 학습은 표준 `Trainer` + `TrainingArguments` (커스텀은 `compute_loss` 오버라이드 수준으로 최소화).
- 데이터는 `datasets` 라이브러리, 오디오는 lazy decoding.
- **causality 단위 테스트가 최우선**: 입력 뒷부분을 랜덤 변조해도 앞 타임스텝 출력이 bit-exact 불변임을 CI에서 검증. (인과성 주장의 기술적 담보)
- 의존성: `transformers`, `datasets`, `torch`, `torchaudio`, `accelerate`, `evaluate` — uv로 관리.

## 7. 리스크와 대응

| 리스크 | 대응 |
|---|---|
| causal 변환 후 하락 폭이 작아 스토리 약화 | 헤드까지 완전 causal화 확인, lookahead sweep으로 비용 곡선 제시 |
| 의사 라벨 품질(다자 대화·중첩 발화에서 CrisperWhisper 성능 저하) | 중첩 구간(코퍼스의 4.8%)은 CTC 로스에서 마스킹, 품질 필터 단계 강화 |
| 조인트 로스 이득이 bidirectional 갭을 못 메움 | 최소 성공 기준(실험 1 대비 향상)으로도 논문 성립하도록 ablation·분석 강화 |
| 베이스라인 재현 실패(저자 체크포인트 부재) | 재현 수치를 명시하고 자체 재현 베이스라인 기준으로 상대 비교 일관성 유지 |
| MSP-Podcast 라이선스/데이터 확보 지연 | 인코더 파인튜닝을 `tiantiaf/wavlm-large-msp-podcast-emotion-dim`로 대체하는 경로 확보 |

## 8. 마일스톤

1. **M1 — 인프라**: 데이터 빌드 + 베이스라인 재현(실험 0) + causality 테스트 통과
2. **M2 — 실험 1**: causal 변환, 인과성 비용 정량화
3. **M3 — 의사 라벨**: CrisperWhisper 2.0 파이프라인 완료, 라벨 품질 리포트
4. **M4 — 실험 3**: 조인트 로스 ablation 매트릭스 완주
5. **M5 — 정리**: 실험 2 재정리(속성별 분석), 스트리밍 검증, Test2 제출, 논문 초고
