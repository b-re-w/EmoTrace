# EmoTrace

**논문 제목(잠정)**: *EmoTrace: Online Prediction of Continuous Emotion Traces via Auxiliary Verbatim and Speaker Supervision*

> 최종 수정: 2026-08-08
> 관련 문헌: [docs/references/](docs/references/) 3편 (MSP-Conversation, Transcription Policy/CrisperWhisper 2.0, VibeVoice-ASR)

## 1. 연구 목표

MSP-Conversation 데이터셋에서 **모든 타임스텝(100ms 간격)에 대해 valence/arousal/dominance(VAD)를 온라인으로 예측**하는 실시간 모델을 개발한다. 모델은 각 시점 t에서 과거 오디오만 사용하는 causal 구조여야 한다.

## 2. 논문 전개 (내러티브)

논문의 논리 사슬은 실험 순서와 일치한다:

1. **[실험 1]** WavLM을 causal로 변환하고 bidirectional teacher로부터 증류해 CausalWavLM을 만든 뒤 VAD 단일 태스크로 학습하면, bidirectional 베이스라인 대비 성능이 크게 하락한다. 이는 미래 컨텍스트 상실에 따른 구조적 비용이다.
2. **[가설]** 이 하락의 상당 부분은 VAD 단일 로스만으로는 causal representation이 감정 예측에 필요한 정보를 충분히 인코딩하지 못하기 때문이다. → Joint loss로 representation을 보강해야 한다.
3. **[실험 2 · 근거]** frozen 백본 3종(VibeVoice-ASR, WavLM, TDNN 화자모델) feature 결합 파일럿에서 bidirectional 베이스라인을 상회하는 결과를 이미 확보 → 감정 예측이 **content 정보와 화자 정보에 의존**함을 보이는 진단 실험으로 재배치한다. (백본 융합은 최종 시스템이 아님)
4. **[실험 3 · 본실험]** 실험 1의 causal 모델에 **전사(발성 이벤트 태그 포함) + 화자 예측 + VAD 조인트 로스**를 적용하면 실험 1 대비 성능이 회복/향상됨을 보인다.

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

→ **결론: causal 변환은 직접 수행한다** (5.2절). 공개된 변환 모델이 없다는 점 자체가 기여의 여지.

### 3.2 Verbatim(감정 태그) 데이터/모델 — **모두 공개되어 있음**

- **모델**: `nyralabs/CrisperWhisper2.0_{large,medium,small,turbo}` — verbatim/intended 모드 전환, verbatimize, 단어 타이밍, longform(conditional continuation) 지원. 의사 라벨 생성기로 사용.
- **데이터**: `nyralabs/disfluency_speech_english` — 4,957 발화 / 9.4h, **train 4,458 / val 250 / test 249**, `verbatim_transcript` + `intended_transcript` 페어 제공. **실제 라벨 데이터이므로 보조 태스크 학습에 디스틸 없이 직접 사용 가능.**
- `nyralabs/disfluency_speech_german` — 독일어 test-only (본 연구에는 불필요, 참고용)

→ **결론: 하이브리드 전략.** (a) disfluency_speech_english train split을 verbatim ASR 보조 태스크의 실라벨 데이터로 사용 + (b) MSP-Conversation 오디오 자체에는 CrisperWhisper 2.0으로 verbatim 의사 라벨 생성(동일 오디오에 멀티 로스를 걸기 위해 필요). (a)만으로는 도메인(단일화자 스튜디오 vs 다자 팟캐스트)이 달라 (b)가 본선이고 (a)는 보강재.

### 3.3 베이스라인 재현 자산

- `tiantiaf/wavlm-large-msp-podcast-emotion-dim` — WavLM-large를 MSP-Podcast로 V/A/D 파인튜닝(Vox-Profile/SAILER 계열). MSP-Conversation 논문 베이스라인의 1단계 인코더와 유사 → **베이스라인 재현 시 초기화 후보**. 논문 저자의 정확한 체크포인트가 아니므로 재현 수치가 다를 수 있음을 명시할 것.
- `microsoft/wavlm-large` — 원본 백본.

### 3.4 데이터셋 접근

**데이터는 이미 로컬 NAS에 확보되어 있다**(읽기 전용 마운트).

- **MSP-Conversation v2.0** — `/home/work/DOLAB_NAS/datasets/audio/MSP_CONVERSATION_2.0/`

| 파일 | 내용 |
|---|---|
| `Audio.tar.gz` (9.6GB) | **대화 전체** 오디오 310개. conversation part로 자르는 것은 공식 유틸리티가 수행 |
| `Traces.zip` | 처리된 트레이스 2,400개 = `{Valence,Arousal,Dominance}/MSP-Conversation_{conv}_{part}.csv`. 열은 `Central_Time, Start_Time, End_Time, Mean, {annotator ids...}` — **평가자별 개별 트레이스가 그대로 들어 있다** |
| `Annotations.zip` (534MB) | **원 주석** 14,578개 = `{속성}/MSP-Conversation_{conv}_{part}_{worker}.csv`. 프레임률 불균일·반응지연 미보정 상태 |
| `Speaker_Diarization.zip` | 대화별 JSON 310개. 항목은 `{Start_Time, End_Time, Speaker, MSP_Podcast_Speaker, Gender}`이며 **구간이 겹칠 수 있다**(중첩 발화) |
| `Time_Labels.zip` | `conversations.txt`, `conversation_parts.txt`(`{part};{start};{end}` 초 단위, 팟캐스트 기준), `test2_conversation_parts.txt`, `segments.json`(MSP-Podcast 발화 턴의 대화 내 위치) |
| `partitions.txt` | `MSP-Conversation_{conv};{Train\|Development\|Test1\|Test2}` |
| `msp_conv_utils.py.zip` | **공식 유틸리티** — §4 참조 |

- **MSP-Podcast v2.0** — `/home/work/DOLAB_NAS/datasets/audio/MSP_PODCAST_2.0/`
  - `Audios.tar.gz`(48.8GB), `Labels.zip`, `Partitions.txt`, `Speaker_id.txt`
  - **`Transcripts.zip`(178MB)과 `ForceAligned.zip`(427MB)** — 실제 사람 전사와 강제정렬 결과. MSP-Conversation과 겹치는 12,555개 발화 턴에 대해 **의사 라벨 품질을 실측 검증**할 수 있는 근거 자료이며, `segments.json`으로 대화 내 위치가 매핑된다.

추출 위치는 쓰기 가능한 로컬(`/home/work/voice_research/ser/data/`, 여유 3.9TB)로 하고 NAS는 읽기 전용으로만 접근한다.

**데이터 함정 (실측 확인)**
- `Audio.tar.gz`는 이름과 달리 **gzip이 아니라 평문 tar**다(`tar -xf`로 풀어야 함). 아카이브 루트에 wav가 평평하게 들어 있다.
- 오디오 310개 중 **10개가 16kHz 모노가 아니다**: 44.1kHz 7개, 32kHz 1개, 22.05kHz 1개, 그리고 16kHz 스테레오 1개(총 스테레오 7개). 학습 전 16kHz 모노로 정규화해야 하며, 그러지 않으면 콜레이터가 샘플레이트 불일치로 실패한다. 해당 파일: `MSP-Conversation_{1183,1184,1185,1186,1187,1188,1189,1190,1277,1512}.wav`.

### 3.5 WavLM vs VibeVoice-ASR 구조 비교 — causal 변환 설계에 주는 함의

`transformers` 5.14 구현체(`models/wavlm`, `models/vibevoice_asr`, `models/vibevoice_acoustic_tokenizer`)와 `microsoft/VibeVoice-ASR-HF` config를 직접 확인한 결과.

| | WavLM-large | VibeVoice-ASR |
|---|---|---|
| 입력 | 16kHz | 24kHz |
| 프런트엔드 | CNN 7층 (stride 총 320) | dual tokenizer encoder (acoustic h=64 / semantic h=128), ConvNeXt-1D 블록 depths [3,3,3,3,3,3,8], kernel 7, depthwise conv + FFN(×4) + RMSNorm + LayerScale |
| 프레임 간격 | **20ms** | **133ms** (다운샘플 2·2·4·5·5·8 = 3200×) |
| 본체 | Transformer **encoder** 24층, h=1024, 16헤드, FFN 4096, pre-LN | Qwen2.5-7B **decoder-only** 28층, h=3584, 28헤드, KV 4헤드(GQA), SwiGLU, RMSNorm |
| 위치 정보 | depthwise conv pos-emb(kernel 128, **symmetric**) + gated relative position bias(320버킷, max_dist 800) | RoPE (θ=1e6, max_pos 131072) |
| **인과성** | **양방향** (masked span prediction 사전학습) | **완전 causal** — 모든 conv가 `CausalConv1d`(left-pad only) + `Conv1dPaddingCache` 스트리밍, LLM은 decoder-only |
| 융합 | — | acoustic/semantic 각각 Linear→RMSNorm→Linear 후 **합산**, `audio_token_id` 자리에 masked_scatter |
| 학습 목표 | frame-level 음향/음소 표현 | 토큰 시퀀스 생성 (Rich Transcription) |

**함의 1 — 파일럿의 VibeVoice 이득은 미래 정보에서 온 것이 아니다.** VibeVoice-ASR은 토크나이저부터 LLM까지 구조적으로 causal이다(`VibeVoiceAcousticTokenizerCausalConv1d`는 좌측 패딩만 사용하며 스트리밍 캐시까지 갖춤). 즉 실험 2에서 VibeVoice feature가 준 향상은 **causal 모델이 원리적으로 획득 가능한 정보**이며, 이는 "joint loss로 그 정보를 주입한다"는 실험 3의 논리를 강하게 뒷받침한다. 논문에서 명시할 것.

**함의 2 — 반대로 WavLM·TDNN 기여분에는 비인과 성분이 섞여 있다.** 파일럿의 WavLM feature는 양방향이고, TDNN 화자 모델도 attentive statistics pooling이 발화 전역에 걸리면 비인과다. 실험 2를 논문에 실을 때 이 **비대칭을 명시**하고, 가능하면 causal 조건(§5.2)에서 재측정한다.

**함의 3 — 프레임률이 causal 설계의 핵심 제약.** causal 모델은 KV 캐시를 들고 진행하므로 20ms 프레임은 긴 컨텍스트에서 비싸다(5분 conversation part = 15,000 프레임). VibeVoice가 1시간을 27k 토큰에 담는 이유가 이것이다. 베이스라인의 1분 분할은 양방향 WavLM이 발화 전체를 한 번에 봐야 하는 데서 온 제약이었으나, causal 모델은 KV 캐시로 part 전체를 끊김 없이 흘릴 수 있다 — 부수적 이점으로 보고할 것. **단 "흘릴 수 있다"와 "장거리를 활용한다"는 다르다**(아래 D1의 16초 포화 참조). **단, 구체적 프레임률 설계는 아래 미결 사항이며 조인트 로스 구조와 직결된다.**

> **[D1] 프레임률·컨텍스트 설계 — 대부분 결정됨**
>
> **제약**: CTC는 100ms 프레임에서 사실상 학습되지 않는다(문자/음소 정렬에는 통상 20–40ms 해상도 필요).
>
> **결정 1 — 전 구간 20ms 단일 프레임률.** 100ms로 내려갈 유일한 근거는 레이블 주기였는데, 이는 코퍼스 저자들의 후처리 산물이지 현상의 성질이 아니다: 원 조이스틱 주석은 **약 33 samples/sec(≈30ms)**로 수집되었고 120ms 윈도/100ms 홉 median filter를 거쳐 100ms가 되었다(MSP-Conversation §IV-A). 모델을 100ms에 묶을 이유가 없다.
>
> **결정 2 — 원 주석에서 20ms 트레이스를 직접 재생성한다(보간 아님).** 릴리스에 포함된 공식 유틸리티 `msp_conv_utils.make_traces(VersionFldr, SaveFldr, window=120, overlap=20, time_delay=3000)`는 **프레임률과 반응지연을 인자로 받아** 원 주석으로부터 트레이스를 만든다. 기본값은 window 120ms / overlap 20ms이므로 홉이 100ms인데, `overlap=100`으로 두면 **같은 median filter 평활을 유지한 채 홉만 20ms**가 된다.
>
> 즉 공개된 100ms 트레이스를 사후 보간할 필요가 없다. 원 조이스틱 주석(≈33 samples/sec)에서 우리가 원하는 격자로 바로 만들며, 이는 코퍼스 저자가 그 용도로 제공한 함수를 그대로 쓰는 것이라 방법론적으로도 안전하다. 반응지연 3초 보정도 동일 함수가 처리한다. **평가는 베이스라인과 비교 가능하도록 기본 설정(홉 100ms)으로 만든 트레이스에서 수행한다.**
>
> **결정 3 — 단일 모델, 얇은 헤드.** 시간 모델링은 CausalWavLM 24층 안에서 끝나고 헤드는 선형 투영이다. 근거는 §5.3 참조(헤드 용량이 ablation을 오염시키는 문제).
>
> 이득: 풀링 모듈과 그 인과성 검증이 통째로 사라지고, 실시간 출력 주기가 100ms → **20ms로 5배 조밀**해지며, 모든 헤드가 같은 시간축이라 disfluency 이벤트와 VAD 동역학의 정렬 분석이 직접 가능하다.
>
> **결정 4 — sliding-window causal attention을 명시적으로 도입한다(윈도 16초 = 800프레임).** WavLM의 gated relative position bias는 방향당 160버킷으로, 0–79프레임(**~1.58초**)은 정확히 구분, 80–799프레임(**1.6–16.0초**)은 로그 압축, **800프레임(16.0초) 이상은 전부 동일 버킷**이다(`max_bucket_distance=800`). 즉 16초는 임의의 숫자가 아니라 **모델이 거리를 구분할 수 있는 한계 그 자체**이며, 윈도를 여기에 맞추면 버리는 정보가 위치 정보가 이미 소실된 구간뿐이다.
>
> 근거와 이득:
> - teacher(양방향 WavLM)도 동일한 16초 한계를 가지므로, 그 너머의 장거리 능력은 **증류로 얻어질 수 있는 것이 아니다**. 전 구간 어텐션을 유지해도 얻는 것이 불분명하다.
> - 스트리밍 비용이 스텝당 **상수**가 된다(O(윈도), O(t) 아님) → 무한 길이 스트림에 대해 지연·메모리가 발목잡지 않는다. 실시간 주장에 직결.
> - 학습도 O(L·W)로 떨어져 긴 시퀀스를 그대로 넣을 수 있다.
> - 사전학습이 짧은 크롭으로 수행된 것과 어텐션 스팬이 일치하므로 길이 분포 불일치가 사라진다.
> - 구현: `transformers.masking_utils.create_sliding_window_causal_mask`가 `config.sliding_window`를 읽는다 → `CausalWavLMConfig`에 필드 하나 추가로 끝난다.
>
> **"장거리 맥락이 도움이 되는가"는 이제 윈도 크기 sweep으로 답한다**(예: 4초 / 8초 / 16초 / 32초 / 무제한). 윈도를 키워 이득이 있으면 그때 `max_bucket_distance` 확장이나 단방향 버킷 재매핑을 검토한다. **두 번째 시퀀스 모델을 얹는 방식은 채택하지 않는다.**
>
> **(별개 사안) 학습 시 시퀀스 길이**: O(L²) 메모리 때문에 배치 길이는 따로 정한다. 이는 데이터 배치 결정이지 모델의 컨텍스트 범위가 아니다.

**함의 4 — 차용할 구현 패턴.** VibeVoice의 `Conv1dPaddingCache`(좌측 패딩 상태를 캐싱해 청크 간 연속성 보장)는 우리가 causal conv positional embedding을 스트리밍으로 돌릴 때 그대로 참조할 수 있다. `transformers`에 이미 있으므로 재구현하지 않는다. 새로 얹는 causal 헤드에는 RMSNorm + LayerScale + RoPE(큰 θ)를 쓰되, **WavLM 본체는 사전학습 보존을 위해 구조를 바꾸지 않는다**(마스크와 패딩만 변경).

## 4. 최종 모델 구조 (EmoTrace)

§3.5의 구조 분석과 D1 결정을 반영한 확정 사양. **단일 causal 모델 + 얇은 헤드**이며, 별도의 시퀀스 모델을 얹지 않는다.

```
입력 오디오 16kHz  (정규화: 데이터셋 고정 상수 — 발화 전체 통계 금지)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ CausalWavLM  (WavLM-large 24층에서 파생, 프레임 20ms)        │
│                                                             │
│  CNN feature extractor  ── 원본 그대로 (+5ms 선행, 문서화)  │
│  causal positional conv ── 좌측 패딩 127, SamePadLayer 제거 │
│  causal self-attention  ── sliding window 16초(800프레임)   │
│                            + gated rel-pos bias (원본 유지) │
│                            + KV 캐시 / sdpa·flash 경로      │
└─────────────────────────────────────────────────────────────┘
        │ 20ms 프레임 표현 (1024차원)
        ├──────────────┬──────────────────┬───────────────────┐
        ▼              ▼                  ▼                   │
   [VAD head]     [ASR head]         [SPK head]               │
   Linear→3        Linear→CTC         Linear→화자             │
   V/A/D 회귀      발성 이벤트 태그    프레임 단위             │
   CCC loss        포함 전사           supervision            │
   (20ms 조밀)     ([uh],[laughter]…)                         │
```

**구성 절차** — 아래 순서 자체가 모델 정의의 일부다.
1. **구조 변환**: 위 3개 지점 수술 (§5.1). 이 시점의 모델은 학습이 덜 된 중간 상태이며 보고 대상이 아니다.
2. **Bi→Causal 증류**: 양방향 WavLM을 teacher로, 프레임별 은닉 상태를 맞춰 수렴까지 학습 → **CausalWavLM 완성**. 라벨 불필요.
3. **태스크 학습**: 얇은 헤드를 붙여 조인트 로스로 학습 → **EmoTrace**.

**로스**: `L = w_vad·L_CCC + w_asr·L_CTC + w_spk·L_spk`

**핵심 사양 요약**

| 항목 | 값 | 근거 |
|---|---|---|
| 프레임 간격 | **20ms** (전 구간 단일) | CTC가 100ms에서 학습 불가. 레이블의 100ms는 코퍼스 후처리 산물(원본 ≈30ms) |
| 어텐션 스팬 | **16초 sliding window** | rel-pos bias가 800프레임(16초)에서 포화 — 윈도를 분해능 한계에 일치시킴. 스트리밍 비용이 스텝당 상수가 됨 |
| VAD 레이블 | 100ms → **20ms 선형 보간** | 스텝의 80%가 무감독이 되는 것을 피함. 원 신호가 ≈30ms였으므로 해상도 복원에 가까움 |
| 헤드 | **선형 투영만** | 헤드에 용량을 두면 ablation이 "표현 개선"이 아니라 "헤드 용량"을 측정하게 됨 |
| CTC 타깃 | **발성 이벤트 태그 포함 전사** (단일 헤드) | intended는 verbatim의 부분수열이라 중복. 두 타깃은 동시 헤드가 아니라 ablation 조건 |
| 출력 | 20ms마다 V/A/D | 평가는 베이스라인 비교를 위해 원 100ms 지점에서 |

**의도적으로 넣지 않은 것**: 별도 causal Transformer 헤드, 50→100ms 풀링 모듈, intended 전용 두 번째 CTC 헤드. 모두 초기 설계에 있었으나 근거가 없어 제거했다.

## 5. 실험 설계

### 5.0 실험 0 (준비): 베이스라인 재현

- 논문 레시피 그대로: WavLM-large(MSP-Podcast 파인튜닝) → 120ms 윈도/100ms 홉 mean-pool → FFN(512×3, ReLU, dropout 0.5) → BiLSTM(2층, hidden 256) 헤드, CCC loss, 1분 세그먼트(5초 오버랩), Adam lr 1e-4, batch 64.
- 목적: 우리 구현으로 논문 수치(Test1 LSTM ST: V 0.676 / A 0.669 / D 0.517)에 근접함을 확인 → 이후 모든 비교의 신뢰 기반.

### 5.1 실험 1: Causal 변환과 인과성 비용의 격리

**변환 대상 (WavLM-large 내부의 미래 정보 누수 지점 3곳):**

| # | 지점 | 누수량 | 처리 |
|---|---|---|---|
| 1 | Self-attention (24층, 전 프레임 참조) | 전체 미래 | causal mask 적용 |
| 2 | `WavLMPositionalConvEmbedding` (kernel 128, padding 64, groups 16) | **미래 64프레임 = 1.28초** | causal(left-only) padding으로 교체 |
| 3 | CNN feature extractor (kernel [10,3,3,3,3,2,2], stride [5,2,2,2,2,2,2], padding 없음) | 프레임 시작점 기준 +5ms (수용장 400샘플=25ms, stride 320샘플=20ms) | 무시하되 문서에 명시 |

**2번이 실질적 핵심이다.** attention mask만 씌우고 positional conv를 방치하면 매 레이어마다 1.28초 미래가 새어 들어와 "causal 모델"이 성립하지 않는다. 이 누수는 `tests/test_causality.py`가 반드시 잡아내야 하는 대상이다.

**gated relative position bias 처리**: WavLM은 signed relative position을 320버킷으로 나누는데(`(relative_positions > 0) * num_buckets`로 미래/과거 절반씩 분할, max_distance 800), causal mask 하에서는 **미래 절반 160버킷이 영구히 죽는다**. 두 가지 선택지를 ablation한다.
- (a) 그대로 두기 — 사전학습 가중치 완전 보존, 단 표현력 절반 낭비
- (b) 단방향 버킷팅으로 재매핑 — 과거 방향 거리 해상도가 2배가 되지만 임베딩 재해석이 필요(부분 재학습)
- 기본은 (a). (b)는 여유가 있으면 시도.

**구조 변환만으로는 모델이 성립하지 않는다.** WavLM의 사전학습 목표는 masked prediction으로, **본질적으로 양방향 컨텍스트를 요구한다**. causal mask를 씌우는 순간 가중치는 학습 시 경험한 적 없는 조건에 놓인다. 그리고 WavLM은 표현 학습 모델이므로 **causal 조건에서 이어서 돌릴 수 있는 자기 목표가 없다** — 자신의 사전학습 목표(masked prediction)는 양방향이라 causal에서 정의되지 않고, ASR 같은 다른 태스크를 붙이는 것은 재적응이 아니라 그냥 다른 학습이다.

**따라서 causal 모델을 만드는 유일한 정합적 방법은 Bi→Causal 지식 증류다.** bidirectional WavLM을 teacher, 구조 변환된 모델을 student로 두고 frame별 hidden state를 맞춘다. 이것은 실험 변형이 아니라 **CausalWavLM의 구성 절차 그 자체**다.
- teacher의 표현이 곧 타깃이므로 라벨이 불필요하고 오디오만으로 스케일된다(MSP-Podcast/LibriSpeech/MSP-Conversation).
- "충분히 적응시켰는가"라는 반론을 **증류 손실 수렴**이라는 객관적 정지 기준으로 차단한다.
- **수렴 후 남는 증류 잔차 = 과거로부터 예측 불가능한 정보량**, 즉 인과성 비용의 직접 추정치다. 레이어별·시간별 잔차 분포는 논문 그림으로 쓸 수 있다.
- 주의: 증류는 통상 student ≤ teacher 상한을 만든다. **초기화 용도이며, teacher를 넘는 것은 실험 3의 조인트 로스가 담당한다.**

즉 파이프라인은 **구조 변환 → 증류(수렴까지) → CausalWavLM 완성**이며, 구조만 바꾼 중간 상태는 학습이 덜 된 고장난 모델이지 보고 대상이 아니다. 실험 1은 완성된 CausalWavLM 위에 VAD 단일 로스를 얹은 **단일 모델**로 수행한다.

**예비 실측 — 증류 잔차의 층별 분포 (2026-08-08, 2에포크 시점)**

증류 학습 초기(376스텝 중 100스텝 부근, 손실 0.29)의 층별 잔차는 다음과 같다. 잔차는 teacher(양방향)의 프레임별 은닉 상태를 student(causal)가 과거만으로 얼마나 재현하지 못하는지를 나타내므로, **미래 문맥 없이는 얻을 수 없는 정보량의 층별 추정치**다.

| 층 | 1 | 4 | 8 | 12 | 16 | 20 | **24** |
|---|---|---|---|---|---|---|---|
| 잔차 | 0.155 | 0.221 | 0.261 | 0.279 | 0.284 | 0.274 | **0.663** |

두 가지가 관찰된다.

1. **깊이에 따라 증가하다 중간층에서 평탄해진다.** 얕은 층은 국소 음향 특징이라 과거만으로 잘 복원되고, 깊어질수록 문맥 의존적이 된다. 12~20층이 0.27~0.28로 평탄한 것은 그 구간의 표현이 비슷한 수준의 문맥 의존성을 갖는다는 뜻이다.
2. **최종 24층만 0.663으로 급등한다** — 중간층 대비 2.4배. WavLM의 최상위 표현이 유독 양방향 문맥에 의존함을 시사한다.

**함의**: MSP-Conversation 논문 베이스라인은 바로 이 최종층 출력을 SER 헤드의 입력으로 쓴다. 따라서 2번 관찰은 causal 변환 시 성능이 크게 떨어지는 이유에 대한 **구조적 설명 후보**이며, 동시에 **VAD 헤드를 최종층이 아니라 중간층(12~20)에 붙이거나 층별 가중 합을 쓰는 편이 유리할 수 있다는 가설**을 낳는다. 실험 1에서 층 선택을 변인으로 두고 확인한다.

주의: 학습 초기 값이므로 수렴 후 재측정해야 하며, 잔차의 절대값은 목적함수(cosine/MSE)와 정규화 방식에 의존한다.

**격리 원칙 (가장 중요):** 실험 1이 비교하는 두 모델은 **bidirectional teacher**와 **그 teacher의 causal 근사(CausalWavLM)**다. 즉 비교의 의미는 "동일한 표현을 미래 없이 얼마나 재현할 수 있는가"이며, student는 teacher로부터 파생되므로 초기화·데이터 계보가 자동으로 공유된다. 나머지 조건 — VAD 헤드 구조, 하이퍼파라미터, 평가 프로토콜 — 은 동일하게 맞추고, 헤드는 양쪽 모두 실험 3과 같은 구조를 쓴다(BiLSTM 베이스라인은 §5.0의 재현 결과로 별도 보고). 비교표에 "변경점: causality만"을 명시한다.

**예측 헤드**: 얇은 선형 투영 — 실험 3과 동일한 헤드를 실험 1부터 사용해 헤드 용량 confound를 제거한다. 비교 대상인 bidirectional 모델도 같은 얇은 헤드를 쓰며, 논문 베이스라인의 BiLSTM 헤드 수치는 §5.0의 재현 결과로 별도 보고한다.

**추가 분석(옵션)**: lookahead 허용량 sweep(0s/0.5s/1s/2s/∞)으로 인과성 비용 곡선 제시.

### 5.2 실험 2: 파일럿 재정리 (진단 실험)

이미 확보된 결과를 논문용으로 재구성:

- frozen VibeVoice-ASR / WavLM / TDNN feature의 **단독 및 조합 ablation** (7가지 조합) — "feature 수 증가" confound에 대한 방어.
- **속성별 기여 분해**: content feature(VibeVoice-ASR)가 valence에, 화자 feature(TDNN)가 dominance에 특히 기여하는지 확인. 이 대응이 나오면 3→4 단계(어떤 보조 로스를 쓸 것인가)의 논리가 완결된다.
- **인과성 비대칭 통제 (§3.5 함의 2)**: 백본별로 causality가 다르다 — VibeVoice-ASR은 완전 causal, WavLM은 양방향, TDNN은 pooling 범위에 따라 비인과. 각 feature의 causality를 표에 명시하고, TDNN은 causal 윈도(과거만) 조건으로 재추출해 비교한다. 그래야 "causal 모델이 획득 가능한 정보"라는 주장이 성립한다.
- 결론 문장: "감정 트레이스 예측에 필요한 정보 중 VAD 단일 로스가 포착하지 못하는 성분이 content·화자 표현에 존재하며, 그 성분은 (VibeVoice가 보여주듯) 인과적으로 계산 가능하다."

### 5.3 실험 3 (본실험): Joint Loss 학습

**모델**: 실험 1의 Causal WavLM + 멀티 헤드.

```
CausalWavLM (20ms 프레임, 단일 모델 — 모든 시간 모델링은 이 24층 안에서 끝난다)
 ├── [VAD head]   Linear → 3dim 회귀, 5프레임마다 CCC loss
 ├── [ASR head]   Linear → CTC (verbatim 타깃; [UH],[UM],[laughter] 등
 │                paralinguistic 태그를 원자 토큰으로 포함, CrisperWhisper 2.0 규약)
 └── [SPK head]   Linear → frame-level 화자 supervision
```

**헤드는 전부 얇은 투영이다(선형 또는 1–2층 MLP). 헤드에 시퀀스 모델을 두지 않는다.**
- 논문의 주장은 "보조 supervision이 **표현**을 보강한다"이다. VAD 경로에만 두꺼운 Transformer를 얹으면 VAD 단독 조건이 헤드 안에서 상당 부분을 보상해버려, ablation이 측정하는 값이 과학이 아니라 **헤드 용량**의 함수가 된다.
- 태스크 간 용량 비대칭(두꺼운 VAD 헤드 vs 얇은 CTC 헤드)도 ablation을 오염시킨다.
- 스택이 하나이므로 KV 캐시도 하나 — 스트리밍 구현이 단순해진다.

**로스**: `L = w_vad·L_CCC + w_asr·L_CTC + w_spk·L_spk`, 가중치는 Dev에서 탐색(uncertainty weighting 등 자동화 기법은 옵션).

**타깃 소스**:

| 태스크 | 타깃 | 소스 |
|---|---|---|
| VAD | 100ms 간격 V/A/D 트레이스 | MSP-Conversation 정답 라벨 |
| ASR (최종: verbatim) | 태그 포함 verbatim 전사 | CrisperWhisper 2.0 verbatim 모드 의사 라벨 (+ `nyralabs/disfluency_speech_english` train 실라벨 병행) |
| ASR (ablation 조건: intended) | 정제 전사 | CrisperWhisper 2.0 intended 모드 의사 라벨 |
| SPK | frame별 화자 활동/전환 | MSP-Conversation 수작업 diarization |

**SPK 태스크 (확정)**: frame별 3개 이진 타깃. 화자 **정체성**을 쓰지 않으므로 대화 간 일반화가 되고 순열 불변이며, ELAN diarization에서 직접 유도된다.

| 타깃 | 정의 |
|---|---|
| `speech` | 해당 프레임에 1명 이상 발화 중 |
| `overlap` | **2명 이상 동시 발화** |
| `change` | 활성 화자 집합이 직전 프레임과 달라짐(발화 시작/종료 모두 포함) |

중첩 발화는 코퍼스의 4.8%(Test2는 9.2%)를 차지하므로 무시할 수 없다. `overlap`을 명시적 타깃으로 두면 (1) 화자 전환 감지가 중첩 구간에서 정의 불능이 되는 문제를 피하고, (2) **중첩 구간에서 CTC 로스를 마스킹**할 근거를 모델 자신이 갖게 된다(중첩 구간의 전사 의사 라벨은 신뢰도가 낮다).

로스는 프레임별 BCE이며, `change`는 희소하므로 positive weight를 적용한다.

**CTC 헤드는 하나다.** intended와 verbatim은 **동시에 붙이는 두 헤드가 아니라 같은 헤드의 두 타깃 조건**이다.
- intended 전사는 verbatim 전사에서 disfluency를 삭제한 것이므로(CrisperWhisper는 intended ⊆ verbatim 부분수열임을 검증한다) **verbatim 타깃이 intended 타깃의 정보를 포함한다** — 두 헤드를 동시에 두는 것은 대부분 중복이다.
- 게다가 프레임 동기 CTC에서 intended 타깃은 "음향적으로 존재하는 구간을 출력하지 말라"는 요구가 되어, 정렬 학습에는 부자연스러운 신호다.
- 따라서 **최종 모델은 verbatim 타깃 하나**를 쓰고, "content만" vs "content+paralinguistic"의 대비는 ablation에서 **타깃을 바꿔** 측정한다.

**CTC 어휘**: 문자 단위 + paralinguistic 태그를 원자 토큰으로 추가(`[UH]`, `[UM]`, `[laughter]`, `[cough]`, ...). 두 타깃 조건은 어휘를 공유한다(intended 조건에서는 태그 토큰이 등장하지 않을 뿐).

**핵심 ablation 매트릭스** (모두 causal, 동일 레시피):

| # | VAD | CTC 타깃 | +SPK | 검증하는 주장 |
|---|-----|------|------|----|
| 1 | ✓ | — | | 실험 1 (하한) |
| 2 | ✓ | intended | | content 정보의 기여 |
| 3 | ✓ | verbatim | | paralinguistic 정보의 추가 기여 (2와 비교) |
| 4 | ✓ | — | ✓ | 화자 정보의 기여 (특히 dominance) |
| 5 | ✓ | verbatim | ✓ | 최종 모델 |

기대 스토리: #2~#4의 속성별 이득 패턴이 실험 2의 feature 의존성 분석과 일치 → 내러티브 완결.

### 5.4 평가 프로토콜

- **지표**: conversation part 단위 CCC (V/A/D 각각), Test1 중심 보고 + Test2 리더보드 제출.
- **온라인 추론 검증**: 학습 시 causal mask 사용만으로는 부족 — 실제 스트리밍 조건(chunked inference + KV 캐시, 과거만 접근)으로 Test1을 재평가해 수치 일치를 확인. 지연(latency)·RTF 보고.
- **세그먼트 처리**: 베이스라인과 동일하게 1분 세그먼트로 학습하되, 평가 시 part 전체를 스트리밍으로 처리(causal 모델은 컨텍스트 연장이 자유로움 — 부수적 이점으로 보고 가능).

## 6. 의사 라벨링 파이프라인 (선행 작업)

1. MSP-Conversation 오디오 확보 → conversation part 단위 정리
2. CrisperWhisper 2.0 (large)로 각 part를 2회 전사: verbatim 모드 / intended 모드 (longform conditional continuation 사용, 단어 타임스탬프 포함 저장)
3. 품질 필터: verbatim/intended 간 content 일치 검증(intended ⊆ verbatim 서브시퀀스 체크), 이상 샘플 제외
4. diarization 라벨과 시간 정렬 → frame-level 화자 타깃 생성
5. `datasets` 포맷으로 패키징 (audio, vad_trace@100ms, verbatim_text, intended_text, speaker_frames)

## 7. 코드 설계 (transformers 표준 준수)

```
ser_project/
  datasets/
    base.py                   # 공통 Dataset 인터페이스
    msp_conversation.py       # 원본 → datasets 변환
    pseudo_label.py           # CrisperWhisper 2.0 의사 라벨 파이프라인
    collator.py               # 멀티태스크 collator (텍스트 패딩, 트레이스 정렬)
  models/
    WavLM/                    # 베이스라인 재현용 bidirectional 래퍼 (실험 0)
    CausalWavLM/              # CausalWavLMConfig(WavLMConfig 상속, is_causal 필드)
                              # CausalWavLMModel(WavLMPreTrainedModel 상속)
                              #  - causal attention mask
                              #  - CausalWavLMPositionalConvEmbedding (left-pad, §5.1 표 2번)
                              #  - 스트리밍용 conv padding cache (VibeVoice 패턴 차용)
    EmoTrace/                 # 최종 조인트 모델 (PreTrainedModel): 백본 + 4개 헤드
                              #  - EmoTraceConfig / EmoTraceForVADPrediction
                              #  - forward()가 loss dict 반환 (Trainer 호환)
  utils/
    train.py                  # HfArgumentParser + Trainer 진입점
    evaluate.py               # part 단위 CCC compute_metrics
    losses.py                 # CCCLoss, 조인트 가중 합산
    streaming.py              # chunked causal inference (KV cache), latency 측정
experiment_1_causal.ipynb     # 실험 1: causal 변환 + 인과성 비용
experiment_2_pilot.ipynb      # 실험 2: frozen 백본 진단 (파일럿 재정리)
experiment_3_emotrace.ipynb   # 실험 3: 조인트 로스 본실험
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

## 8. 리스크와 대응

| 리스크 | 대응 |
|---|---|
| causal 변환 후 하락 폭이 작아 스토리 약화 | 헤드까지 완전 causal화 확인, lookahead sweep으로 비용 곡선 제시 |
| 의사 라벨 품질(다자 대화·중첩 발화에서 CrisperWhisper 성능 저하) | 중첩 구간(코퍼스의 4.8%)은 CTC 로스에서 마스킹, 품질 필터 단계 강화 |
| 조인트 로스 이득이 bidirectional 갭을 못 메움 | 최소 성공 기준(실험 1 대비 향상)으로도 논문 성립하도록 ablation·분석 강화 |
| 베이스라인 재현 실패(저자 체크포인트 부재) | 재현 수치를 명시하고 자체 재현 베이스라인 기준으로 상대 비교 일관성 유지 |
| MSP-Podcast 라이선스/데이터 확보 지연 | 인코더 파인튜닝을 `tiantiaf/wavlm-large-msp-podcast-emotion-dim`로 대체하는 경로 확보 |

## 9. 마일스톤

1. **M1 — 인프라**: 데이터 빌드 + 베이스라인 재현(실험 0) + causality 테스트 통과
2. **M2 — 실험 1**: causal 변환, 인과성 비용 정량화
3. **M3 — 의사 라벨**: CrisperWhisper 2.0 파이프라인 완료, 라벨 품질 리포트
4. **M4 — 실험 3**: 조인트 로스 ablation 매트릭스 완주
5. **M5 — 정리**: 실험 2 재정리(속성별 분석), 스트리밍 검증, Test2 제출, 논문 초고
