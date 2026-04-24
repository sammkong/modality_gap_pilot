# Development Plan

## 폴더 구조, 환경, 단계별 개발 로드맵

이 문서는 "무엇을 어떤 순서로 만들 것인가"를 기술한다. 연구 설계(`01_PROJECT_OVERVIEW.md`)와 AI 코딩 지침(`02_AI_ASSISTANT_GUIDELINES.md`)을 먼저 읽은 후 이 문서로 온다.

---

## 1. 실행 환경 아키텍처

### 1.1 하이브리드 구조 개요

```
┌─────────────────────────┐          ┌─────────────────────────┐
│   로컬 VS Code (CPU)    │          │   Colab Pro (GPU)       │
│                         │          │                         │
│  - 코드 작성            │          │  - CLIP embedding       │
│  - 통계 분석            │  ──────→ │  - BLIP-2 inference     │
│  - 시각화               │  GitHub  │  - GPU 작업 전반        │
│  - 수동 judgment        │          │                         │
└──────────┬──────────────┘          └──────────┬──────────────┘
           │                                    │
           │          ┌──────────────┐          │
           └────────→ │ Google Drive │ ←────────┘
                      │  (공유 저장) │
                      └──────────────┘
```

### 1.2 동기화 전략

**코드**: GitHub (로컬 ↔ Colab)
- 로컬에서 작성 → push → Colab에서 `git clone` 또는 `git pull`
- Colab 노트북은 리포 안의 `notebooks/` 폴더에 두고 Drive에 마운트

**데이터/결과물**: Google Drive (로컬 ↔ Colab)
- Colab: `/content/drive/MyDrive/modality_gap_pilot/`에 마운트
- 로컬: Google Drive for Desktop으로 동일 폴더 접근
- Embedding `.pt` 파일, CSV 결과 등 모두 Drive에 저장

### 1.3 왜 이렇게 분리하는가

- **코드는 Git**: 버전 관리, diff 추적, Codex/Claude Code 활용
- **데이터는 Drive**: 크기가 크고 (embedding .pt 파일은 수백 MB), Git에 넣기 부적합
- **Colab은 실행 전용**: 런타임 끊김 대비, 코드 수정은 로컬에서

---

## 2. 폴더 구조

```
modality_gap_pilot/                    # Git 리포 루트
│
├── README.md                          # 프로젝트 간단 소개
├── 01_PROJECT_OVERVIEW.md             # 연구 설계 문서
├── 02_AI_ASSISTANT_GUIDELINES.md      # AI 코딩 지침
├── 03_DEVELOPMENT_PLAN.md             # 본 문서
│
├── configs/
│   └── config.yaml                    # 모든 경로, 모델명, 하이퍼파라미터
│
├── src/                               # 모든 소스 코드
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── environment.py             # IS_COLAB 체크, 경로 분기
│   │   ├── cache.py                   # embedding 캐시 로드/저장
│   │   └── logging_utils.py
│   │
│   ├── stage1_structural_gap/         # [Colab] Structural gap 계산
│   │   ├── extract_coco_embeddings.py
│   │   ├── compute_centroids.py
│   │   └── verify_robustness.py
│   │
│   ├── stage2_residual/               # [Local] Residual 계산 (CPU로 충분)
│   │   ├── extract_eval_embeddings.py # 이것만 Colab
│   │   └── compute_residuals.py       # 로컬
│   │
│   ├── stage3_blip2/                  # [Colab] BLIP-2 inference
│   │   ├── run_inference.py
│   │   └── compute_entropy.py
│   │
│   ├── stage4_correctness/            # [Local] 수동 judgment
│   │   └── judgment_helper.py         # 간단한 CLI UI
│   │
│   └── stage5_analysis/               # [Local] 최종 분석
│       ├── correlations.py
│       ├── regression_analysis.py
│       └── visualizations.py
│
├── notebooks/                         # Colab에서 여는 노트북
│   ├── colab_stage1_structural_gap.ipynb
│   ├── colab_stage2_eval_embeddings.ipynb
│   ├── colab_stage3_blip2_inference.ipynb
│   └── local_stage5_analysis.ipynb    # 로컬 jupyter로 열기
│
├── data/                              # Git에 포함 안 함 (.gitignore)
│   ├── reference/                     # COCO 일부
│   └── eval/                          # VQAv2 100개 샘플
│
├── cache/                             # Git에 포함 안 함
│   ├── embeddings/
│   │   ├── coco_image_embeds.pt
│   │   ├── coco_text_embeds.pt
│   │   ├── eval_image_embeds.pt
│   │   └── eval_text_embeds.pt
│   └── centroids/
│       ├── mu_v.pt
│       ├── mu_t.pt
│       └── g_hat.pt
│
├── outputs/                           # 일부만 Git 포함
│   ├── stage1/
│   │   ├── structural_gap_summary.json   # commit
│   │   └── robustness_report.json        # commit
│   ├── stage2/
│   │   └── residuals.csv                  # commit
│   ├── stage3/
│   │   └── blip2_outputs.csv              # commit
│   ├── stage4/
│   │   └── correctness.csv                # commit
│   └── stage5/
│       ├── correlations.json              # commit
│       ├── regression_results.json        # commit
│       └── figures/                       # commit
│
├── results.csv                        # 모든 stage 결과 통합 (commit)
│
├── tests/                             # 간단한 sanity check
│   ├── test_residual_math.py
│   └── test_entropy_calculation.py
│
├── requirements.txt                   # 로컬용 (CPU)
├── requirements-colab.txt             # Colab용 (GPU lib 포함)
└── .gitignore
```

### 2.1 .gitignore 내용

```gitignore
# 대용량 데이터
data/
cache/
*.pt
*.pth
*.bin

# 환경
__pycache__/
*.pyc
.venv/
venv/
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Colab 임시 파일
sample_data/

# Secrets (혹시라도)
*.env
credentials.json
```

### 2.2 configs/config.yaml 예시

```yaml
# ========== Paths ==========
paths:
  # 로컬/Colab 자동 분기는 src/utils/environment.py에서 처리
  base_local: "./"
  base_colab: "/content/drive/MyDrive/modality_gap_pilot/"
  
  coco_annotations: "data/reference/coco_captions_val.json"
  coco_images: "data/reference/coco_images/"
  vqav2_annotations: "data/eval/vqav2_val_subset.json"
  vqav2_images: "data/eval/vqav2_images/"

# ========== Models ==========
models:
  clip:
    name: "openai/clip-vit-base-patch32"
    dtype: "float16"
  blip2:
    name: "Salesforce/blip2-opt-2.7b"
    load_in_4bit: true
    max_new_tokens: 20

# ========== Data sizes ==========
data:
  reference_n_pairs: 2000       # COCO에서 뽑을 pair 수
  eval_n_samples: 100           # VQAv2에서 평가할 샘플 수

# ========== Experiment ==========
experiment:
  seed: 42
  normalize_embeddings: true   # CLIP embedding L2-normalize 여부

# ========== Robustness check ==========
robustness:
  n_splits: 2                  # subset split 개수
  min_cosine_similarity: 0.95  # g_hat 간 similarity 임계값
```

---

## 3. 사용 데이터셋 상세

### 3.1 Reference Corpus: COCO Captions

**용도**: Structural gap 계산 (Stage 1)

**선택 이유**:
- Image-text pair가 깨끗하고 잘 정렬됨
- Hugging Face `datasets`로 바로 로드 가능
- CLIP이 학습 중 본 distribution과 유사 (OOD 우려 적음)

**로딩 방식 (Colab)**:
```python
from datasets import load_dataset
coco = load_dataset("HuggingFaceM4/COCO", split="validation", streaming=False)
# 또는 "yerevann/coco-karpathy" 등 대체 가능
```

**추출 규모**: 2,000~5,000 image-caption pair
- 2,000으로 시작, robustness가 부족하면 5,000으로 증가
- 한 이미지당 캡션 1개만 사용 (여러 개 있으면 첫 번째)

### 3.2 Evaluation Samples: VQAv2

**용도**: Residual 및 uncertainty 측정 (Stage 2~4)

**선택 이유**:
- VQA 태스크이므로 BLIP-2 평가에 자연스러움
- COCO 이미지 위에 구축되어 reference와 도메인 일치
- 정답이 명확해 수동 judgment가 가능

**로딩 방식 (Colab)**:
```python
from datasets import load_dataset
vqa = load_dataset("HuggingFaceM4/VQAv2", split="validation", streaming=True)
# 전체가 너무 크므로 streaming으로 일부만 받음
```

**샘플링 규모**: 100개
- Pilot에서는 100개가 현실적 (수동 judgment 고려)
- 의도적 다양성 확보:
  - 쉬운 질문 (색상, 개수, yes/no) ~30%
  - 중간 난이도 (객체 인식, 속성) ~40%
  - 어려운 질문 (추론, 행동, 공간 관계) ~30%

### 3.3 중요: Reference vs Eval 분리

COCO reference와 VQAv2 eval은 **이미지가 일부 겹칠 수 있다** (VQAv2는 COCO 이미지 기반). Stage 2 시작 전에 다음을 확인:

```python
# COCO에서 뽑은 이미지 ID와 VQAv2 선택 샘플의 이미지 ID 교집합 제거
coco_image_ids = set(reference_ids)
vqa_samples = [s for s in vqa_samples if s['image_id'] not in coco_image_ids]
```

이 체크를 건너뛰면 "reference에 이미 있던 이미지가 eval에도 들어가는" 누수 발생.

---

## 4. 사용 모델 상세

### 4.1 CLIP: `openai/clip-vit-base-patch32`

- **크기**: ~600MB (FP16으로 ~300MB)
- **Embedding dim**: 512
- **이유**: 논문들이 가장 많이 쓰는 baseline. Liang et al.도 이 계열 사용.
- **로드 환경**: Colab GPU (T4 이상)

**로드 코드 템플릿**:
```python
import torch
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(
    model_name, torch_dtype=torch.float16
).to("cuda").eval()
clip_processor = CLIPProcessor.from_pretrained(model_name)
```

### 4.2 BLIP-2: `Salesforce/blip2-opt-2.7b`

- **크기**: ~15GB (원본), ~4GB (4-bit quantization)
- **이유**: Colab Pro T4/L4에서 4-bit로 돌릴 수 있는 최대 크기
- **주의**: `flan-t5-xl` 버전은 더 크므로 OOM 위험

**로드 코드 템플릿**:
```python
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    quantization_config=bnb_config,
    device_map="auto",
).eval()
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
```

**중요**: BLIP-2 로드 전에 CLIP을 반드시 `del`하고 `torch.cuda.empty_cache()` 호출.

---

## 5. 단계별 개발 로드맵 (4주)

### 🟢 Week 1: 환경 세팅 + Stage 1 (Structural Gap)

#### Day 1 (로컬) — 프로젝트 초기화

**할 일**:
1. GitHub 리포 생성 (private 권장)
2. 로컬에 clone
3. 위의 폴더 구조 생성 (빈 폴더라도)
4. `.gitignore` 작성
5. `requirements.txt` / `requirements-colab.txt` 작성
6. `configs/config.yaml` 초안 작성
7. `src/utils/environment.py` 작성 (IS_COLAB 분기)
8. `src/utils/cache.py` 작성 (embedding load/save helper)

**결과물**: 빈 골격이지만 구조가 완성된 리포

**requirements.txt (로컬)**:
```
numpy
pandas
scipy
statsmodels
scikit-learn
matplotlib
seaborn
pyyaml
tqdm
```

**requirements-colab.txt**:
```
torch
transformers>=4.35
datasets
accelerate
bitsandbytes
pillow
ftfy
regex
```

#### Day 2 (Colab) — 환경 테스트

**할 일**:
1. Colab Pro 노트북 생성 (`notebooks/colab_stage1_structural_gap.ipynb`)
2. Google Drive 마운트
3. GitHub 리포 clone
4. `requirements-colab.txt` 설치
5. CLIP 로드 → 더미 이미지 1장으로 embedding 추출 테스트
6. 성공하면 커밋 & 다음 단계로

**성공 기준**: CLIP embedding shape이 `(1, 512)`인지 확인

#### Day 3-4 (Colab) — COCO embedding 추출

**할 일**:
1. `src/stage1_structural_gap/extract_coco_embeddings.py` 작성
2. COCO val에서 2,000 pair 로드 (caption은 하나씩만)
3. 이미지 batch (32개씩) + 텍스트 batch (64개씩)로 embedding 추출
4. `cache/embeddings/coco_image_embeds.pt`, `coco_text_embeds.pt` 저장
5. Embedding shape 및 값 분포 확인

**체크리스트**:
- [ ] Embedding shape `(2000, 512)`
- [ ] NaN/Inf 없음
- [ ] L2-normalize 여부 config와 일치
- [ ] Drive에 저장 완료

#### Day 5 (Colab or Local) — Centroid & Gap 계산

GPU 불필요. 이미 저장한 embedding으로 CPU에서도 가능.

**할 일**:
1. `src/stage1_structural_gap/compute_centroids.py` 작성
2. $\mu_v = $ image embeddings 평균
3. $\mu_t = $ text embeddings 평균
4. $G = \|\mu_v - \mu_t\|$, $G_{\cos} = 1 - \cos(\mu_v, \mu_t)$
5. $\hat{g} = (\mu_t - \mu_v) / \|\cdot\|$
6. `cache/centroids/`에 저장
7. `outputs/stage1/structural_gap_summary.json`에 수치 기록

**예상 결과 (참고)**:
- $G$: 0.5 ~ 0.9 범위
- $G_{\cos}$: 0.3 ~ 0.7 범위
- 이 범위에서 크게 벗어나면 코드 점검

#### Day 6-7 (로컬) — Robustness 검증

**할 일**:
1. `src/stage1_structural_gap/verify_robustness.py` 작성
2. 2,000개를 1,000 + 1,000으로 split
3. 각 subset에서 $\hat{g}_1$, $\hat{g}_2$ 계산
4. $\cos(\hat{g}_1, \hat{g}_2)$ 측정
5. **성공 기준**: cosine similarity > 0.95
6. 실패 시: 샘플 수를 5,000으로 늘리고 재시도

**Week 1 완료 시 결과물**:
- `outputs/stage1/structural_gap_summary.json`
- `outputs/stage1/robustness_report.json`
- `cache/centroids/` (mu_v, mu_t, g_hat 저장)
- **Section 7 성공 기준 1, 6 판정 완료**

---

### 🟡 Week 2: Stage 2 (Residual) + Stage 3 (BLIP-2)

#### Day 8 (로컬) — VQAv2 샘플 선정 스크립트 준비

**할 일**:
1. VQAv2에서 100개 뽑는 로직 작성 (난이도 다양성 반영)
2. COCO reference 이미지 ID와 겹치지 않는지 확인
3. 최종 100개 샘플 목록을 `data/eval/vqav2_selection.json`으로 저장

#### Day 9 (Colab) — Eval embedding 추출

**할 일**:
1. `notebooks/colab_stage2_eval_embeddings.ipynb` 작성
2. 위에서 선정한 100개 샘플의 이미지와 질문 embedding
3. `cache/embeddings/eval_image_embeds.pt`, `eval_text_embeds.pt` 저장

**주의**: 질문 텍스트는 그대로 CLIP에 넣는다 (prefix/prompt 추가 금지).

#### Day 10-11 (로컬) — Residual 계산

GPU 불필요. 이미 저장한 embedding으로 순수 PyTorch CPU 연산.

**할 일**:
1. `src/stage2_residual/compute_residuals.py` 작성
2. 각 샘플에 대해:
   - $d_i = v_i - t_i$
   - $r_i^{\parallel} = d_i \cdot \hat{g}$ (signed!)
   - $r_i^{\perp} = \|d_i - r_i^{\parallel} \hat{g}\|$
3. `outputs/stage2/residuals.csv`에 저장

**검증 체크리스트**:
- [ ] `r_parallel`에 양수와 음수가 **모두** 존재
- [ ] `r_perp` > 0 for all
- [ ] 분포 histogram 확인 (degenerate 아닌지)

#### Day 12-14 (Colab) — BLIP-2 Inference

**할 일**:
1. `notebooks/colab_stage3_blip2_inference.ipynb` 작성
2. **중요**: CLIP 언로드 후 BLIP-2 로드 (`del clip_model; torch.cuda.empty_cache()`)
3. 각 샘플에 대해:
   - BLIP-2로 답변 생성 (greedy, output_scores=True)
   - mean_entropy, first_token_entropy 계산
4. `outputs/stage3/blip2_outputs.csv` 저장

**Entropy 계산 주의사항** (02_AI_ASSISTANT_GUIDELINES.md Section 3.3 참조):
- Logits를 `.float()`으로 캐스팅 후 softmax
- `log(prob + 1e-12)` epsilon 필수
- Greedy decoding (`do_sample=False`)

**Week 2 완료 시 결과물**:
- `outputs/stage2/residuals.csv`
- `outputs/stage3/blip2_outputs.csv`
- 통합 테이블 초안 (results.csv의 일부 컬럼 채움)

---

### 🟡 Week 3: Stage 4 (Correctness) + Stage 5 초기 분석

#### Day 15 (로컬) — Judgment Helper 준비

**할 일**:
1. `src/stage4_correctness/judgment_helper.py` 작성
   - CSV 읽어서 하나씩 보여줌
   - 이미지, 질문, 정답, BLIP-2 답변 표시
   - 0/1/2 입력 받기
   - 노트 필드도 있으면 유용
2. 판정 기준 문서화 (`outputs/stage4/judgment_criteria.md`)
   - 예: "정답이 'red'인데 모델이 'red car' → 2"
   - 예: "정답이 'two'인데 모델이 'several' → 1"

#### Day 16-17 (로컬) — 수동 판정 수행

**할 일**:
- 100개 샘플을 하루 50개씩 2일에 걸쳐 판정
- 집중력 유지를 위해 한 번에 30분 이상 하지 말 것
- `outputs/stage4/correctness.csv` 생성

**심리 팁**: 판정 도중 결과를 미리 보지 말 것 (bias 유입). 순수하게 BLIP-2 답변이 맞는지만 판단.

#### Day 18 (로컬) — 기초 통계 확인

**할 일**:
1. 모든 stage 결과를 `results.csv`로 통합
2. 각 컬럼의 분포 히스토그램 그리기
3. **이상 징후 체크**:
   - `r_parallel`이 모두 같은 부호 → gap direction 버그 의심
   - `entropy`가 모두 비슷 → BLIP-2 출력 이상
   - `semantic_correct`가 모두 0 또는 모두 2 → 샘플 선정 치우침

이상이 발견되면 이 시점에서 멈추고 파이프라인 재점검.

#### Day 19-21 (로컬) — 1차 상관 분석

**할 일**:
1. Spearman correlation 계산:
   - `r_parallel` vs `mean_entropy`, `first_token_entropy`
   - `r_perp` vs entropy (동일)
   - `r_parallel` vs `semantic_correct`
   - `r_perp` vs `semantic_correct`
2. 각 correlation의 p-value 함께 보고
3. Scatter plot 그리기 (숫자만 믿지 말 것)
4. `outputs/stage5/correlations.json` 저장

**Week 3 완료 시 결과물**:
- `results.csv` 완성 (모든 컬럼 채움)
- `outputs/stage5/correlations.json`
- 1차 scatter plot들

---

### 🟢 Week 4: 심화 분석 + 문서화

#### Day 22-23 (로컬) — Q4 검증 (핵심)

**할 일**:
1. Logistic regression 두 개 비교:
   - Model A: `semantic_correct_binary ~ mean_entropy`
   - Model B: `semantic_correct_binary ~ mean_entropy + r_parallel + r_perp`
   - Binary 변환: correct ∈ {2} → 1, else → 0
2. Pseudo R², AIC, likelihood ratio test로 model B 추가 설명력 확인
3. Confident wrong 분석:
   - `mean_entropy`가 낮은데 `semantic_correct = 0`인 샘플 식별
   - 이들에서 `r_parallel` 또는 `r_perp`가 높은지 확인
   - 몇 개 샘플은 질적으로 케이스 분석 (왜 틀렸는지)
4. `outputs/stage5/regression_results.json` 저장

#### Day 24-25 (로컬) — 종합 해석

**할 일**:
1. Parallel vs Perpendicular 비교:
   - 어느 성분이 더 강한 signal인가?
   - 두 성분이 서로 독립적인가 (둘 다 유의?)
2. Section 7의 6가지 성공 기준 각각 통과 여부 명시
3. Null result도 honest하게 정리:
   - "Entropy 상관은 유의하나 correctness 상관은 약함" 같은 중간 결과도 기록
4. 최종 figure 3~5개 준비 (논문/발표용 품질)

#### Day 26-28 (로컬) — 문서 작성

**할 일**:
1. **Pilot 리포트** (5~8페이지, markdown 또는 docx)
   - 연구 질문
   - 방법
   - 결과 (수치 + 그림)
   - 해석
   - 다음 단계 제안
2. **지도 미팅 슬라이드** (5~8장)
3. **다음 단계 계획**:
   - 성공 시: scale up to 500 samples, 다른 CLIP variant 실험, BLIP-2 외 모델 추가
   - 중간 결과: 조건부 상관 분석, 다변량 신호 결합
   - 실패 시: 가설 재검토, 대안 signal 탐색

**Week 4 완료 시 결과물**:
- Pilot 리포트
- 슬라이드
- `results.csv` 최종본
- 모든 분석 결과 JSON 및 figure

---

## 6. 중간 체크포인트

각 주의 마지막 날, 다음 질문에 답해보기:

### Week 1 체크
- Structural gap이 안정적으로 나왔는가? (robustness cosine > 0.95)
- **안 나왔다면**: 샘플 수 늘리기 또는 embedding normalize 방식 재검토

### Week 2 체크
- Residual 분포가 degenerate하지 않은가?
- BLIP-2 답변이 "yes"/"no"로만 치우치지 않았는가?
- **이상 있으면**: 샘플 다양성 재검토

### Week 3 체크
- 1차 correlation에서 방향성이 보이는가? (부호라도)
- **전혀 안 보이면**: 가설 자체 재검토 (Plan B: 다변량 신호로 전환)

### Week 4 체크
- Q4가 성립하는가? (residual이 entropy 대비 추가 정보)
- **성립하면**: 강한 pilot 결과로 본 실험 준비
- **안 성립하면**: "entropy로 충분한 케이스"와 "residual이 필요한 케이스"를 조건부로 분석

---

## 7. 자주 발생할 문제와 대처

| 문제 | 원인 | 대처 |
|:---|:---|:---|
| Colab OOM | BLIP-2와 CLIP 동시 로드, FP32 사용 | 각 모델을 별도 세션에서 로드, 4-bit quantization 확인 |
| Embedding shape 이상 | batch 처리 실수 | 단일 샘플로 먼저 테스트, 그 다음 batch |
| Robustness cosine < 0.95 | 샘플 수 부족 또는 normalize 누락 | 5,000개로 증가, normalize 여부 재확인 |
| 모든 `r_parallel`이 양수 | gap vector 방향 뒤집힘 | $\hat{g} = \mu_t - \mu_v$인지 확인 (반대 방향 아님) |
| BLIP-2 답변이 비어있음 | Prompt format 오류 | BLIP-2 Question-Answering prompt format 확인 |
| Entropy가 모두 0 | Greedy에서 one-hot, log 처리 이슈 | temperature 체크, `+ 1e-12` 확인 |
| Judgment 진행 느림 | UI 없이 CSV만 편집 | 간단한 CLI 도우미 작성, 30분 단위로 분할 |
| Drive 동기화 꼬임 | 로컬/Colab 동시 수정 | 한 번에 한 곳에서만 편집 |

---

## 8. "시작하기" 체크리스트

오늘/내일 당장 할 수 있는 것:

- [ ] GitHub 리포 생성 (private)
- [ ] 로컬에 clone, 위의 폴더 구조 생성 (빈 파일이라도)
- [ ] `.gitignore` 작성 & commit
- [ ] 이 3개 MD 파일 리포에 추가 & commit
- [ ] `requirements.txt`, `requirements-colab.txt` 작성
- [ ] `configs/config.yaml` 초안 작성
- [ ] Colab Pro 노트북 하나 만들어서 GitHub clone + Drive 마운트 테스트
- [ ] CLIP 로드 테스트 (더미 이미지 1장으로)

이 체크리스트가 끝나면 Day 3로 진입할 준비 완료.

---

## 9. 기간 요약

| 주차 | 주요 내용 | 로컬/Colab 비중 |
|:---:|:---|:---:|
| **Week 1** | 세팅 + Stage 1 (Structural Gap) | 로컬 40% / Colab 60% |
| **Week 2** | Stage 2 + Stage 3 | 로컬 50% / Colab 50% |
| **Week 3** | Stage 4 + 1차 분석 | 로컬 90% / Colab 10% |
| **Week 4** | 심화 분석 + 문서화 | 로컬 100% |

**전체 4주 = 28일**.

---

## 10. 마지막 조언

- **Week 1만 끝내도 큰 진전이다.** Structural gap이 안정적으로 관찰되면 지난 pilot보다 훨씬 단단한 기반이 생긴 것. 거기서 멈춰서 지도 교수님께 중간 보고해도 된다.
- **한 번에 모든 stage를 생각하지 말 것.** 한 주 단위로 focus. "이번 주는 Stage 1만" 하는 마음가짐.
- **결과가 나올 때마다 commit & push.** Colab 런타임이 끊겨도 데이터는 Drive에, 코드는 Git에 있어야 안심.
- **판정은 정직하게.** 수동 judgment에서 "이거 맞다고 봐주고 싶다"는 유혹을 조심. bias 유입은 연구 전체를 오염시킨다.

---
