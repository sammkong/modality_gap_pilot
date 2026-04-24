# AI Coding Assistant 작업 지침

## Codex & Claude Code용 Context & Guidelines

이 문서는 Codex와 Claude Code가 본 프로젝트의 코드를 작성/수정할 때 따라야 할 **필수 규칙과 컨텍스트**다. 새 세션을 시작할 때마다 이 문서를 먼저 읽도록 한다.

---

## 0. 이 프로젝트를 처음 만나는 AI에게

당신은 **멀티모달 모델의 uncertainty signal 연구**를 돕고 있다. 구체적 배경은 `01_PROJECT_OVERVIEW.md`와 `03_DEVELOPMENT_PLAN.md`에 있다. 코드를 쓰기 전에 반드시 읽어라.

핵심만 요약하면:
- CLIP으로 image/text embedding을 뽑고, 그로부터 structural modality gap을 계산한다
- 각 샘플의 embedding을 gap 방향으로 projection해서 residual signal을 구한다
- BLIP-2로 답변을 생성하고 entropy를 측정한다
- Residual signal과 entropy/correctness의 관계를 분석한다

---

## 1. 개발/실행 환경 분리 (매우 중요)

본 프로젝트는 **로컬 VS Code에서 개발하고 Colab에서 GPU 작업을 실행**하는 하이브리드 구조다. 이 분리를 코드 작성 단계에서 의식해야 한다.

### 로컬 (VS Code)에서 돌아가는 코드
- Residual signal 계산 (NumPy만 필요)
- 통계 분석 (scipy, statsmodels)
- 시각화 (matplotlib, seaborn)
- 수동 judgment 관련 스크립트
- 데이터 정리/합치기 (pandas)

### Colab에서 돌아가는 코드
- CLIP 모델 로딩 및 embedding 추출
- BLIP-2 모델 로딩 및 inference
- GPU 메모리가 필요한 모든 작업

### 코드 작성 규칙
- **GPU 필요 코드와 그렇지 않은 코드를 파일 단위로 분리**할 것
- Colab 전용 파일은 상단에 다음 주석을 반드시 포함:
  ```python
  # ====================================
  # EXECUTION ENVIRONMENT: Google Colab (GPU required)
  # Do NOT run locally.
  # ====================================
  ```
- 로컬 전용 파일도 마찬가지로 명시:
  ```python
  # ====================================
  # EXECUTION ENVIRONMENT: Local VS Code (CPU only)
  # No GPU dependencies.
  # ====================================
  ```
- 경로는 반드시 환경별 분기 처리:
  ```python
  import os
  IS_COLAB = 'COLAB_GPU' in os.environ
  BASE_DIR = '/content/drive/MyDrive/modality_gap_pilot' if IS_COLAB else './data'
  ```

---

## 2. 필수 기술 규칙

### 2.1 메모리 최적화 (Colab OOM 방지)

**반드시 지켜야 할 것:**

```python
# CLIP 로드 시 FP16
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to("cuda")

# BLIP-2 로드 시 4-bit quantization
from transformers import BitsAndBytesConfig, Blip2ForConditionalGeneration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**반드시 피해야 할 것:**
- CLIP과 BLIP-2를 동시에 메모리에 올리지 말 것 (stage 분리 이유)
- FP32로 CLIP 로드 금지
- BLIP-2를 quantization 없이 로드 금지
- 큰 배치(BLIP-2 batch_size > 1) 금지 — OOM 위험

### 2.2 Embedding 캐시 (반드시 지킬 것)

Colab 런타임은 끊긴다. Embedding을 다시 뽑는 것은 시간 낭비다.

**모든 embedding 계산 코드는 다음 패턴을 따를 것:**

```python
import os
import torch

CACHE_PATH = f"{BASE_DIR}/cache/embeddings/coco_image_embeds.pt"

if os.path.exists(CACHE_PATH):
    print(f"Loading cached embeddings from {CACHE_PATH}")
    embeddings = torch.load(CACHE_PATH)
else:
    print("Computing embeddings...")
    embeddings = compute_embeddings(...)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    torch.save(embeddings, CACHE_PATH)
    print(f"Saved to {CACHE_PATH}")
```

### 2.3 재현성

- Random seed를 모든 스크립트 상단에서 고정:
  ```python
  import torch, numpy as np, random
  SEED = 42
  torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(SEED)
  ```
- 모델 버전은 `configs/config.yaml`에서 관리, 하드코딩 금지.

---

## 3. 수식 구현 시 주의사항 (절대 틀리면 안 되는 것)

### 3.1 Residual Signal 정의

```python
# d_i = v_i - t_i (image - text)
# 이 방향을 바꾸지 말 것. g_hat과의 부호 해석이 꼬인다.
d_i = v_i - t_i

# g_hat = (mu_t - mu_v).normalize() — 이 방향을 바꾸지 말 것
g_hat = (mu_t - mu_v) / torch.norm(mu_t - mu_v)

# Parallel: SIGNED value (절댓값 금지!)
r_parallel = torch.dot(d_i, g_hat)  # scalar, can be negative

# Perpendicular: magnitude (항상 양수)
d_perp_vec = d_i - r_parallel * g_hat
r_perp = torch.norm(d_perp_vec)
```

**핵심**:
- `abs(r_parallel)`을 쓰면 연구의 핵심 가설을 깨뜨린다. 절대 사용하지 말 것.
- `r_perp`는 magnitude이므로 `abs()` 불필요.

### 3.2 CLIP Embedding 정규화

CLIP은 관례상 cosine similarity를 쓰므로 보통 L2-normalize한다. 하지만 **centroid 계산 시점**에서는 주의가 필요하다:

```python
# 옵션 A: normalize 후 centroid (권장 — 기존 논문 convention과 일치)
image_embeds_normalized = F.normalize(image_embeds, dim=-1)
mu_v = image_embeds_normalized.mean(dim=0)

# 옵션 B: raw embedding에서 centroid
mu_v = image_embeds.mean(dim=0)
```

**본 프로젝트는 옵션 A(normalize 후 centroid)를 사용**한다. Liang et al. (2022)과 일치시키기 위함. 코드에서 반드시 이 순서를 지킬 것.

### 3.3 Entropy 계산

BLIP-2 generation에서 entropy를 정확히 뽑는 방법:

```python
# generate 시 scores 반환하도록 설정
outputs = model.generate(
    **inputs,
    max_new_tokens=20,
    output_scores=True,
    return_dict_in_generate=True,
    do_sample=False,  # greedy decoding
)

# scores는 각 step의 logits (batch, vocab_size) × num_steps
scores = outputs.scores  # tuple of tensors

# 각 step에서 softmax → entropy
import torch.nn.functional as F
entropies = []
for step_logits in scores:
    probs = F.softmax(step_logits.float(), dim=-1)  # float() 필수 (FP16 numerical issue)
    step_entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    entropies.append(step_entropy.item())

mean_entropy = sum(entropies) / len(entropies)
first_token_entropy = entropies[0]
```

**주의사항**:
- FP16 logits로 softmax하면 numerical instability 발생. `.float()` 필수.
- `1e-12` epsilon 없으면 `log(0) = -inf`.
- Greedy decoding(`do_sample=False`)으로 통일. Stochastic sampling은 entropy 해석을 복잡하게 만듦.

---

## 4. 데이터 처리 규칙

### 4.1 파일 포맷 통일

- **Embedding**: PyTorch `.pt` (torch.save/load)
- **Metadata, 결과**: CSV (pandas)
- **Config**: YAML
- **중간 분석**: Parquet (대용량일 때만)

### 4.2 Results CSV 스키마

최종 `results.csv`는 다음 컬럼을 포함해야 한다:

```
sample_id, image_path, question, gt_answer, blip2_answer,
r_parallel, r_perp,
mean_entropy, first_token_entropy,
semantic_correct, exact_correct,
notes
```

스키마 변경이 필요하면 코드에서 변경하지 말고 먼저 사용자에게 확인할 것.

### 4.3 경로는 Config에서 읽기

경로 하드코딩 금지. 모든 경로는 `configs/config.yaml`에 두고 코드에서는 로드해서 사용:

```python
import yaml
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

IMAGE_DIR = cfg["paths"]["coco_images"]
```

---

## 5. 코드 스타일 & 구조

### 5.1 스크립트 템플릿

각 stage 스크립트는 다음 구조를 따를 것:

```python
"""
Stage X: [Stage 이름]
Environment: [Local / Colab]
Input: [입력 파일/경로]
Output: [출력 파일/경로]
"""

# 1. Imports
# 2. Config 로드
# 3. Random seed 고정
# 4. 환경 확인 (IS_COLAB 등)
# 5. 메인 로직 (함수로 분리)
# 6. if __name__ == "__main__": 아래에서 실행

def main(cfg):
    ...

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
```

### 5.2 함수 네이밍

- `compute_*`: 새로운 값을 계산 (e.g., `compute_residual`)
- `load_*`: 디스크/캐시에서 불러오기
- `save_*`: 디스크로 저장
- `analyze_*`: 통계/시각화

### 5.3 로깅

`print()` 대신 최소한의 일관된 로깅을 사용:

```python
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"Computing embeddings for {len(samples)} samples")
```

---

## 6. 절대 하지 말아야 할 것 (Red Flags)

### 6.1 연구 설계를 임의로 바꾸지 말 것

다음 사항은 **사용자의 명시적 승인 없이** 절대 변경하지 말 것:

- Residual 정의 (parallel/perpendicular 분해, signed 유지)
- 사용 모델 (CLIP-base, BLIP-2-opt-2.7b)
- 데이터셋 선택 (COCO reference + VQAv2 eval)
- 수식 (특히 $\hat{g}$ 방향, $d_i$ 방향)

만약 "더 좋아 보이는" 대안을 제안하고 싶다면, 코드를 바꾸지 말고 **먼저 사용자에게 물어볼 것**.

### 6.2 침묵 실패 (Silent Failure) 금지

다음은 모두 금지:
- `try/except: pass` (예외를 삼키면 버그 못 잡음)
- 경고 없이 기본값으로 fallback
- 파일이 없을 때 빈 결과로 계속 진행

올바른 방식:
```python
try:
    embeddings = torch.load(CACHE_PATH)
except FileNotFoundError:
    logger.error(f"Cache not found at {CACHE_PATH}. Run Stage 1 first.")
    raise
```

### 6.3 "가짜 결과" 생성 금지

- 테스트를 위한 더미 데이터라도 **실제 결과 CSV에 섞어넣지 말 것**
- 더미는 별도 디렉토리 (`dev/mock_data/`)에 명시적으로 분리
- `print("TODO: implement properly")` 후 빈 값 반환 금지

### 6.4 데이터 유출 방지

- Evaluation 샘플의 정답(gt_answer)을 BLIP-2 prompt에 넣지 말 것 (당연한 얘기지만 실수 자주 함)
- Reference corpus(COCO)와 eval set(VQAv2)의 이미지 ID가 겹치지 않는지 Stage 2 시작 시 확인

---

## 7. 디버깅 & 검증 체크리스트

새 코드를 쓴 후 사용자에게 결과를 보여주기 전 다음을 셀프 체크:

### Stage 1 결과 검증
- [ ] $\mu_v$, $\mu_t$의 shape이 CLIP embedding dimension과 일치 (512)
- [ ] $G = \|\mu_v - \mu_t\|$ 값이 0보다 유의미하게 큼 (0.3 이상 예상)
- [ ] $G_{\cos}$ 값이 0~1 범위 안에 있음
- [ ] Subset split 간 $\hat{g}$ cosine similarity > 0.95

### Stage 2 결과 검증
- [ ] `r_parallel`에 양수와 음수가 **모두** 존재 (전부 양수면 부호 버그)
- [ ] `r_perp`는 모두 양수
- [ ] 분포가 degenerate하지 않음 (histogram으로 확인)

### Stage 3 결과 검증
- [ ] BLIP-2 답변이 빈 문자열이 아님
- [ ] `mean_entropy`, `first_token_entropy`가 양수
- [ ] Entropy 값이 reasonable 범위 (0 ~ log(vocab_size) ≈ 10.8)

### Stage 4 결과 검증
- [ ] 모든 샘플에 judgment가 있음 (NaN 없음)
- [ ] 3단계(0/1/2) 분포가 너무 치우치지 않음

### Stage 5 결과 검증
- [ ] Spearman correlation의 p-value가 함께 보고됨
- [ ] Scatter plot이 시각적으로 확인됨 (correlation 숫자만 믿지 말 것)
- [ ] Logistic regression 결과에서 residual 계수의 유의성 표시

---

## 8. 사용자와의 커뮤니케이션

### 8.1 작업 시작 전 확인

다음 경우에는 코드를 쓰기 전에 사용자에게 먼저 질문:
- 연구 설계에 영향을 주는 선택 (위 6.1 참조)
- 여러 구현 방식이 가능할 때 (특히 수식 구현)
- 예상치 못한 데이터 이슈 발견 시

### 8.2 결과 보고 시

숫자만 보여주지 말고 **해석까지** 제공:

```
Nope (나쁨):
"Spearman correlation = 0.34"

좋음:
"Spearman correlation = 0.34 (p=0.002, n=100)
→ 약한~중간 수준의 양의 상관. 가설 방향과 일치하지만,
  Section 16의 성공 기준 중 4번('최소한의 방향성')은 통과.
  단, 단독 signal로 강하다고 주장하기는 부족."
```

### 8.3 불확실할 때

"이 선택이 맞는지 확신이 없으면 추측하지 말고 물어볼 것." 본 프로젝트는 **정확성이 속도보다 중요하다.** Pilot의 결과가 연구 전체 방향을 결정한다.

---

## 9. 파일 수정 시 Git 관리

- 각 stage가 끝날 때마다 commit
- Commit 메시지 규칙: `[Stage X] <간단한 설명>`
  - 예: `[Stage 1] Compute structural gap on COCO 2k pairs`
- 대용량 파일(embedding `.pt`, model weight)은 `.gitignore`에 추가
- `results.csv`는 커밋해도 됨 (작고 버전 관리 가치 있음)

`.gitignore` 기본:
```
cache/
outputs/embeddings/
*.pt
*.pth
__pycache__/
.ipynb_checkpoints/
```

---

## 10. 빠른 참조: 현재 단계 확인

코드 작성 전 현재 어느 stage인지 확인:

1. `03_DEVELOPMENT_PLAN.md`의 로드맵 확인
2. `outputs/` 폴더에서 완료된 stage 확인
3. 불확실하면 사용자에게 "지금 Stage X 작업 중인가요?" 확인

**각 stage는 이전 stage의 출력에 의존한다. 순서를 건너뛰지 말 것.**

---

## 마지막 원칙

> **"빠르고 잘못된 코드"보다 "느리고 정확한 코드"가 낫다.**
> 이 프로젝트는 연구의 근본 전제를 검증한다. 코드의 작은 실수가 연구 결론을 바꿀 수 있다.
> 확신이 없으면 멈추고 물어라.

---
