# Modality Gap as Uncertainty Signal: Pilot Study

## 프로젝트 전체 개요

---

## 0. 프로젝트의 출발점

본 프로젝트는 다음 질문에 답하기 위한 pilot 연구이다.

> **"멀티모달 모델에서 Modality Gap은 정량적으로 측정 가능한가, 그리고 그 측정값은 추론 단계에서 불확실성 신호(uncertainty signal)로 활용될 수 있는가?"**

이 질문은 상위 연구(iNES 개인 연구: Adaptive Uncertainty Mediation for Multimodal Agents)의 **가장 근본 전제**를 검증한다. 상위 연구는 "input-side signal로 uncertainty를 감지하고, 원인에 따라 Captioning/RAG/Sampling 등 조건부 중재를 수행하는 에이전트"를 설계하는데, 이 모든 것이 성립하려면 먼저 **"CLIP embedding space의 기하학적 구조가 downstream 생성 모델의 uncertainty를 예측할 수 있다"**는 가설이 경험적으로 뒷받침되어야 한다. 본 pilot은 그 가설을 최소 단위로 검증하는 것을 목표로 한다.

---

## 1. 핵심 연구 질문

본 연구는 두 단계로 질문을 분리한다.

### RQ1. Structural Modality Gap은 정량적으로 측정 가능한가?

선행연구(Liang et al., 2022)에서 제시한 modality gap을 embedding space 수준에서 구조적으로 정의하고, reference corpus에서 실제로 안정적으로 측정되는지 확인한다.

### RQ2. Structural Modality Gap 위에서 정의한 sample-level residual signal은 추론 불확실성과 관련이 있는가?

단순한 semantic distance가 아니라, **구조적 modality gap 방향으로의 sample-level 편차**로 정의된 신호가 entropy 및 오류 가능성과 연결되는지 검증한다.

---

## 2. 연구의 핵심 개념 구분

본 연구는 다음 두 가지 개념을 명시적으로 분리한다.

### Structural Modality Gap (집단 수준)

CLIP embedding space에서 image embedding 분포와 text embedding 분포 사이의 **구조적 분리**. Liang et al. (2022)이 논의한 modality-level 현상이며, 본 연구에서는 centroid 차이를 통해 정량화한다.

### Sample-level Residual Signal (샘플 수준)

개별 샘플이 위 구조적 gap 방향 안에서 **평균적 정렬 상태로부터 얼마나 더 벗어나는지**를 나타내는 신호. 추론 단계에서 사용할 수 있는 **input-side signal**로 정의된다.

---

## 3. 수식 정의

### 3.1 Structural Gap (Stage 1에서 계산)

CLIP embedding space에서:
- Image embedding centroid: $\mu_v$
- Text embedding centroid: $\mu_t$

Structural gap baseline:

$$G = \|\mu_v - \mu_t\|$$

또는 cosine 기반:

$$G_{\cos} = 1 - \cos(\mu_v, \mu_t)$$

Structural gap direction vector:

$$\hat{g} = \frac{\mu_t - \mu_v}{\|\mu_t - \mu_v\|}$$

### 3.2 Sample-level Residual (Stage 2에서 계산)

각 샘플 $i$에 대해 $d_i = v_i - t_i$로 두고:

**Parallel component (signed):**

$$r_i^{\parallel} = d_i \cdot \hat{g}$$

**Perpendicular component (magnitude):**

$$r_i^{\perp} = \|d_i - r_i^{\parallel} \hat{g}\|$$

**중요**:
- Parallel component는 **signed 값으로 유지**한다 (`abs()` 사용 금지).
- Perpendicular component는 magnitude이므로 항상 양수.
- 두 성분 모두 기록하여 어느 쪽이 uncertainty와 더 강한 상관을 갖는지 분석한다.

### 3.3 Uncertainty Metrics (Stage 3에서 계산)

BLIP-2 생성 출력에 대해:
- **Mean token entropy**: 전체 생성 토큰의 평균 entropy
- **First token entropy**: 첫 생성 토큰의 entropy

두 metric을 모두 기록하여 비교한다.

### 3.4 Correctness (Stage 4에서 판정)

- **Semantic correctness** (주 metric): 수동 판정, 3단계 (0 = incorrect, 1 = partial, 2 = correct)
- **Exact correctness** (보조): 문자열 일치 여부 (0/1)

---

## 4. 핵심 가설

본 연구는 다음 인과 사슬을 **가설**로 둔다 (증명된 사실이 아님).

> **Structural misalignment 증가 → training distribution에서의 이탈 가능성 증가 → generative uncertainty 증가**

즉, 구조적 modality gap 방향으로 크게 벗어난 샘플은:
- Training 중 자주 본 alignment region에서 멀리 떨어져 있을 가능성이 높고
- 생성 시 답변 후보 분포가 더 퍼질 가능성이 있으며
- 결과적으로 entropy가 증가하거나 오류 가능성이 높아질 수 있다.

이 가설이 경험적으로 성립하는지를 Stage 5의 분석에서 검증한다.

---

## 5. CLIP과 BLIP-2를 함께 쓰는 이유

- **CLIP**: cross-modal alignment가 명시적으로 학습된 대표 embedding space 제공 → structural gap 측정에 적합
- **BLIP-2**: 실제 생성 기반 VQA 모델로서 downstream uncertainty와 오류를 관찰 가능

본 연구의 관심은 동일 모델 내부 신호가 아니라, **"CLIP space의 input-side structural signal이 downstream generative uncertainty를 예측할 수 있는가"**이다. 두 모델을 쓰는 것은 한계가 아니라 연구의 의도된 설계다.

---

## 6. 분석 질문 (실험으로 검증할 것들)

| ID | 질문 | 검증 방법 |
|:---:|:---|:---|
| Q1 | Structural modality gap $G$는 reference corpus에서 안정적으로 측정되는가? | Stage 1에서 subset split 비교로 robustness 확인 |
| Q2 | Residual signal이 증가할수록 entropy가 증가하는가? | Stage 5에서 Spearman correlation |
| Q3 | Residual signal이 증가할수록 semantic correctness가 감소하는가? | Stage 5에서 Spearman correlation |
| Q4 | Residual signal은 output entropy와 **다른** 정보를 주는가? | Logistic regression: `correctness ~ entropy` vs `correctness ~ entropy + residual`의 설명력 비교 |
| Q5 | Residual signal은 생성 **전** 계산 가능한 input-side signal로서 가치가 있는가? | Q4 결과 + 실용적 활용 시나리오 분석 |

---

## 7. Pilot의 최소 성공 기준

Pilot은 "증명"이 아니라 "가설이 테스트 가능한 구조인지 확인"이 목적이다.

1. Structural gap baseline $G$가 안정적으로 계산된다
2. Sample-level residual이 수학적으로 명확히 정의되고 계산된다
3. Semantic correctness metric이 구축된다
4. Residual과 entropy/정답성 사이에 최소한의 방향성이 관찰된다
5. 결과가 약하더라도, 가설을 검증 가능한 구조로 재정의하는 데 성공한다
6. Structural gap baseline $G$가 reference corpus 교체(subset split)에도 안정적으로 관찰된다

---

## 8. 실험 전체 구조 (5 Stages)

```
[Stage 1] Structural Gap 구축      →  μ_v, μ_t, ĝ, G 계산 (COCO reference)
           ↓
[Stage 2] Sample-level Residual    →  각 샘플의 r_parallel, r_perp 계산 (VQAv2 eval)
           ↓
[Stage 3] BLIP-2 Inference         →  답변 생성 + entropy 측정
           ↓
[Stage 4] Correctness 판정         →  수동 semantic judgment
           ↓
[Stage 5] 통합 분석                 →  상관, regression, baseline 비교
```

---

## 9. 주장의 범위 (정직한 포지셔닝)

실험이 성공적으로 끝났을 때 주장할 수 있는 것과 없는 것을 명확히 구분한다.

### ✅ 주장할 수 있는 것
- "Sample-level residual이 uncertainty와 경험적으로 상관된다"
- "이 signal은 생성 전에 계산 가능하므로 input-side uncertainty estimation에 실용적 가치가 있다"
- "Output entropy만으로는 포착되지 않는 추가 정보를 제공한다" (Q4가 성립할 때)

### ❌ 주장하면 안 되는 것
- "Modality gap이 uncertainty의 **원인**이다" (상관 ≠ 인과)
- "구조적 misalignment가 **반드시** uncertainty를 유발한다"

인과를 주장하려면 intervention 실험이 필요하며, 이는 본 pilot의 범위를 벗어난다.

---

## 10. 실행 환경 개요

본 프로젝트는 **로컬 VS Code + Colab 하이브리드** 환경에서 진행한다.

| 영역 | 환경 | 이유 |
|:---|:---|:---|
| 코드 작성/버전 관리 | 로컬 VS Code | Codex / Claude Code 활용, Git 관리 |
| CLIP embedding 추출 | Colab Pro (GPU) | 모델 로딩에 GPU 필요 |
| BLIP-2 inference | Colab Pro (GPU) | 4-bit quantization도 GPU 필수 |
| Residual 계산, 통계 분석 | 로컬 VS Code | 순수 NumPy/pandas, GPU 불필요 |
| 수동 semantic judgment | 로컬 (CSV 편집 또는 간단 UI) | 집중 필요, 로컬이 편함 |
| 데이터 동기화 | Google Drive + GitHub | 런타임 끊김 대비, 코드 버전 관리 |

구체적인 워크플로우와 폴더 구조는 `03_DEVELOPMENT_PLAN.md`에 기술한다.

---

## 11. 용어 정리 (Glossary)

| 용어 | 정의 |
|:---|:---|
| **Structural Modality Gap ($G$)** | Reference corpus에서 계산된 image/text centroid 간 거리. Dataset-level 양. |
| **Gap direction ($\hat{g}$)** | $\mu_t - \mu_v$의 unit vector. Residual projection에 사용. |
| **Residual signal ($r_i$)** | 개별 샘플이 gap 구조에서 벗어난 정도. Parallel과 perpendicular 성분으로 분해. |
| **Input-side signal** | 생성 전에 계산 가능한 uncertainty 지표. (↔ output-side signal) |
| **Semantic correctness** | 문자열이 아닌 의미 기준 정답 판정 (0/1/2 3단계) |
| **Reference corpus** | Structural gap 계산에 쓰이는 대규모 image-text pair 집합. 본 연구에서는 COCO. |
| **Evaluation samples** | Residual signal과 uncertainty를 측정할 샘플 집합. 본 연구에서는 VQAv2. |

---

## 12. 참고 선행연구

- **Liang et al. (2022)**, "Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning" — Structural modality gap의 존재와 원인 규명
- **Farquhar et al. (2024)**, "Detecting hallucinations in large language models using semantic entropy" — Output-side uncertainty 대표 사례 (본 연구의 대조군)
- **BLIP-2 (Li et al., 2023)** — 생성 기반 VQA baseline 모델
- **CLIP (Radford et al., 2021)** — Embedding space 기반

---
