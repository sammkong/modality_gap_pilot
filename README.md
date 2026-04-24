# Modality Gap as Uncertainty Signal Pilot

멀티모달 모델의 embedding space에서 관찰되는 structural modality gap이 downstream 생성 단계의 uncertainty signal로 활용될 수 있는지 검증하기 위한 pilot 연구 프로젝트

## 연구 질문

1. CLIP embedding space에서 image/text 분포 사이의 structural modality gap을 안정적으로 측정할 수 있는가?
2. 개별 샘플의 residual signal이 BLIP-2의 entropy 및 정답성과 관련되는가?

## 핵심 아이디어

- CLIP으로 image embedding과 text embedding을 추출한다.
- reference corpus에서 centroid를 계산해 structural gap과 gap direction을 정의한다.
- evaluation sample마다 residual을 parallel / perpendicular 성분으로 분해한다.
- BLIP-2로 답변을 생성하고 token entropy를 측정한다.
- residual, entropy, correctness 사이의 관계를 분석한다.

## 사용 모델

- CLIP: `openai/clip-vit-base-patch32`
- BLIP-2: `Salesforce/blip2-opt-2.7b`