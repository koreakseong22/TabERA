# TabERA

**Tabular Explainable Retrieval Architecture**
Dual-Space Centroid 기반 설명가능 검색 아키텍처

---

## 핵심 아이디어

TabERA는 세 가지 인지과학 이론에서 영감을 받은 **cluster-conditioned exemplar retrieval** 아키텍처입니다.

> 인간은 새로운 자극을 처리할 때, 추상화된 그룹 표상을 먼저 참조한 뒤 구체적 사례와 비교한다.

데이터에서 centroid를 발견하고, centroid를 검색 제한·설명의 앵커로 사용합니다.

```
① 전역 맥락 먼저   →  centroid_x: "이 샘플은 alcohol=10.24 그룹에 속한다"
② 구체적 사례 비교  →  FeatureStore: "이웃 #1의 fixed_acidity=7.1"
③ feature 기여도   →  FeatureCrossAttention: "volatile_acidity 15.1%"
```

### 인지과학적 근거

TabERA is inspired by several cognitive theories that describe how humans
organize and utilize category-level information.

**Central Tendency Effect** (Posner & Keele, 1968): Humans tend to form
internal representations corresponding to the average structure of observed
stimuli. In TabERA, the EMA-based centroid update can be interpreted as
maintaining such central tendency representations in a computational form.

**Schema Theory** (Bartlett, 1932): Humans rely on abstract structural
representations when processing new inputs, before considering individual
instances. Similarly, TabERA performs centroid-based routing prior to
instance-level retrieval, introducing a hierarchical processing structure.

**Dual-Process Theory** (Kahneman, 2011): Human cognition distinguishes
between fast, coarse-grained judgments (System 1) and slower, analytical
reasoning (System 2). In TabERA, centroid routing can be viewed as a coarse
filtering step (System 1) that restricts the search space for subsequent
fine-grained retrieval (System 2).

> We emphasize that these connections are not intended as direct cognitive
> models, but rather as conceptual inspirations that guided the design of
> the architecture.

---

## 핵심 설계

| 구성요소 | 파일 | 이론 근거 |
|---|---|---|
| **CentroidLayer** (Dual-Space) | `libs/prototypes.py` | Central Tendency Effect (Posner & Keele, 1968) |
| **EMA (centroid_x 전용)** | `libs/prototypes.py` | 점진적 평균 추출 — 설명의 faithfulness 보장 |
| **centroid-first routing** | `libs/prototypes.py` | Schema Theory (Bartlett, 1932) |
| **STE Routing** | `libs/prototypes.py` | Bengio et al. (2013) + VQ-VAE (van den Oord, 2017) |
| **AttentionAggregator** | `libs/evidence.py` | Scaled dot-product attention |
| **MemoryBank** | `libs/tabera.py` | Dual-Process System 2 — 그룹 내 세밀 검색 |
| **FeatureStore** | `libs/tabera.py` | 이웃 원본 feature 값 조회 |

### Dual-Space Centroid

```
centroid_emb (P, D)  — nn.Parameter, gradient + commitment loss로 학습
                        STE routing, MemoryBank 그룹핑에 사용

centroid_x   (P, F)  — register_buffer, EMA로만 갱신
                        실제 데이터 평균 feature 값 추적
                        역정규화 후 "alcohol=10.24" 형태로 출력
```

Dual-Space 분리는 **engineering 결정**입니다. 연산 공간(centroid_emb)과
해석 공간(centroid_x)을 분리함으로써 설명의 faithfulness를 보장합니다.

### STE (Straight-Through Estimator)

```python
# VQ-VAE 표준 설계 (van den Oord, 2017)
hard = one_hot(argmax(logits))          # forward: 이산 결정
soft = softmax(logits)                   # backward: gradient 운반
routing = soft + (hard - soft).detach() # STE
```

Gumbel-Softmax 대신 STE를 사용하는 이유:
- 설계가 요구하는 것은 **hard routing** (그룹 배정)
- Gumbel은 불필요한 noise와 파라미터(τ) 추가
- VQ-VAE가 검증한 표준 설계

---

## 폴더 구조

```
TabERA/
├── libs/
│   ├── tabera.py            # TabERA 모델 (TabERA, MemoryBank, FeatureStore)
│   ├── prototypes.py        # CentroidLayer (STE, KMeans++, EMA, Dual-Space)
│   ├── evidence.py          # AttentionAggregator, FeatureCrossAttention
│   ├── supervised.py        # TabERAWrapper (학습 루프, EMA 호출)
│   ├── eval.py              # 평가 지표 (CE/BCE/MSE)
│   ├── search_space.py      # Optuna HPO 탐색 공간 (12개 파라미터)
│   └── data.py              # OpenML 데이터 로더
├── optim_logs/
│   └── seed={seed}/
│       ├── data={id}..model=tabera.pkl               # HPO study (Optuna)
│       ├── data={id}..seed={seed}..model=tabera.csv  # trial별 결과
│       ├── data={id}..seed{seed}_model_state.pt      # 최적 model state
│       ├── data={id}..seed{seed}_preds.npy           # test 예측값
│       └── data={id}..seed{seed}_meta.pkl            # 평가 지표
├── figures/
│   └── seed={seed}/
│       ├── A_embed_{id}_seed{seed}_{proj}.png        # 임베딩 공간 구조
│       ├── B_centroid_{id}_seed{seed}_{proj}.png     # Centroid 발견
│       ├── C_retrieval_{id}_seed{seed}_{proj}.png    # Retrieval
│       └── data_{id}_seed{seed}_{proj}.pkl           # 투영 데이터 캐시
├── optimize.py              # HPO 실행
├── reproduce.py             # Best config 재현 + model state 저장
├── visualize_embeddings.py  # 3-Figure 임베딩 시각화
├── ensemble.py              # 예측 앙상블
├── fetch_tabzilla.py        # TabZilla 데이터셋 목록 조회
├── run_tabzilla.ps1         # TabZilla 벤치마크 실행 (TabERA)
└── requirements.txt
```

---

## 설치

```powershell
# Python 3.12.10 권장
python -m venv venv
venv\Scripts\activate

# PyTorch (CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 나머지 패키지
pip install -r requirements.txt
```

> **주의**: conda base가 활성화된 경우 먼저 `conda deactivate` 후 venv를 활성화하세요.
> pip 대신 `python -m pip`를 사용하세요.

---

## 실행 방법

### 1. HPO (Hyperparameter Optimization)

```powershell
python optimize.py --gpu_id 0 --openml_id 43986 --n_trials 100
```

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--gpu_id` | GPU 인덱스 (-1 = CPU) | 0 |
| `--openml_id` | OpenML 데이터셋 ID | 필수 |
| `--n_trials` | Optuna trial 수 | 100 |
| `--seed` | 랜덤 시드 | 1 |

결과:
```
optim_logs/seed=1/data={id}..model=tabera.pkl
optim_logs/seed=1/data={id}..seed=1..model=tabera.csv
```

### 2. Best Config 재현 + Model State 저장

```powershell
python reproduce.py --gpu_id 0 --openml_id 43986 --seed 1
```

`--explain` 플래그를 추가하면 테스트 샘플별 feature 기여도 설명이 출력됩니다.

```powershell
python reproduce.py --gpu_id 0 --openml_id 43986 --seed 1 --explain
```

결과:
```
optim_logs/seed=1/data={id}..seed1_preds.npy
optim_logs/seed=1/data={id}..seed1_meta.pkl
optim_logs/seed=1/data={id}..seed1_model_state.pt
```

> `reproduce.py`는 `optimize.py`가 실제 사용한 `n_prototypes`(`n_prototypes_actual`)를
> `user_attrs`에서 복원하므로 HPO와 동일한 설정으로 재현됩니다.

### 3. 임베딩 시각화 (3-Figure)

```powershell
# 첫 실행: reproduce.py 완료 후 model state 로드 (권장)
python visualize_embeddings.py --openml_id 43986 --seed 1 --proj tsne --from_state

# pkl 재사용: Figure 스타일만 수정할 때
python visualize_embeddings.py --openml_id 43986 --seed 1 --proj tsne --from_pkl
```

| 옵션 | 설명 |
|---|---|
| `--proj` | 투영 방법: `pca` 또는 `tsne` |
| `--from_state` | reproduce.py가 저장한 최적 model state 로드 |
| `--from_pkl` | 기존 투영 pkl 재사용 (Figure 스타일 수정 시) |
| `--k_show` | Figure C에서 표시할 이웃 수 (기본값: 5) |

결과:
```
figures/seed=1/A_embed_{id}_seed1_tsne.png    # Figure A: 임베딩 공간 구조
figures/seed=1/B_centroid_{id}_seed1_tsne.png # Figure B: Centroid 발견
figures/seed=1/C_retrieval_{id}_seed1_tsne.png # Figure C: Retrieval
```

**seed=1~5 전체 파이프라인 실행:**

```powershell
foreach ($seed in 1..5) {
    Write-Host "=== seed=$seed ===" -ForegroundColor Cyan
    python optimize.py             --openml_id 43986 --seed $seed
    python reproduce.py            --openml_id 43986 --seed $seed
    python visualize_embeddings.py --openml_id 43986 --seed $seed --proj tsne --from_state
}
```

### 4. TabZilla 벤치마크

```powershell
# TabERA 30개 데이터셋 실행 (완료된 것은 자동 스킵)
.\run_tabzilla.ps1
```

---

## 설명 출력 예시

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TabERA Explanation — Sample #0 (Wine Quality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① 전역 맥락  [centroid_x — Centroid 수준]
   Centroid_3  (confidence=94.3%)
   alcohol=10.24,  pH=3.31,  fixed_acidity=7.21

② 이웃 맥락  [FeatureStore — Exemplar 수준]
   Neighbour_0: 42.1%
      → alcohol=10.41, pH=3.28, fixed_acidity=7.1
   Neighbour_1: 28.3%
      → volatile_acidity=0.28, sulphates=0.63

③ Feature 기여도  [FeatureCrossAttention]
   volatile_acidity  15.1%
   chlorides         14.9%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## HPO 파라미터 (12개)

| 파라미터 | 범위 | 역할 |
|---|---|---|
| `embed_dim` | {64, 128, 256} | 임베딩 차원 |
| `k` | {8, 16, 32, 64} | 검색 이웃 수 |
| `n_prototypes` | {4, 8, 12, 16} | centroid 수 P |
| `loss_diversity` | 1e-4 ~ 5e-1 | centroid 분산 강제 |
| `loss_commitment` | 1e-4 ~ 1e-1 | VQ-VAE commitment loss |
| `lr` | 1e-4 ~ 1e-2 | 학습률 |
| `dropout` | 0.0 ~ 0.5 | dropout 비율 |
| `n_heads` | {1, 2, 4, 8} | Attention head 수 |
| `batch_size` | {128, 256, 512} | 배치 크기 |
| `embedder_layers` | {1, 2, 3, 4} | ResidualMLP 깊이 |
| `weight_decay` | 1e-6 ~ 1e-3 | L2 정규화 |
| `anneal_factor` | 0.85 ~ 0.99 | (하위 호환 유지) |

---

## 실험 결과 요약

### Wine Quality (id=43986, 100 trials)

| 지표 | TabERA | ModernNCA |
|---|---|---|
| acc_best | 0.6785 | 0.7046 |
| **logloss best** | **0.983 ✅** | 1.499 |
| **logloss mean** | **1.471 ✅** | 1.565 |
| acc std | 2.80% | 5.96% |

### TabZilla 24개 데이터셋

| 지표 | 값 |
|---|---|
| acc_best mean | 0.879 |
| acc_best ≥ 0.90 | 10개 (42%) |
| logloss_best mean | 0.346 |
| 평균 학습시간 | 7.1s |

---

## 참고 문헌

- Posner, M. I., & Keele, S. W. (1968). On the genesis of abstract ideas. *Journal of Experimental Psychology*, 77(3), 353-363.
- Bartlett, F. C. (1932). *Remembering: A Study in Experimental and Social Psychology*. Cambridge University Press.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Gorishniy, Y., et al. (2023). TabR: Tabular Deep Learning Meets Nearest Neighbors. arXiv:2307.14338.
- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. arXiv:1308.3432.
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning (VQ-VAE). NeurIPS 2017.
- Arthur, D. & Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. SODA 2007.
- Ye, H., et al. (2024). PTaRL: Prototype-based Tabular Representation Learning. arXiv:2407.05364.
- McElfresh, D., et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? NeurIPS 2023.
- Zhan, X., et al. (2020). Online Deep Clustering for Unsupervised Representation Learning. CVPR 2020.