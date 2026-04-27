# TabERA

**Tabular Explainable Retrieval Architecture**
Dual-Space Prototype 기반 설명가능 검색 아키텍처

---

## 핵심 아이디어

TabERA는 Rosch(1975)의 **Prototype Theory**에서 영감을 받았습니다.

> 인간은 개체를 인지할 때 전형적 prototype과의 유사도를 **먼저** 확인한 후, 구체적 사례와 비교한다.

이 인지 구조를 모델 설계에 직접 반영합니다.

```
① 전역 맥락 먼저   →  centroid_x: "이 샘플은 alcohol=10.24 그룹에 속한다"
② 구체적 사례 비교  →  FeatureStore: "이웃 #1의 fixed_acidity=7.1"
③ feature 기여도   →  FeatureCrossAttention: "volatile_acidity 15.1%"
```

---

## 핵심 설계

| 구성요소 | 파일 | 이론 근거 |
|---|---|---|
| **CentroidLayer** (Dual-Space) | `libs/prototypes.py` | Prototype Theory (Rosch, 1975) |
| **STE Routing** | `libs/prototypes.py` | Bengio et al. (2013) + VQ-VAE (van den Oord, 2017) |
| **EMA (centroid_x 전용)** | `libs/prototypes.py` | 설명의 사실성(faithfulness) 보장 |
| **AttentionAggregator** | `libs/evidence.py` | Scaled dot-product attention |
| **MemoryBank** | `libs/tabera.py` | 캐시 기반 그룹 내 벡터화 검색 |
| **FeatureStore** | `libs/tabera.py` | 이웃 원본 feature 값 조회 |

### Dual-Space Centroid

```
centroid_emb (P, D)  — nn.Parameter, gradient + commitment loss로 학습
                        STE routing, MemoryBank 그룹핑에 사용

centroid_x   (P, F)  — register_buffer, EMA로만 갱신
                        실제 데이터 평균 feature 값 추적
                        역정규화 후 "alcohol=10.24" 형태로 출력
```

### STE (Straight-Through Estimator)

```python
# VQ-VAE 표준 설계 (van den Oord, 2017)
hard = one_hot(argmax(logits))          # forward: 이산 결정
soft = softmax(logits)                   # backward: gradient 운반
routing = soft + (hard - soft).detach() # STE
```

Gumbel-Softmax 대신 STE를 사용하는 이유:
- 가설이 요구하는 것은 **hard routing** (그룹 배정)
- Gumbel은 불필요한 noise와 파라미터(τ) 추가
- VQ-VAE가 검증한 표준 설계

---

## 폴더 구조

```
TabERA/
├── libs/
│   ├── tabera.py        # TabERA 모델 (TabERA, MemoryBank, FeatureStore)
│   ├── prototypes.py    # CentroidLayer (STE, KMeans++, EMA, Dual-Space)
│   ├── evidence.py      # AttentionAggregator, FeatureCrossAttention
│   ├── supervised.py    # TabERAWrapper (학습 루프, EMA 호출)
│   ├── eval.py          # 평가 지표 (CE/BCE/MSE)
│   ├── search_space.py  # Optuna HPO 탐색 공간 (12개 파라미터)
│   └── data.py          # OpenML 데이터 로더
├── optim_logs/          # HPO 결과 (.pkl, .csv)
├── results/             # 재현 결과
├── optimize.py          # HPO 실행
├── reproduce.py         # Best config 재현 + 설명 출력
├── ensemble.py          # 예측 앙상블
├── fetch_tabzilla.py    # TabZilla 데이터셋 목록 조회
├── run_tabzilla.ps1     # TabZilla 벤치마크 실행 (TabERA)
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

결과: `optim_logs/seed=1/data={id}..model=tabhera.pkl`

### 2. Best Config 재현 + 설명 출력

```powershell
python reproduce.py --gpu_id 0 --openml_id 43986 --explain
```

`--explain` 플래그를 사용하면 테스트 샘플별 설명 경로가 출력됩니다.

### 3. TabZilla 벤치마크

```powershell
# TabERA 30개 데이터셋 실행 (완료된 것은 자동 스킵)
.\run_tabzilla.ps1


---

## 설명 출력 예시

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TabERA Explanation — Sample #0 (Wine Quality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① 전역 맥락  [centroid_x — Prototype 수준]
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
| `n_prototypes` | HPO 탐색 (실제: sqrt(N)) | centroid 수 P |
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

- Rosch, E. (1975). Cognitive representations of semantic categories. *Journal of Experimental Psychology: General*, 104(3), 192-233.
- Gorishniy, Y., et al. (2023). TabR: Tabular Deep Learning Meets Nearest Neighbors. arXiv:2307.14338.
- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. arXiv:1308.3432.
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning (VQ-VAE). NeurIPS 2017.
- Arthur, D. & Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. SODA 2007.
- Ye, H., et al. (2024). PTaRL: Prototype-based Tabular Representation Learning. arXiv:2407.05364.
- McElfresh, D., et al. (2023). When Do Neural Nets Outperform Boosted Trees? (TabZilla). NeurIPS 2023.
