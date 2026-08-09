"""
libs/diagnostics.py
===================
TabERA 관찰 계층 (observer).

경계선
──────
```
tabera.py       predictor   tensor → tensor. 설명을 "만들지" 않는다.
diagnostics.py  observer    forward output을 "관찰해서" 설명을 재구성한다.
```

이 파일의 모든 함수는 `model`과 `forward()`의 출력 dict만 받는다. 모델
내부를 고치지 않고, gradient를 만들지 않으며, 학습에 아무 영향이 없다.

왜 이렇게 나누는가
──────────────────
설명 계산이 `forward()` 안에 있으면
  - training/inference API에 설명 요구사항이 섞인다
  - benchmark/serving 코드가 쓰지도 않을 비용을 낸다
  - "prediction graph"와 "diagnostic graph"가 한 함수에 공존한다
  - 모델이 FeatureStore 같은 외부 상태를 알아야 한다
그래서 forward는 예측 상태만 내보내고, 해석은 전부 여기서 한다.

⚠ 복원 가능성의 근거 (구현 전 수치로 확인함)
```
이웃 임베딩   memory.keys[topk_idx]                    = retrieve()의 keys_full[idx]
이웃 유사도   normalize(query_retr)·normalize(위 값)    = retrieve() 내부와 같은 식
이웃 라벨     memory.labels[topk_idx]
logit_dev     logits − dev_head(context_emb)           = W·(β·r), bias 상쇄
```
단 **유효성만은 복원 불가능**하다 — 검색이 못 채운 슬롯은 `topk_idx=0`이라
외부에서는 "0번 이웃"과 구분되지 않는다. 그래서 forward가
`out["neighbor_mask"]` 하나만 추가로 내보낸다.

⚠ 이 파일은 판정을 하지 않는다
```
❌  if ambiguity_ratio > 1.15: ambiguous = True
✅  {"ambiguity_ratio": 1.02}          ← 값만. 판정은 분석 스크립트에서
```
그리고 **자르지 않는다** — 순위 전체를 반환한다. top-N 절단은 표시 계층의
일이다. 계산 단계에서 자르면 분석 스크립트가 전체 분포를 못 본다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from libs.tabera import PROTO_DEV_FUSION_MODES

__all__ = [
    "retrieved_neighbors",
    "local_label_evidence",
    "prototype_deviation",
    "group_relative_feature_stats",
    "feature_gaps",
    "prototype_conditioning_overlap",
    # ⚠ 아래 둘을 여기 안 넣어서 reproduce.py 에서 AttributeError 가 났다.
    #   함수는 파일에 있는데 __all__ 에 빠뜨린 경우다. 새 함수를 추가할 때
    #   여기도 같이 갱신할 것 — smoke_test.py 가 아래 검사로 잡는다.
    "prototype_class_alignment",
    "context_space_diversity",
]


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

def _entropy(counts: Sequence[int], total: int) -> float:
    if total <= 0:
        return 0.0
    acc = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            acc -= p * float(np.log(p))
    return acc


def _cols(model, n_features: int) -> List[str]:
    return list(getattr(model, "column_names", None)
                or [f"f{i}" for i in range(n_features)])


def _cat_num_idx(model, n_features: int):
    cat = list(getattr(model.embedder, "cat_col_idx", []) or [])
    num = list(getattr(model.embedder, "num_col_idx", None)
               or [i for i in range(n_features) if i not in cat])
    return cat, num


# ─────────────────────────────────────────────────────────────
# ① 검색된 이웃 복원
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def retrieved_neighbors(
    model,
    out: Dict,
    feature_store=None,
) -> Optional[List[List[dict]]]:
    """
    forward output에서 이웃 목록을 복원한다.

    반환: [B][유효 이웃 수] 의 dict 리스트. 유사도 내림차순(= topk 순서).
      {rank, memory_idx, sample_id, similarity, label, features}

    ⚠ `retrieve()`가 유사도 값을 버리므로 여기서 다시 계산한다. 검색에
      쓰인 것과 **같은 식**이다: retrieve()도 `q_norm @ self._keys_norm`를
      쓰고, 반환하는 `nk`는 `keys_full[idx]`이므로 `memory.keys[topk_idx]`와
      같은 텐서다.

    ⚠ 못 채운 슬롯은 아예 반환하지 않는다. "similarity 0.000인 이웃"으로
      보이면 없는 사례를 있다고 설명하게 된다.

    ⚠ features는 저장 당시 값 그대로다(numeric은 [0,1] quantile). 사람이
      읽는 단위 역변환은 표시 계층의 몫 — 모델도 이 함수도 dataset의
      quantile_transformer를 모른다.

    feature_store=None이면 model.feature_store를 쓴다. 없으면 features는
    None으로 채운다(검색 자체는 그대로 복원됨).
    """
    topk = out.get("topk_idx")
    qr   = out.get("query_retr")
    if topk is None or qr is None:
        return None

    mask = out.get("neighbor_mask")
    fs   = feature_store if feature_store is not None else getattr(model, "feature_store", None)

    topk_c = topk.detach().cpu()
    qn = F.normalize(qr.detach(), dim=-1).unsqueeze(1)                  # (B,1,D)
    kn = F.normalize(model.memory.keys[topk].detach(), dim=-1)          # (B,k,D)
    sim = (qn * kn).sum(-1).cpu().numpy()                               # (B,k)

    lab = model.memory.labels[topk].detach().cpu().numpy()
    sid = model.memory.sample_ids[topk].detach().cpu().numpy()
    msk = mask.detach().cpu().numpy() if mask is not None else None
    feats = fs.retrieve(topk) if fs is not None else None               # [B][k] dict

    B, k = topk_c.shape
    result: List[List[dict]] = []
    for b in range(B):
        rows: List[dict] = []
        for j in range(k):
            if msk is not None and not bool(msk[b, j]):
                continue
            rows.append({
                "rank":       len(rows),
                "memory_idx": int(topk_c[b, j]),
                "sample_id":  int(sid[b, j]),
                "similarity": float(sim[b, j]),
                "label":      float(lab[b, j]),
                "features":   (feats[b][j] if feats is not None else None),
            })
        result.append(rows)
    return result


# ─────────────────────────────────────────────────────────────
# ② 지역 라벨 증거 (= ambiguity evidence)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def local_label_evidence(model, out: Dict) -> Optional[List[dict]]:
    """
    이웃의 라벨 구성을 **배정된 prototype 전체 분포와 함께** 반환한다.

    ⚠ 이웃 다수결을 예측 근거로 쓰면 안 된다. 같은 prototype 안에서는
      raw feature로도 다수결을 못 넘는다는 것이 측정으로 확인돼 있다 —
      그러면 "8개 중 6개가 A"는 그룹 분포의 표본 노이즈다.

    ⚠ 그럼에도 의미가 있는 이유는 따로 있다: prototype purity를 통제해도
      neighbor entropy가 오분류를 유의하게 예측한다. 즉 이웃은 **답**이
      아니라 **이 지역이 얼마나 섞여 있는가**를 말한다.

    → 그래서 지역 분포를 그룹 분포 없이는 절대 내보내지 않는다. 둘을 같이
      보지 않으면 "4/6"이 높은지 낮은지조차 알 수 없다(그룹이 82%면
      4/6=67%는 오히려 낮다).

    ⚠ 판정하지 않는다. `ambiguity_ratio` 값만 반환하고, "모호한 지역인가"의
      기준선은 분석 스크립트가 경험적 분포에서 정한다.

    ⚠ scope: 이 검색은 NN(q, G_p)이지 NN(q, D)가 아니다. 호출부가 문구를
      틀리게 쓰지 못하도록 필드로 남긴다.
    """
    topk = out.get("topk_idx")
    if topk is None:
        return None
    ha = out.get("hard_group")
    if ha is None:
        ha = out.get("centroid_id")
    if ha is None:
        return None

    tasktype = getattr(model, "tasktype", None)
    n_mem    = int(model.memory.filled.item())
    mem_lab  = model.memory.labels[:n_mem].detach().cpu().numpy()
    nb_lab   = model.memory.labels[topk].detach().cpu().numpy()
    mask     = out.get("neighbor_mask")
    msk      = mask.detach().cpu().numpy() if mask is not None else None
    ha_np    = ha.detach().cpu().numpy()
    sg       = getattr(model.prototype_layer, "sample_groups", None)

    grp_cache: Dict[int, np.ndarray] = {}
    result: List[dict] = []
    for b in range(nb_lab.shape[0]):
        sel = [j for j in range(nb_lab.shape[1])
               if msk is None or bool(msk[b, j])]
        labs = nb_lab[b, sel]
        p = int(ha_np[b])
        if p not in grp_cache:
            ids = (sg[p] if (sg is not None and p < len(sg) and sg[p] is not None) else [])
            ids = [i for i in ids if 0 <= i < n_mem]
            grp_cache[p] = mem_lab[ids] if ids else mem_lab[:0]
        glab = grp_cache[p]

        d: Dict[str, object] = {
            "scope":       "prototype_conditioned",   # NN(q, G_p)
            "prototype":   p,
            "n_neighbors": int(len(labs)),
            "group_size":  int(glab.shape[0]),
        }
        if tasktype == "regression":
            # 회귀는 "라벨 구성"이 없다 — 지역 산포를 그룹 산포와 비교한다
            # (같은 역할: 이 지역이 얼마나 흔들리는가).
            d["local_mean"] = float(labs.mean()) if len(labs) else float("nan")
            d["local_std"]  = float(labs.std())  if len(labs) else float("nan")
            d["group_mean"] = float(glab.mean()) if glab.shape[0] else float("nan")
            d["group_std"]  = float(glab.std())  if glab.shape[0] else float("nan")
            d["dispersion_ratio"] = (
                float(d["local_std"] / d["group_std"])
                if (d["group_std"] == d["group_std"] and d["group_std"] > 1e-9)
                else float("nan"))
        else:
            lc: Dict[int, int] = {}
            for v in labs:
                kk = int(round(float(v)))
                lc[kk] = lc.get(kk, 0) + 1
            gc: Dict[int, int] = {}
            for v in glab:
                kk = int(round(float(v)))
                gc[kk] = gc.get(kk, 0) + 1
            d["label_counts"]       = lc
            d["group_label_counts"] = gc
            # ⚠ 키 이름에 **출처**를 박는다. 맨 `entropy`는 쓰지 않는다 —
            #   npz export의 `entropy`가 evidence_w(attention weight) 분포의
            #   entropy이고 proto_dev에서는 상수 log(k)인데, 이름만 보고
            #   "이웃 불확실성"으로 읽히는 문제가 실제로 있었다. 같은 이름을
            #   여기서 다른 뜻으로 또 쓰면 그 혼동이 한 겹 더 쌓인다.
            #   이 값은 검색된 이웃의 **라벨 분포** entropy = H(Y_N(x))다.
            d["label_entropy"]       = _entropy(lc.values(), len(labs))
            d["group_label_entropy"] = _entropy(gc.values(), int(glab.shape[0]))
            d["ambiguity_ratio"] = (
                float(d["label_entropy"] / d["group_label_entropy"])
                if d["group_label_entropy"] > 1e-9 else float("nan"))
        result.append(d)
    return result


# ─────────────────────────────────────────────────────────────
# ③ prototype-relative deviation (정확한 가법 분해)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def prototype_deviation(model, out: Dict) -> Optional[List[dict]]:
    """
    h = c + d 에서 편차 항 d 의 크기·방향·logit 기여.

    ⚠ 이 분해는 **근사가 아니라 항등식**이다. dev_head가 단일
      nn.Linear(embed_dim, n_output)이므로
          logits = W·(c + d) + b = (W·c + b) + W·d
      이고 두 항의 합이 항상 logits와 정확히 같다. bias는 앞항에
      흡수되므로 W·d에는 섞이지 않는다. SHAP/IG처럼 baseline을 임의로
      골라 근사하는 것과 성격이 다르다 — 구조가 이미 가법적이라 나눠 쓸
      뿐이다.

    ⚠ d 는 forward와 **글자 그대로 같은 식**으로 재구성한다. 별도 근사식을
      쓰면 설명과 실제가 어긋난다.
          proto_dev      d = σ(β_raw) · normalize(q − c)
          proto_dev_vec  d = σ(β_raw) ⊙ normalize(q − c)      (β는 (D,))
          proto_dev_agg  d = σ(β_raw)·normalize(q−c) + σ(γ_raw)·normalize(a−c)
      `context_emb`는 RVQ 적용 후 값이므로 dev_head에 실제로 들어간 c와
      같다(확인함).

    ⚠ `dim_contrib`는 **embedding 차원**이지 feature가 아니다. embedder가
      PLE/PLR + MLP + LayerNorm이라 embedding 차원을 입력 feature로 되돌리는
      경로가 없고, gradient로 잇는 것도 안 된다(categorical에서 그래프가
      끊김). 여기서 말할 수 있는 것은 "편차가 몇 개 차원에 몰려 있는가"라는
      집중도이지 feature 이름이 아니다.

    ⚠ 자르지 않는다. 전체 D개 차원의 기여를 그대로 반환한다 — 상위 N개
      절단은 표시 계층의 일이다.
    """
    fm = getattr(model, "fusion_mode", None)
    dev_head = getattr(model, "dev_head", None)
    if fm not in PROTO_DEV_FUSION_MODES or not isinstance(dev_head, torch.nn.Linear):
        return None
    q = out.get("query_emb")
    c = out.get("context_emb")
    lg = out.get("logits")
    if q is None or c is None or lg is None:
        return None

    q = q.detach(); c = c.detach(); lg = lg.detach()
    beta = torch.sigmoid(model.dev_beta_raw.detach())
    dev_q = F.normalize(q - c, dim=-1)
    if fm == "proto_dev_agg":
        a = out.get("agg_emb").detach()
        gamma = torch.sigmoid(model.dev_gamma_raw.detach())
        d = beta * dev_q + gamma * F.normalize(a - c, dim=-1)
    else:
        d = beta * dev_q          # proto_dev(스칼라) / proto_dev_vec((D,) 브로드캐스트)

    W        = dev_head.weight.detach()          # (O, D)
    lg_proto = dev_head(c)                       # (B, O) = W·c + b
    lg_dev   = lg - lg_proto                     # (B, O) = W·d

    if lg.shape[-1] > 1:
        pred_m  = lg.argmax(dim=-1)
        proto_m = lg_proto.argmax(dim=-1)
        changed = pred_m != proto_m
    else:
        # ⚠ 이진/회귀는 n_output=1이라 argmax가 항상 0이다. 이진은 부호가
        #   클래스를 정하므로 부호 변화로 판정한다.
        pred_m  = torch.zeros(lg.shape[0], dtype=torch.long, device=lg.device)
        proto_m = pred_m
        changed = (lg.squeeze(-1) > 0) != (lg_proto.squeeze(-1) > 0)

    # ── 확률 이동 ────────────────────────────────────────────────
    # ⚠ dev_share(= |ld| / (|lp| + |ld|))는 **로짓 크기 비율**이라 크기를
    #   과장한다. credit-g 실측: dev_share가 5.6~19.3%로 읽히는데 실제
    #   확신도는 0.2~1.2%p만 움직였다. 로짓이 ±0.6 구간이라 sigmoid가
    #   거의 선형이기 때문이다. 사람이 읽는 단위는 확률이므로 그것을 낸다.
    #
    # ⚠ prototype만의 확률은 "같은 클래스"로 재는 것이 맞다 — 최종 예측
    #   클래스가 prototype 단계에서 몇 %였는지를 보여야 이동이 읽힌다.
    tasktype = getattr(model, "tasktype", None)
    if tasktype == "regression":
        prob_proto = prob_final = None
        proto_pred = None
    elif lg.shape[-1] == 1:
        _pf = torch.sigmoid(lg.squeeze(-1))
        _pp = torch.sigmoid(lg_proto.squeeze(-1))
        _cls = (_pf > 0.5)
        prob_final = torch.where(_cls, _pf, 1 - _pf)
        prob_proto = torch.where(_cls, _pp, 1 - _pp)   # 최종 예측 클래스 기준
        proto_pred = (_pp > 0.5).long()
    else:
        _ar0 = torch.arange(lg.shape[0], device=lg.device)
        prob_final = torch.softmax(lg, -1)[_ar0, pred_m]
        prob_proto = torch.softmax(lg_proto, -1)[_ar0, pred_m]
        proto_pred = proto_m

    contrib = W[pred_m] * d                      # (B, D), 합 = lg_dev[:, m]
    d_np       = d.norm(dim=-1).cpu().numpy()
    lgp_np     = lg_proto.cpu().numpy()
    lgd_np     = lg_dev.cpu().numpy()
    m_np       = pred_m.cpu().numpy()
    chg_np     = changed.cpu().numpy()
    pp_np      = prob_proto.cpu().numpy() if prob_proto is not None else None
    pf_np      = prob_final.cpu().numpy() if prob_final is not None else None
    ppred_np   = proto_pred.cpu().numpy() if proto_pred is not None else None
    contrib_np = contrib.cpu().numpy()

    result = []
    for b in range(lg.shape[0]):
        m  = int(m_np[b])
        lp = float(lgp_np[b, m])
        ld = float(lgd_np[b, m])
        result.append({
            "dev_norm":       float(d_np[b]),      # ‖d‖. proto_dev면 r이 단위라 = β
            "logit_proto":    lp,                  # W·c + b
            "logit_dev":      ld,                  # W·d
            # 예측 채널에서 편차가 차지하는 비중. logit은 부호가 있어 단순
            # 비율이 발산할 수 있어 |lp|+|ld|로 나눈다 — "크기 대비 비중"
            # 으로만 읽을 것.
            "dev_share":      float(abs(ld) / max(abs(lp) + abs(ld), 1e-12)),
            "argmax_changed": bool(chg_np[b]),
            "pred_channel":   m,
            # 최종 예측 클래스가 prototype 단계에서 가졌던 확률 → 최종 확률.
            # 회귀는 None.
            "prob_proto":     (float(pp_np[b]) if pp_np is not None else None),
            "prob_final":     (float(pf_np[b]) if pf_np is not None else None),
            # prototype만으로 예측했다면 어느 클래스였는가(코드).
            "proto_pred":     (int(ppred_np[b]) if ppred_np is not None else None),
            "dim_contrib":    contrib_np[b].tolist(),   # 전체 D개 (절단 없음)
            "n_dims":         int(contrib_np.shape[1]),
        })
    return result


# ─────────────────────────────────────────────────────────────
# ④ 그룹 대비 feature 통계 (feature 공간, 기술 통계)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def group_relative_feature_stats(
    model,
    out: Dict,
    X: torch.Tensor,
    feature_store=None,
) -> Optional[List[dict]]:
    """
    "같은 그룹의 전형적 샘플 대비 이 샘플은 무엇이 다른가"를 raw feature로
    직접 비교한다.

    ⚠ label_all_groups()와 축이 다르다.
    ```
    label_all_groups              그룹 A  vs  다른 그룹들   "이 그룹만 유별난 feature"
    group_relative_feature_stats  샘플 x  vs  자기 그룹 A   "이 샘플만 유별난 feature"
    ```

    ⚠ **attribution이 아니다.** prototype_deviation(embedding 공간의 정확한
      logit 분해)과 나란히 놓되 인과로 연결하면 안 된다 — embedder가
      비선형이라 embedding 차원 ↔ feature 대응이 없고, gradient로 잇는
      것도 막혀 있다. "이 feature 때문에 예측이 이렇게 나왔다"는 문장은
      이 값으로 만들 수 없다.

    ⚠ 그룹 크기로 거르지 않는다. 표본이 작으면 mean/std가 못 믿을 값이지만,
      그 판단 기준을 여기서 임의로 정하면(예: n<5면 버림) 근거 없는 detector가
      된다. `group_size`와 `group_std`를 반드시 같이 반환하니 소비자가
      판단한다. 다만 `std < 1e-6`(그룹 안에서 상수)일 때 z를 건너뛰는 것은
      판정이 아니라 0으로 나누는 것을 막는 수치 보호다.

    ⚠ 자르지 않는다. 전체 feature를 |z| / rarity 내림차순으로 반환한다.

    반환: [B] 의 dict. {"numeric": [...], "categorical": [...], "group_size": n}
      numeric     : z = (x − 그룹평균) / 그룹표준편차
      categorical : rarity = 1 − (그룹 안에서 이 값의 비율)
                    group_mode / group_mode_freq 동봉 — 최빈값과 같은지는
                    소비자가 판단한다(여기서 거르지 않는다)
    """
    fs = feature_store if feature_store is not None else getattr(model, "feature_store", None)
    sg = getattr(model.prototype_layer, "sample_groups", None)
    if fs is None or not sg:
        return None
    ha = out.get("hard_group")
    if ha is None:
        ha = out.get("centroid_id")
    if ha is None:
        return None

    n_fill = fs._filled
    store  = fs._store[:n_fill].cpu().numpy()
    Xnp    = X.detach().cpu().numpy()
    n_feat = Xnp.shape[1]
    cat, num = _cat_num_idx(model, n_feat)
    cols     = _cols(model, n_feat)
    ha_np    = ha.detach().cpu().numpy()

    cache: Dict[int, np.ndarray] = {}
    result = []
    for b in range(Xnp.shape[0]):
        p = int(ha_np[b])
        if p not in cache:
            ids = sg[p] if (p < len(sg) and sg[p] is not None) else []
            ids = [i for i in ids if 0 <= i < n_fill]
            cache[p] = store[ids] if ids else store[:0]
        rows = cache[p]
        n_g  = int(rows.shape[0])

        num_out, cat_out = [], []
        if n_g > 0:
            for fi in num:
                if fi >= n_feat or fi >= rows.shape[1]:
                    continue
                col = rows[:, fi].astype(np.float64)
                mu, sd = float(col.mean()), float(col.std())
                val = float(Xnp[b, fi])
                if sd < 1e-6:      # 수치 보호(판정 아님)
                    continue
                # ⚠ 그룹 내 백분위. **단조 변환에 불변**이라 quantile 공간에서
                #   재도 실공간 백분위와 같다. z는 quantile 공간에서 계산되는데
                #   화면의 값/대표값은 실단위로 역변환되므로 축이 어긋나고,
                #   그 결과 "1 (그룹 대표값 1, z=-0.79)"처럼 같아 보이는데
                #   z만 붙는 상태가 된다(credit-g 20건에서 반복 확인).
                #   백분위는 그 불일치가 원천적으로 없다.
                # ⚠ 동점 처리는 midrank(below + equal/2) — 이산형 feature는
                #   같은 값이 많아 "미만 비율"만 쓰면 위치가 왜곡된다.
                below = float((col < val).mean())
                equal = float((col == val).mean())
                num_out.append({
                    "feature_idx":  fi,
                    "feature_name": cols[fi] if fi < len(cols) else f"f{fi}",
                    "kind":         "numeric",
                    "value":        val,
                    "group_mean":   mu,
                    "group_std":    sd,
                    # z는 화면에서 빠지지만 **분석·정렬용으로 유지**한다
                    # (feature ranking에는 z가 더 유용할 수 있다).
                    "z":            (val - mu) / sd,
                    "group_pct":       below + equal / 2.0,   # midrank, 0~1
                    "group_pct_below": below,
                    "group_pct_equal": equal,
                })
            for fi in cat:
                if fi >= n_feat or fi >= rows.shape[1]:
                    continue
                code = int(round(float(Xnp[b, fi])))
                col  = np.rint(rows[:, fi]).astype(np.int64)
                freq = float((col == code).sum()) / n_g
                vals, cnts = np.unique(col, return_counts=True)
                cat_out.append({
                    "feature_idx":     fi,
                    "feature_name":    cols[fi] if fi < len(cols) else f"f{fi}",
                    "kind":            "categorical",
                    "value":           code,
                    "group_freq":      freq,
                    "group_mode":      int(vals[int(cnts.argmax())]),
                    "group_mode_freq": float(int(cnts.max())) / n_g,
                    "rarity":          1.0 - freq,
                })
        num_out.sort(key=lambda d: abs(d["z"]), reverse=True)
        cat_out.sort(key=lambda d: d["rarity"], reverse=True)
        result.append({"numeric": num_out, "categorical": cat_out, "group_size": n_g})
    return result


# ─────────────────────────────────────────────────────────────
# ⑤ query ↔ 이웃 feature 차이 (전체)
# ─────────────────────────────────────────────────────────────

def feature_gaps(query: Dict[str, float],
                 neighbour: Dict[str, float],
                 cat_names: set) -> List[dict]:
    """
    query와 한 이웃 사이의 feature별 차이를 **전부** 반환한다.

    ⚠ 이전 구현(`_select_query_similar_features`)은 `gap > 0.15`인 feature를
      후보에서 아예 제외했다. 즉 결과를 보여주기 전에 정보를 삭제했고,
      "왜 비슷한지"만 보이고 "어디가 다른지"는 숨는 확증편향 표시가 됐다.
      임계값으로 무엇을 볼지 결정하는 것은 detector와 같은 문제다 —
      여기서는 자르지 않고, 정렬과 절단은 표시 계층이 한다.

    gap 정의는 Gower 방식 그대로다. categorical은 LabelEncoder 정수 코드에
    순서가 없으므로 뺄셈이 "얼마나 다른가"가 될 수 없어 0/1로 둔다.
    (`delta`는 표시 계층이 query 값을 복원(qv = neighbour − delta)하는 데만
    쓴다 — categorical의 delta 자체를 크기로 읽으면 안 된다.)

    반환: [{name, kind, query_value, neighbor_value, delta, gap}, ...]
          입력 feature 순서 그대로. 정렬은 호출부에서.
    """
    rows = []
    for k, v in neighbour.items():
        if k not in query:
            continue
        is_cat = k in cat_names
        if is_cat:
            gap = 0.0 if query[k] == v else 1.0
        else:
            gap = abs(v - query[k])
        rows.append({
            "name":           k,
            "kind":           "categorical" if is_cat else "numeric",
            "query_value":    query[k],
            "neighbor_value": v,
            "delta":          v - query[k],
            "gap":            gap,
        })
    return rows


# ─────────────────────────────────────────────────────────────
# ⑥ Q1 — prototype conditioning이 실제로 이웃 집합을 바꾸는가
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def prototype_conditioning_overlap(
    model,
    out: Dict,
    sample_ids: Optional[torch.Tensor] = None,
    n_permutations: int = 100,
    rng_seed: int = 0,
) -> Optional[Dict]:
    """
    같은 모델에서 NN(q, G_p) 와 NN(q, D) 를 비교한다.

    검증 대상 문장
    ──────────────
    문서가 retrieval을 "prototype-conditioned"라고 부른다. 그 수식어가
    실제 내용을 갖는지 확인한다.
        겹침이 낮다  → 조건부라는 서술이 필요하고 정확하다
        겹침이 높다  → 그룹 제약이 사실상 무효. 표현을 고치고, 대신
                       "그룹 제약이 검색 결과를 거의 안 바꾼다"를 별도
                       관찰로 보고해야 한다
    어느 쪽이 나올지 모르는 상태로 잰다.

    ⚠ 모델을 변형하지 않는다. `model.global_retrieve = True` 같은 속성
      mutation은 진단 함수가 이후 평가 상태를 조용히 바꾸는 부작용을
      만든다. 여기서는 MemoryBank.retrieve()에 이미 있는 오버라이드
      (`hard_assignment=None` → 전체 검색)를 그대로 쓴다 — 인자를 새로
      추가하면 같은 뜻을 두 가지로 표현하게 되어 모순된 호출이 가능해진다.

    ⚠ null baseline이 없으면 숫자를 읽을 수 없다. 그룹 크기 60에서 k=8을
      뽑으면 우연만으로도 Jaccard가 0.07 근처다. 두 가지를 같이 낸다.
        analytic   전역 top-k 중 m개가 이 그룹 안에 있을 때
                   그룹에서 k개를 무작위로 뽑으면 E|A∩B| = k·m/G
        permutation 실제로 n_permutations회 뽑아 분포를 본다
      `n_global_in_group`(=m) 자체가 가장 읽기 쉬운 값이다 — 전역 최근접이
      애초에 전부 그룹 안에 있으면 조건부는 아무것도 바꿀 수 없다.

    ⚠ 자르지 않고 판정하지 않는다. 샘플별 값을 전부 돌려주고, 요약과
      해석은 분석 스크립트가 한다. group_size를 반드시 같이 반환한다 —
      |G|=20과 |G|=500은 같은 Jaccard라도 뜻이 다르다.

    반환: {"per_sample": [...], "meta": {...}} 또는 None
      per_sample: sample_idx, group_id, group_size, k, n_local, n_global,
                  n_intersect, jaccard, top1_match, rank_corr,
                  n_global_in_group, null_jaccard_analytic,
                  null_jaccard_perm_mean, null_jaccard_perm_std,
                  fallback(bool), local_ids, global_ids
    """
    if getattr(model, "global_retrieve", False):
        # 이 모델은 이미 전역 검색이라 비교 대상이 없다. 조용히 1.0을
        # 내면 "조건부가 무효"라는 결론으로 오독된다.
        raise ValueError(
            "model.global_retrieve=True인 모델에서는 이 비교가 성립하지 않습니다 "
            "— out['topk_idx']가 이미 전역 검색 결과입니다.")

    topk = out.get("topk_idx")
    qr   = out.get("query_retr")
    if topk is None or qr is None:
        return None
    ha = out.get("hard_group")
    if ha is None:
        ha = out.get("centroid_id")
    if ha is None:
        return None

    n_mem = int(model.memory.filled.item())
    k     = int(topk.shape[1])
    if n_mem < k:
        # 메모리가 k개도 못 채웠으면 group-constrained 경로도 전체 검색
        # fallback을 탄다 — 겹침이 1.0으로 나오지만 그건 조건부가 무효라는
        # 뜻이 아니라 조건부가 애초에 적용되지 않았다는 뜻이다.
        return {"per_sample": [], "meta": {
            "skipped": "memory.filled < k — 조건부 검색 자체가 적용되지 않음",
            "memory_filled": n_mem, "k": k}}

    excl = sample_ids if getattr(model, "exclude_self_retrieval", False) else None
    # 전역 검색: hard_assignment=None 이 곧 오버라이드다(모델 변경 없음).
    _, _, g_idx = model.memory.retrieve(qr, k, hard_assignment=None, exclude_ids=excl)

    # 같은 축(정규화 cosine)에서 local/global 유사도를 잰다 — distance_gap용.
    _keys_n = F.normalize(model.memory.keys[:n_mem].detach(), dim=-1)
    _qn     = F.normalize(qr.detach(), dim=-1)
    _all    = _qn @ _keys_n.T                                   # (B, n_mem)
    sim_l   = _all.gather(1, topk.clamp(0, n_mem - 1)).cpu().numpy()
    sim_g   = _all.gather(1, g_idx.clamp(0, n_mem - 1)).cpu().numpy()

    mask  = out.get("neighbor_mask")
    msk   = mask.detach().cpu().numpy() if mask is not None else None
    l_np  = topk.detach().cpu().numpy()
    g_np  = g_idx.detach().cpu().numpy()
    ha_np = ha.detach().cpu().numpy()
    sg    = getattr(model.prototype_layer, "sample_groups", None)
    rng   = np.random.default_rng(rng_seed)

    grp_cache: Dict[int, np.ndarray] = {}
    per_sample = []
    for b in range(l_np.shape[0]):
        sel = [j for j in range(k) if msk is None or bool(msk[b, j])]
        A = l_np[b, sel]                      # 조건부 top-k (유사도 내림차순)
        B = g_np[b]                           # 전역 top-k
        p = int(ha_np[b])
        if p not in grp_cache:
            ids = (sg[p] if (sg is not None and p < len(sg) and sg[p] is not None) else [])
            grp_cache[p] = np.array([i for i in ids if 0 <= i < n_mem], dtype=np.int64)
        G = grp_cache[p]
        gsize = int(G.shape[0])

        sim_l_b = sim_l[b, sel] if len(sel) else np.array([])
        setA, setB = set(A.tolist()), set(B.tolist())
        inter = setA & setB
        union = setA | setB
        jac = (len(inter) / len(union)) if union else float("nan")

        # 순위 상관: 교집합 원소의 A 내 순위 vs B 내 순위
        if len(inter) >= 2:
            ra = {v: i for i, v in enumerate(A.tolist())}
            rb = {v: i for i, v in enumerate(B.tolist())}
            xs = np.array([ra[v] for v in inter], dtype=float)
            ys = np.array([rb[v] for v in inter], dtype=float)
            from scipy.stats import spearmanr          # noqa: F401 (선택 의존)
            rc = float(spearmanr(xs, ys).statistic)
        else:
            rc = float("nan")

        # null: 전역 top-k 중 m개가 이 그룹 안에 있을 때, 그룹에서 무작위
        # k개를 뽑으면 얼마나 겹치는가
        m = int(len(setB & set(G.tolist()))) if gsize else 0
        if gsize >= 1:
            exp_i = k * m / gsize
            null_a = exp_i / max(2 * k - exp_i, 1e-12)
            draws = []
            take = min(k, gsize)
            for _ in range(n_permutations):
                R = set(rng.choice(G, size=take, replace=False).tolist())
                iu = len(R & setB)
                draws.append(iu / max(len(R | setB), 1))
            null_p_mean, null_p_std = float(np.mean(draws)), float(np.std(draws))
        else:
            null_a = null_p_mean = null_p_std = float("nan")

        # ⚠ Jaccard만으로는 부족하다. 겹침이 같아도 "그룹 밖 후보가 거의
        #   비슷했다"와 "그룹 밖에 훨씬 가까운 게 있었다"는 전혀 다른
        #   상황이고, 후자만이 제약이 실제로 비용을 치렀다는 뜻이다.
        #   같은 cosine 축에서 local과 global의 1번째/k번째를 직접 뺀다.
        _sl = sim_l_b                        # 이 샘플의 local 이웃 cosine
        _sg_ = sim_g[b]                      # 전역 top-k cosine (내림차순)
        gap_top1 = float(_sg_[0] - _sl.max()) if len(_sl) else float("nan")
        gap_topk = float(_sg_[-1] - _sl.min()) if len(_sl) else float("nan")

        per_sample.append({
            "sample_idx":  b,
            "group_id":    p,
            "group_size":  gsize,
            "k":           k,
            "n_local":     int(len(setA)),
            "n_global":    int(len(setB)),
            "n_intersect": int(len(inter)),
            "jaccard":     jac,
            "top1_match":  (bool(A[0] == B[0]) if len(A) and len(B) else None),
            "rank_corr":   rc,
            "n_global_in_group":      m,
            "null_jaccard_analytic":  float(null_a),
            "null_jaccard_perm_mean": null_p_mean,
            "null_jaccard_perm_std":  null_p_std,
            # 그룹 제약이 치른 유사도 비용(>=0). 0에 가까우면 "그룹 밖에
            # 더 가까운 후보가 없었다" = 제약이 사실상 공짜였다는 뜻.
            "distance_gap_top1": gap_top1,
            "distance_gap_topk": gap_topk,
            # ⚠ fallback 샘플은 그룹 확장 검색을 탔으므로 "그룹 제약"이
            #   이미 느슨하다. 제외 여부는 분석 단계에서 정하되, 제외
            #   비율 자체가 결과다 — 30%가 fallback이면 "그룹 제약"이라는
            #   서술이 이미 절반만 사실이다.
            "fallback":    bool(gsize < k or (msk is not None and not msk[b].all())),
            "local_ids":   A.tolist(),
            "global_ids":  B.tolist(),
        })

    return {"per_sample": per_sample, "meta": {
        "memory_filled": n_mem, "k": k,
        "exclude_self_retrieval": bool(getattr(model, "exclude_self_retrieval", False)),
        "n_permutations": n_permutations,
    }}


# ─────────────────────────────────────────────────────────────
# ⑦ prototype이 무엇을 표현하는가 — density mode vs class anchor
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def prototype_class_alignment(model) -> Optional[Dict]:
    """prototype이 클래스를 나눠 맡고 있는지 측정한다.

    왜 필요한가
    ───────────
    `P >= C` 는 **필요조건이지 충분조건이 아니다.** prototype 항만으로 낼 수
    있는 서로 다른 argmax가 최대 P개이므로 P < C면 편차가 떠맡아야 하지만,
    P = C 여도 C개 centroid가 전부 한 클래스 방향으로 갈 수 있다.
    P를 늘렸을 때 원하는 것은
        ✅ predictive anchor  — 클래스를 나눠 맡되 그룹 안에 구조가 남음
        ❌ class memory       — centroid가 클래스 라벨로 붕괴
    이고, 둘을 가르는 지표가 alignment 와 H(Y_G) 다.

    반환
    ────
        alignment        mean_p max_y P(y|p).  1/C ≈ 무작위, 1.0 = 클래스 순수
        alignment_std    prototype 간 편차
        group_entropy    mean_p H(Y|p).  0에 붙으면 class memory
        n_prototypes     설정값 P
        n_effective      비어있지 않은 prototype 수
        n_eff_entropy    exp(H(assignment 분포)) — 크기 불균형까지 반영한 유효 개수
                         (P개 중 12개만 산다는 것과, 살아있어도 하나가 61%를
                          먹는다는 것은 다른 문제다)
        dead_ratio       비어있는 비율
        per_prototype    [{p, size, top_class, top_prop, entropy}, ...]

    ⚠ 이 함수는 train 분포(sample_groups + memory.labels)만 본다. test 라벨을
      쓰지 않으므로 학습 중에도 안전하게 로깅할 수 있다.
    ⚠ regression에서는 클래스가 없으므로 None을 반환한다.
    """
    if getattr(model, "tasktype", None) == "regression":
        return None
    sg = getattr(model.prototype_layer, "sample_groups", None)
    if not sg:
        return None
    n_mem = int(model.memory.filled.item())
    if n_mem == 0:
        return None
    lab = model.memory.labels[:n_mem].detach().cpu().numpy().round().astype(int)
    C = int(lab.max()) + 1

    rows, sizes, aligns, ents = [], [], [], []
    for p, ids in enumerate(sg):
        ids = [i for i in (ids or []) if 0 <= i < n_mem]
        if not ids:
            rows.append({"p": p, "size": 0, "top_class": None,
                         "top_prop": None, "entropy": None})
            sizes.append(0)
            continue
        y = lab[ids]
        cnt = np.bincount(y, minlength=C)
        prop = cnt / cnt.sum()
        top = int(prop.argmax())
        nz = prop[prop > 0]
        H = float(-(nz * np.log(nz)).sum())
        rows.append({"p": p, "size": len(ids), "top_class": top,
                     "top_prop": float(prop[top]), "entropy": H})
        sizes.append(len(ids))
        aligns.append(float(prop[top]))
        ents.append(H)

    sizes_a = np.asarray(sizes, dtype=np.float64)
    tot = sizes_a.sum()
    # 크기 가중 유효 개수. 살아있는 개수만 세면 "하나가 61%를 먹는" 상황을
    # 놓친다(phoneme 실측: 65개 중 42개 생존, 그러나 한 그룹이 학습 데이터의 61%).
    p_assign = sizes_a[sizes_a > 0] / tot if tot > 0 else np.array([1.0])
    n_eff_H = float(np.exp(-(p_assign * np.log(p_assign)).sum()))

    return {
        "alignment":     float(np.mean(aligns)) if aligns else float("nan"),
        "alignment_std": float(np.std(aligns))  if aligns else float("nan"),
        "group_entropy": float(np.mean(ents))   if ents   else float("nan"),
        "chance_alignment": 1.0 / C,
        "n_prototypes":  len(sg),
        "n_effective":   int((sizes_a > 0).sum()),
        "n_eff_entropy": n_eff_H,
        "dead_ratio":    float((sizes_a == 0).mean()),
        "n_classes":     C,
        "per_prototype": rows,
    }


# ─────────────────────────────────────────────────────────────
# ⑧ context space가 무너졌는가 — prototype vocabulary collapse
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def context_space_diversity(model) -> Optional[Dict]:
    """prototype이 실제로 서로 다른 context를 제공하고 있는지 측정한다.

    왜 dead ratio 로는 부족한가
    ───────────────────────────
    ⚠ prototype이 **균등하게 쓰이는 것**은 목표가 아니다. TabERA의 centroid는
      uniform partition의 cluster center가 아니라 데이터 manifold의 mode를
      가리키는 latent context anchor다. 실제 분포가
          A 70% / B 20% / C 5% / D 5%
      라면 좋은 prototype 배치도 그만큼 불균등한 게 맞다. 균등화를 목표로
      두면(예: KL(usage‖Uniform)) 모델이 clustering 쪽으로 변질된다.

    ⚠ 진짜 문제는 **vocabulary collapse** 다. P=50을 뒀는데 실제 context가
      12개 방향밖에 못 만들면, `context_emb`는 사실상 "하나의 prototype +
      약간의 잡음"이 된다. LLM vocabulary가 50k인데 12 token만 쓰이는 것과
      같다. 이건 균등성 문제가 아니라 **표현 용량을 버리고 있는 것**이다.

    그래서 재는 것
    ──────────────
        usage_entropy_eff   exp(H(사용 분포)).  "몇 개가 실질적으로 쓰이는가"
        gini                사용 편중도 (0=균등, 1=독점).  참고용이지 목표 아님
        top1_share          최대 prototype이 먹는 비율
        context_eff_rank    **핵심.** 학습 샘플이 실제로 받는 context_emb 의
                            공분산 고유값 스펙트럼 entropy의 exp.
                            = "context 공간이 실제로 몇 차원을 쓰는가"
        context_eff_rank_uniform
                            사용 분포가 균등했다면 나왔을 eff rank.
                            둘의 비가 **편중 때문에 잃은 표현 차원**이다.
        centroid_cos_mean/min
                            살아있는 centroid 쌍의 cosine.
                            1에 가까우면 centroid들이 같은 방향으로 뭉친 것.

    ⚠ context_eff_rank 는 embed_dim 이 상한이다. P < embed_dim 이면 P 도 상한.
    """
    pl = getattr(model, "prototype_layer", None)
    sg = getattr(pl, "sample_groups", None) if pl is not None else None
    if pl is None or not sg:
        return None
    C_emb = pl.centroid_emb.detach()                       # (P, D)
    P, D = C_emb.shape
    n_mem = int(model.memory.filled.item())
    sizes = np.array([len([i for i in (g or []) if 0 <= i < n_mem]) for g in sg],
                     dtype=np.float64)
    tot = sizes.sum()
    if tot <= 0:
        return None
    w = sizes / tot                                        # 사용 분포 (P,)

    nz = w[w > 0]
    H = float(-(nz * np.log(nz)).sum())
    gini = float((np.abs(sizes[:, None] - sizes[None, :]).sum())
                 / (2 * len(sizes) * sizes.sum())) if tot > 0 else float("nan")

    def _eff_rank(weights):
        # 샘플이 받는 context_emb 의 가중 공분산. context_emb = c_p 이므로
        # Cov = Σ_p w_p (c_p - μ)(c_p - μ)^T,  μ = Σ_p w_p c_p
        wt = torch.as_tensor(weights, dtype=C_emb.dtype, device=C_emb.device)
        mu = (wt[:, None] * C_emb).sum(0, keepdim=True)
        Xc = (C_emb - mu) * wt[:, None].sqrt()
        ev = torch.linalg.svdvals(Xc) ** 2
        ev = ev[ev > 0]
        if ev.numel() == 0:
            return 0.0
        p = ev / ev.sum()
        return float(torch.exp(-(p * p.log()).sum()))

    alive = sizes > 0
    er_actual = _eff_rank(w)
    er_uniform = _eff_rank(alive / max(alive.sum(), 1))     # 살아있는 것만 균등

    Cn = F.normalize(C_emb[torch.as_tensor(alive)], dim=-1)
    G = (Cn @ Cn.T).cpu().numpy()
    iu = np.triu_indices(G.shape[0], k=1)
    pair = G[iu] if len(iu[0]) else np.array([np.nan])

    return {
        "n_prototypes":  int(P),
        "embed_dim":     int(D),
        "n_alive":       int(alive.sum()),
        "usage_entropy_eff": float(np.exp(H)),
        "gini":          gini,
        "top1_share":    float(sizes.max() / tot),
        "context_eff_rank":         er_actual,
        "context_eff_rank_uniform": er_uniform,
        # 편중 때문에 잃은 표현 차원의 비율. 1에 가까우면 편중이 표현을
        # 거의 안 깎았다는 뜻 — 불균등해도 collapse는 아니라는 신호다.
        "eff_rank_ratio": (er_actual / er_uniform) if er_uniform > 0 else float("nan"),
        "centroid_cos_mean": float(np.mean(pair)),
        "centroid_cos_max":  float(np.max(pair)),
    }