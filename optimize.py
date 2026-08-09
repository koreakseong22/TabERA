## Main file for optimizing TabERA for a specific dataset.
## Paper info: TabERA — Tabular Hierarchical Explainable Retrieval Architecture
## Based on: MultiTab (Kyungeun Lee, kyungeun.lee@lgresearch.ai)

import os, argparse

# ── CUDA_VISIBLE_DEVICES를 torch import 전 최우선 설정 ──────────
# MultiTab 원본과 동일하게 argparse 직후, torch import 전에 설정
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id",    type=int, default=0,      help="gpu index")
parser.add_argument("--openml_id", type=int, default=45068,  help="dataset index (See dataset_id.json for detailed information)")
parser.add_argument("--seed",      type=int, default=1,      help="seed for dataset split (cross-validation)")
parser.add_argument("--savepath",  type=str, default=".",    help="path to save the results")
parser.add_argument("--n_trials",  type=int, default=100,    help="Number of optimization trials")
parser.add_argument("--metric",    type=str, default="l2",
                    choices=["l2", "l1", "cosine", "mahalanobis", "wasserstein", "kl"],
                    help="Distance metric for TabR Retriever")
parser.add_argument("--evidence_metric", type=str, default="cosine",
                    choices=["euclidean", "cosine", "cosine_scaled"],
                    help=(
                        "AttentionAggregator(evidence_w, 설명②)의 유사도 공간. "
                        "Optuna 탐색 대상이 아니라 이 HPO 실행 전체에 고정 적용되는 "
                        "구조 선택(no_offset_correction/context_projection과 같은 성격). "
                        "[기본값 변경] euclidean → cosine(reproduce.py와 통일). euclidean은 "
                        "-‖q-k‖²(raw, 정규화 안 됨) — 4개 데이터셋 "
                        "x 5-seed 실측 결과 evidence_w가 n_eff≈1.0(사실상 1-NN)으로 "
                        "100%% 재현되게 붕괴함이 확인됨. cosine은 q,k를 CentroidLayer "
                        "라우팅과 동일하게 정규화 후 2·cos — 같은 실측에서 n_eff를 "
                        "7.6~8.5로 안정적으로 유지, 성능은 유지~일부 개선(credit-g "
                        "balanced accuracy 등). euclidean으로 HPO하려면 명시적으로 "
                        "'--evidence_metric euclidean'을 줄 것 — reproduce.py도 같은 이름의 플래그로 "
                        "이 study를 다시 불러옴. [주의] 기존 --metric 플래그(choices에 "
                        "cosine 포함)와 이름이 비슷해 보이지만 완전히 다른 것 — --metric은 "
                        "params_to_model_kwargs()가 그 값을 안 읽어서 TabERA까지 전달된 "
                        "적 없는 죽은 플래그(2026-07 확인). 원래 의도(TabR Retriever "
                        "거리 지표 — FAISS 근사검색용이었을 가능성)를 알 수 없어 재활용 "
                        "안 하고 이 플래그를 새로 만듦."
                    ))
parser.add_argument("--no_offset_correction", action="store_true",
                    help=(
                        "[ablation] AttentionAggregator의 value 구성에서 "
                        "TabR 스타일 T(query-neighbour) 오프셋 보정을 끄고 "
                        "value=label_emb만 사용. 저장 파일명이 구분되도록 "
                        "savepath 하위에 자동으로 표시됨. Optuna 탐색 대상이 "
                        "아니라 이 실행 전체에 고정 적용되는 구조 선택임 "
                        "(HPO 노이즈와 ablation 신호를 분리하기 위함)."
                    ))
# ── [2026-07, S-1A] retrieval 전용 표현 분기 ──────────────────────
# [주의] 이 인자들은 study_pkl_tag에 반영되어 **study 파일이 분리된다.**
# 반영을 빠뜨리면 reproduce.py가 baseline study를 읽어와, "retrieval 전용
# 표현인데 그것 없이 튜닝된 하이퍼파라미터"로 학습하게 된다 —
# --global_retrieve에서 실제로 겪은 사고이며 에러 없이 조용히 틀린다.
parser.add_argument("--retr_proj_mode", type=str, default="none",
                    choices=["none", "linear", "mlp"],
                    help=(
                        "retrieval 전용 projection. none=V1과 완전 동일. "
                        "linear=D→D, mlp=residual 2-layer. identity 초기화라 "
                        "학습 전에는 셋이 같은 값을 낸다."))
parser.add_argument("--detach_retr_grad", action="store_true",
                    help=(
                        "retrieval loss의 gradient를 shared encoder로 흘리지 않음. "
                        "[v2에서 제거됨] "
                        "없으므로 이 플래그 유무가 같은 결과여야 하고, 그 일치가 "
                        "sanity check가 된다."))
parser.add_argument("--global_retrieve", action="store_true",
                    help=(
                        "[진단용] retrieve()에서 centroid 그룹 제약을 끄고 "
                        "전체 memory bank에서 순수 전역 KNN 검색. "
                        "context_emb(설명①)는 그대로 유지되고 evidence_w/"
                        "agg_emb(설명②)만 영향받음. '그룹-제약 KNN이 정확도에 "
                        "요구하는 대가'를 격리해서 재기 위한 일회성 진단용 — "
                        "본 실험에는 쓰지 않음."
                    ))
parser.add_argument("--detach_context_grad", action="store_true",
                    help=(
                        "[진단용] context_emb는 head 입력으로 그대로 전달하되, "
                        "그쪽에서 오는 gradient만 centroid_emb로 안 흐르게 끊음 "
                        "(diversity_loss gradient는 그대로 흐름). task_loss와 "
                        "diversity_loss가 centroid_emb를 두고 충돌하는지 검증용. "
                        "reproduce.py 단발 재학습으로는 best_params가 이 설정을 "
                        "전제로 찾아진 게 아니라서, 이 플래그를 켠 채로 HPO를 "
                        "새로 돌려 loss_diversity/loss_commitment 최적값이 "
                        "바뀌는지 확인하기 위함."
                    ))
parser.add_argument("--context_projection", action="store_true",
                    help=(
                        "[구조 조정] context_emb를 head로 보내기 전 학습 가능한 "
                        "Linear를 하나 거치게 함. detach_context_grad(gradient "
                        "완전 차단)와 달리 task_loss의 gradient는 여전히 "
                        "centroid_emb까지 도달하되, 프로젝션 행렬이 예측 최적화의 "
                        "일부를 대신 떠맡아 centroid_emb 왜곡을 줄이길 기대하는 "
                        "절충안. raw centroid_emb를 쓰는 설명①(hard_assignment/ "
                        "centroid_x/confidence) 계산에는 관여하지 않음."
                    ))
parser.add_argument("--fusion_mode", type=str, default="proto_dev",
                    choices=["concat", "residual", "gated_sum", "anchor_gate",
                             "context_gated_beta", "proto_residual",
                             "proto_query_residual", "proto_only", "proto_only_linear",
                             "query_only_linear", "proto_dev", "proto_dev_agg", "proto_residual_query"],
                    help=(
                        "[2026-07, 되돌림] 'residual'을 잠시 기본값으로 뒀었으나, 이후 "
                        "폭넓은 비교 실험(6개 데이터셋)에서도 concat 대비 일관된 우위를 "
                        "못 찾아 기본값을 원래대로('concat', main 브랜치와 동일) 되돌림. "
                        "'residual'(query+β·agg)은 계속 명시적으로 선택 가능(비교/ablation "
                        "목적). study_pkl_tag는 이제 concat/residual 둘 다 명시적으로 태그를 "
                        "매김('..fusion_concat'/'..fusion_residual') — 이전(태그 없음=concat) "
                        "규칙으로 저장된 옛날 study 파일은 새 태그 규칙과 파일명이 달라지므로 "
                        "리네임이 필요할 수 있음(레포 마이그레이션 스크립트 참고)."
                    ))
parser.add_argument("--n_prototypes", type=int, default=None,
                    help=("prototype 개수를 직접 지정합니다(구조 변수). 생략하면 "
                          "clip(sqrt(N), C, N/k)로 자동 계산합니다. "
                          "지정하면 study 파일명에 ..P{n}이 붙어 다른 P의 실험과 "
                          "섞이지 않습니다 — 예전에는 P만 다른 실험이 같은 study를 "
                          "덮어써서 재현이 불가능해진 사례가 있었습니다."))
parser.add_argument("--commitment", action="store_true",
                    help=("[ablation] commitment_loss 를 켠다(기본은 꺼짐). "
                          "⚠ §10: 제거해도 cos(q,c)가 안 떨어지고(3/5는 오히려 상승) "
                          "예측도 유의차가 없어(p=0.14) 기본에서 뺐다. STE+CE 가 이미 "
                          "정렬을 만든다. ⚠ reproduce.py 와 반드시 같아야 한다."))
parser.add_argument("--allow_self_retrieval", action="store_true",
                    help=("검색에서 자기 자신을 제외하지 않는다(기본은 제외). "
                          "⚠ reproduce.py 와 같아야 한다 — 이전엔 이 인자가 "
                          "optimize.py 에 없어 HPO 는 self 포함, 재현은 제외로 "
                          "달랐다. proto_dev 에서는 retrieval 이 예측 경로 밖이라 "
                          "예측에 영향이 없었지만 다른 fusion_mode 에서는 다르다."))
parser.add_argument("--gradient_codebook", action="store_true",
                    help=("centroid 를 gradient(codebook_loss + CE)로 학습합니다. "
                          "[v3 이전 기본값] 지금은 EMA 가 기본이므로 이 플래그로 "
                          "옛 동작을 재현합니다. A/O 비교 결과: 예측은 동등하고"
                          "(5 데이터셋 × 5 seed, 전 지표 p>0.4) centroid drift 만 "
                          "gradient 쪽이 크다(끝/최대 11~45% vs 1~16%)."))
parser.add_argument("--ema_codebook", action="store_true",
                    help=("[더 이상 필요 없음 — EMA 가 기본] 하위호환용. "
                          "EMA 로 한다. centroid_emb 가 optimizer 대상에서 빠지고, "
                          "매 배치 배정된 query 의 이동평균으로 직접 갱신된다. "
                          "⚠ reproduce.py 와 반드시 같아야 한다 — HPO 를 gradient "
                          "조건에서 하고 재현만 EMA 로 하면 서로 다른 아키텍처를 "
                          "기준으로 최적화한 셈이 된다."))
parser.add_argument("--disable_dead_reinit", action="store_true",
                    help=("죽은 centroid 재초기화를 끈다. ⚠ A(gradient) vs O(EMA) "
                          "비교에서는 **양쪽 다 꺼야 한다.** 재초기화는 centroid "
                          "위치를 바꾸는 또 하나의 갱신 경로라, 켜두면 비교 대상이 "
                          "'갱신 방식 하나'가 아니라 둘이 된다."))
parser.add_argument("--nbr_lambda", type=float, default=0.0,
                    help=(
                        "[기본 0 — 제거됨] L_nbr 가중치. ⚠ reproduce.py와 반드시 같아야 한다 — "
                        "HPO가 L_nbr 없는 조건에서 HP를 찾고 재현에서만 켜면, 서로 다른 "
                        "아키텍처를 기준으로 최적화한 셈이 된다."
                    ))
parser.add_argument("--use_context_emb", action="store_true",
                    help=(
                        "[2026-07, deprecated — 하위호환용] use_context_emb=True가 다시 "
                        "기본값(v1 복원)이라 이 플래그는 더 이상 아무 효과가 없음(줘도 안전 — "
                        "어차피 기본 동작). context_emb를 head에서 빼려면(v2식) "
                        "--no_context_emb를 쓸 것."
                    ))
parser.add_argument("--no_context_emb", action="store_true",
                    help=(
                        "[2026-07, 되돌림 — 다시 실제 동작하는 플래그] fusion_mode와 같은 "
                        "이유로 use_context_emb 기본값을 True(v1)로 되돌리면서, 이 플래그가 "
                        "다시 'context_emb를 head 입력에서 뺀다'(v2식 비교용)는 원래 의미를 "
                        "함. --use_context_emb는 이제 하위호환용 no-op."
                    ))
parser.add_argument("--disable_retrieval_branch", action="store_true",
                    help=(
                        "[2026-07, 추가] '진짜' retrieval-free baseline(Model 2) HPO용 — "
                        "reproduce.py의 같은 이름 플래그와 정확히 같은 의미. 이 모드에서는 "
                        "k/n_prototypes/evidence_metric 등 retrieval 관련 하이퍼파라미터가 "
                        "전혀 안 쓰이므로(모델이 아예 retrieval을 안 함) 이 플래그를 켠 채로 "
                        "HPO를 새로 돌리는 건 대부분 낭비 — retrieval 있는 구조로 이미 돌린 "
                        "study의 embed_dim/dropout/lr 등만 재사용하고 싶다면 reproduce.py "
                        "쪽에서 --disable_retrieval_branch만 켜고 기존 study를 그대로 "
                        "불러오는 게 낫다(다만 study_pkl_tag가 이 플래그 값에 따라 다른 "
                        "파일을 가리키므로, 그러려면 optimize.py도 한 번은 이 플래그를 켠 "
                        "채로 시작해 study 파일을 만들어야 함 — 파일명 태그: '..no_retrieval')."
                    ))
parser.add_argument("--cat_combine", type=str, default="onehot", choices=["sum", "concat", "onehot"],
                    help=(
                        "categorical embedding 결합 방식. 'onehot'(기본값, 채택 확정)은 "
                        "TabR/ModernNCA 계보 — 학습 파라미터 없는 순수 one-hot. 'sum'/"
                        "'concat'은 이전 실험용 옵션(reproduce.py와 동일)."
                    ))
parser.add_argument("--cat_embed_dim", type=int, default=16,
                    help="cat_combine=concat일 때 컬럼별 embedding 차원 (기본 16).")
parser.add_argument("--num_embedding", type=str, default="ple",
                    choices=["linear", "ple", "plr_lite"],
                    help=(
                        "numeric feature 인코딩 방식. 'ple'(기본값, 채택 확정 — 2026-07 갱신)은 "
                        "PiecewiseLinearEmbeddings(activation=False) — TabM(Gorishniy et al. 2024) "
                        "기본값과 동일 구조. 4개 데이터셋(profb/vehicle/credit-g/jasmine) 실측 근거: "
                        "PLR 대비 val 붕괴(무작위 수준 trial)가 0건으로 감소(PLR은 vehicle 2건, "
                        "credit-g 1건 발생) + routing_scale/PLR 3종 제거로 HPO 탐색 공간이 13→9차원 "
                        "으로 축소됨. 다만 top5-test 성능은 데이터셋마다 갈렸고(4개 중 1개만 PLE "
                        "우세, 나머지는 PLR 우세 — 명확한 성능 우위는 아님), centroid "
                        "margin_percentile은 4개 데이터셋 전부에서 PLE가 더 낮게 나옴(원인 미상, "
                        "추가 조사 필요) — 즉 '성능이 더 좋아서'가 아니라 '재앙적 실패를 없애고 "
                        "탐색을 단순화하기 위해' 채택한 것임을 분명히 해둠. 'plr_lite'는 이전 "
                        "기본값(TabR/ModernNCA 계보) — 필요시 여전히 선택 가능."
                    ))
parser.add_argument("--num_bins", type=int, default=8,
                    help="num_embedding=ple일 때 컬럼당 구간(bin) 개수 (기본 8).")
# [제거됨] --plr_n_frequencies/--plr_freq_scale/--plr_out_dim
# num_embedding=plr_lite면 이제 search_space.py의 get_search_space()가
# trial마다 이 값들을 직접 탐색함(Gorishniy et al. 2022 권장 방식) —
# 고정 CLI 플래그로 두면 100 trial이 전부 같은 값을 써서 일부 데이터셋
# (mfeat-fourier, vehicle 등 numeric-only)에서 완전 붕괴 trial이 반복
# 관찰됨. reproduce.py(최종 1회 학습)에서는 여전히 고정값이 필요하므로
# 거기엔 이 플래그들이 남아있음 — HPO가 찾은 값은 best_params에 저장되고
# reproduce.py가 그 study를 재사용할 때 자동으로 반영됨.
args = parser.parse_args()

# [2026-07, 되돌림] use_context_emb=True가 다시 기본값(v1) — reproduce.py의
# 같은 번역 로직을 그대로 재사용. --no_context_emb를 명시적으로 주지 않으면
# (기본, 대부분의 실행) context_emb 포함(v1). --no_context_emb를 주면 v2식
# (context_emb 제외)으로 명시적 전환. --use_context_emb는 이미 기본 동작과
# 같아서 이제 아무 효과 없는 하위호환 플래그.
if args.no_context_emb:
    args.use_context_emb = False
else:
    args.use_context_emb = True

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)   # ← MultiTab 원본과 동일한 위치

import optuna, torch, json, joblib, datetime, math, gc
from libs.data import TabularDataset
from libs.eval import calculate_metric, is_study_todo, check_if_fname_exists_in_error, get_preds_and_probs
from libs.search_space import (get_search_space, suggest_initial_trial,
                              params_to_model_kwargs, study_pkl_tag,
                              HPO_TRAINING_SCHEDULE, DEFAULT_K_NO_TUNE,
                              _K_UNTUNED_MODES, SEARCHED_K_CHOICES)
from libs.supervised import TabERAWrapper
from libs.tabera import TabERA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.INFO)

# ─────────────────────────────────────────────────────────────
# 데이터셋 정보 로드  (MultiTab 원본과 동일)
# ─────────────────────────────────────────────────────────────

with open("dataset_id.json", "r") as file:
    data_info = json.load(file)

tasktype = data_info.get(str(args.openml_id))["tasktype"]
print(tasktype)

# ─────────────────────────────────────────────────────────────
# 저장 경로 설정  (MultiTab 원본과 동일)
# ─────────────────────────────────────────────────────────────

if not args.savepath.endswith("optim_logs"):
    savepath = os.path.join(args.savepath, "optim_logs", f"seed={args.seed}")
else:
    savepath = args.savepath
if not os.path.exists(savepath):
    os.makedirs(savepath)

_ablation_tag = study_pkl_tag(
    no_offset_correction=args.no_offset_correction,
    retr_proj_mode=args.retr_proj_mode,
    global_retrieve=args.global_retrieve,
    detach_context_grad=args.detach_context_grad,
    context_projection=args.context_projection,
    cat_combine=args.cat_combine,
    num_embedding=args.num_embedding,
    evidence_metric=args.evidence_metric,
    fusion_mode=args.fusion_mode,
    use_context_emb=(not args.no_context_emb),
    disable_retrieval_branch=args.disable_retrieval_branch,
    # ⚠ 자동 계산 P는 여기서 알 수 없다(데이터 로드 전). 명시 지정한
    #   경우에만 태그에 넣는다 — 자동 P가 코드 버전에 따라 달라져 기존
    #   study와 어긋나는 경우는 아래 가드가 막는다.
    n_prototypes=args.n_prototypes,
    # 기본이 EMA 이므로, 태그는 '옛 동작(gradient)' 일 때만 붙인다.
    gradient_codebook=args.gradient_codebook,
    no_commitment=not args.commitment,
    disable_dead_reinit=args.disable_dead_reinit,
    nbr_lambda=args.nbr_lambda,
    num_bins=args.num_bins,
    cat_embed_dim=args.cat_embed_dim,
    detach_retr_grad=args.detach_retr_grad,
)
fname = os.path.join(savepath, f"data={args.openml_id}{_ablation_tag}..model=tabera.pkl")

# ─────────────────────────────────────────────────────────────
# 중복 실행 방지  (MultiTab 원본과 동일)
# ─────────────────────────────────────────────────────────────

train = True
if os.path.exists(fname):
    study = joblib.load(fname)
    train = is_study_todo(study, tasktype)
else:
    study = (optuna.create_study(direction="minimize") if tasktype == "regression"
             else optuna.create_study(direction="maximize"))
    initial_trial = suggest_initial_trial()
    study.enqueue_trial(initial_trial)
    train = check_if_fname_exists_in_error(fname)

completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
remaining_trials = max(0, args.n_trials - completed_trials_count)

# ─────────────────────────────────────────────────────────────
# 메인 최적화  (MultiTab 원본 구조 최대한 유지)
# ─────────────────────────────────────────────────────────────

if train:
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    import platform
    env_info = "{0}:{1}".format(platform.node(), args.gpu_id)
    print(env_info, device)

    # ── 헤더 출력 ──────────────────────────────────────────
    print("=" * 60)
    print("  TabERA  Hyperparameter Optimization")
    print("=" * 60)
    print(f"  Dataset : {data_info[str(args.openml_id)]['fullname']} (id={args.openml_id})")
    print(f"  Task    : {tasktype}  |  Device : {device}")
    print(f"  Trials  : {completed_trials_count} done / {args.n_trials} total  ({remaining_trials} remaining)")
    if args.no_offset_correction:
        print(f"  Ablation: T(query-neighbour) offset correction OFF (value=label_emb only)")
    if args.retr_proj_mode != "none":
        print(f"  [S-1A] retrieval 전용 표현: retr_proj={args.retr_proj_mode}, "
              f"detach_retr_grad={args.detach_retr_grad}")
    if args.global_retrieve:
        print(f"  Diagnostic: retrieve() group-constraint OFF (global KNN, context_emb unaffected)")
    if args.detach_context_grad:
        print(f"  Diagnostic: task_loss gradient to centroid_emb DETACHED (diversity_loss only)")
    if args.context_projection:
        print(f"  Adjustment: context_emb routed through learned Linear projection before head")
    if args.evidence_metric != "euclidean":
        print(f"  Adjustment: evidence_metric={args.evidence_metric} "
              f"(AttentionAggregator similarity space, default=euclidean)")
    print(f"  Encoding: cat_combine={args.cat_combine}, num_embedding={args.num_embedding}")
    print(f"  Save    : {fname}")
    print("=" * 60)

    # ── 데이터 로드  (MultiTab: TabularDataset + _indv_dataset) ──
    dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    y_std = dataset.y_std

    print(f"  Train/Val/Test : {len(y_train):,} / {len(y_val):,} / {len(y_test):,}"
          f"  |  Features: {dataset.n_features}")
    print("-" * 60)

    # output_dim: n_classes 사용 (y가 1D이므로 shape[1] 불가)
    output_dim = dataset.n_classes if tasktype == "multiclass" else 1

    # ── n_prototypes ────────────────────────────────────────────────
    #
    # 기본값은 **P = sqrt(N)** 이다. 바꾸지 않는다.
    #
    # [왜 sqrt(N) 인가]
    #   평균 그룹 크기 |G| = N/P = sqrt(N) 이므로, P가 커지면 prototype
    #   표현력이 늘고 그룹이 작아진다 — 그 둘의 균형점이다. centroid는
    #   TabERA에서 세 역할을 동시에 한다.
    #       prediction anchor (context_emb) / retrieval partition /
    #       explanation unit
    #   그래서 P는 일반적인 k-means의 k 선택 문제가 아니다.
    #
    # [한때 clip(sqrt(N), C, N/k) 로 바꿨다가 되돌린 이유]
    #   ds=1493 (N=1279, C=100)에서 P를 35 → 100 으로 올리면
    #       acc  0.772 → 0.821  (+4.9pp, paired t p=0.009, 5/5 seed)
    #       β    0.691 → 0.51   (편차가 분류를 덜 떠맡음)
    #   여기까지는 capacity mismatch 해소가 맞다. 그러나 동시에
    #       H(Y_G)     1.32  → 0.086
    #       alignment  0.367 → 0.949
    #       |G|/k      4.8   → 1.5
    #   가 되어, prototype이 class-specific memory unit 처럼 행동하고
    #   그룹 내부 구조(설명 계층 ②가 의존하는 것)가 사라진다.
    #   ⚠ 즉 이건 고쳐야 할 버그가 아니라 **두 목적이 서로 다른 P를
    #     요구하는 trade-off** 다. 기본값이 한쪽을 임의로 택하면 안 된다.
    #
    # [그래서 어떻게 하는가]
    #   기본은 sqrt(N) — TabERA의 prototype은 class prototype이 아니라
    #   해석 가능한 지역 서술자(local region descriptor)이기 때문이다.
    #   capacity 쪽을 보고 싶으면 --n_prototypes 로 명시한다(별도 study).
    #   C > P 인 경우는 경고만 남긴다 — 편차 항이 보완한다는 사실을
    #   알고 있어야 결과를 해석할 수 있다.
    _sqrtN = max(4, int(math.sqrt(len(y_train))))
    _C     = dataset.n_classes if tasktype == "multiclass" else None
    _k_ref = (DEFAULT_K_NO_TUNE if args.fusion_mode in _K_UNTUNED_MODES
              else max(SEARCHED_K_CHOICES))

    if args.n_prototypes is not None:
        n_proto_default = int(args.n_prototypes)
        print(f"  n_prototypes: {n_proto_default}  (--n_prototypes 로 직접 지정)")
    else:
        n_proto_default = _sqrtN
        print(f"  Auto n_prototypes: sqrt({len(y_train)}) = {n_proto_default}")

    if _C is not None and n_proto_default < _C:
        print(f"    ⚠ P({n_proto_default}) < C({_C}) — logits ≈ W·c 에서 c는 P개 "
              f"값만 취하므로 prototype 항만으로는 모든 클래스를 표현할 수 "
              f"없습니다. 편차 항 W·Δ 가 분류를 떠맡습니다(β 상승, "
              f"argmax_changed 상승). 성능을 보려면 --n_prototypes {_C} 로 "
              f"별도 study를 만드세요 — 다만 그룹 크기가 {len(y_train)//_C}로 "
              f"줄어 설명 계층이 약해집니다.")
    _grp = len(y_train) // max(n_proto_default, 1)
    if _grp < _k_ref:
        print(f"    ⚠ 평균 그룹 크기({_grp}) < k({_k_ref}) — retrieval이 "
              f"cross-group fallback을 탑니다.")

    # ── PLE(Piecewise Linear Encoding) 구간 경계 계산 (num_embedding=ple일 때만) ──
    # reproduce.py와 동일 로직 — objective() 밖에서 한 번만 계산 (trial마다
    # 다시 계산할 이유 없음, 데이터에서 파생되는 값이라 trial 하이퍼파라미터와 무관).
    num_bin_edges = None
    if args.num_embedding == "ple" and len(dataset.X_num) > 0:
        X_num_train = X_train[:, dataset.X_num]
        q = torch.linspace(0.0, 1.0, args.num_bins + 1, device=X_num_train.device)
        num_bin_edges = torch.quantile(X_num_train, q, dim=0).T.contiguous()

    # ── 자동 가설 생성 (컬럼 평균 기준) ──────────────────

    # ── [재현성 가드] 기존 study가 다른 P로 만들어졌는가 ──────────
    # ⚠ study 파일명에는 자동 계산 P가 안 들어간다. 그래서 코드 버전이
    #   바뀌어 자동 P가 달라지면(ds=1493: 35 → 100) **같은 파일에 다른
    #   구조의 trial이 섞이거나 덮인다.** 실제로 P=35 study가 P=100
    #   실행에 사라진 사례가 있었다. P는 구조 변수이므로 섞이면 안 된다.
    _prev_P = {t.user_attrs.get("n_prototypes_actual")
               for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE
               and t.user_attrs.get("n_prototypes_actual") is not None}
    if _prev_P and _prev_P != {n_proto_default}:
        raise SystemExit(
            f"\n[중단] 기존 study의 n_prototypes={sorted(_prev_P)} 인데 이번 실행은 "
            f"{n_proto_default} 입니다.\n"
            f"  파일: {fname}\n"
            f"  P는 구조 변수라 같은 study에 섞으면 재현이 불가능해집니다.\n"
            f"  → --n_prototypes {n_proto_default} 로 지정해 별도 study(..P{n_proto_default})를 "
            f"만들거나, 기존 study를 다른 이름으로 옮기세요.")

    global best_so_far
    best_so_far = study.best_value if completed_trials_count > 0 else None

    # ── Objective  (MultiTab 원본 구조와 동일) ─────────────
    def objective(trial):
        params       = get_search_space(trial, num_features=X_train.size(1),
                                        use_ema_codebook=not args.gradient_codebook,
                                        no_commitment=not args.commitment,
                                        data_id=args.openml_id, metric=args.metric,
                                        num_embedding=args.num_embedding,
                                        fusion_mode=args.fusion_mode)
        # sqrt(N) 기반 n_prototypes override (가설 ② 복잡도 개선)
        params["n_prototypes"] = n_proto_default
        trial.set_user_attr("n_prototypes_actual", n_proto_default)
        model_kwargs = params_to_model_kwargs(params, dataset.n_features, output_dim)

        model = TabERA(
            **model_kwargs,
            column_names=dataset.col_names,
            # [필수 수정 — 이전엔 아예 빠져 있었음] AttentionAggregator의
            # 이웃 라벨 인코딩이 classification(nn.Embedding)/regression
            # (nn.Linear)을 구분하려면 tasktype이 필요함. TabR 원본
            # (yandex-research/tabular-dl-tabr)도 이렇게 조건부로 나뉘어
            # 있음 — 없으면 명목형 클래스 라벨을 raw 정수로 취급하는,
            # 오늘 categorical feature에서 고친 것과 같은 문제가 생김.
            tasktype=tasktype,
            n_classes=(output_dim if tasktype == "multiclass" else (2 if tasktype == "binclass" else None)),
            # [수정] 기존 min(2*N, 10_000) 캡은 N_train > 10,000인 데이터셋에서
            # MemoryBank가 X_train 전체를 담지 못하게 만들어 sample_groups가
            # 실제 그룹의 일부(최대 28%, id=41027 기준)만 반영하게 됨.
            # keys+vals 메모리 비용은 N=35,855, D=256 기준 ~73MB로 미미하므로
            # X_train 전체를 담도록 캡을 없앰 (그룹-제약 KNN의 원래 설계 의도 복원).
            memory_size=len(y_train),
            # [ablation] --no_offset_correction 플래그로 T(query-neighbour)
            # 오프셋 보정을 켜고 끔. Optuna 탐색 대상이 아니라 이 실행 전체에
            # 고정 적용 (기본값 True = 기존 TabR 방식 그대로).
            use_offset_correction=not args.no_offset_correction,
            retr_proj_mode=args.retr_proj_mode,
            detach_retr_grad=args.detach_retr_grad,
            global_retrieve=args.global_retrieve,
            detach_context_grad=args.detach_context_grad,
            use_context_projection=args.context_projection,
            evidence_metric=args.evidence_metric,
            # ⚠ [감사 수정] 이전엔 이 인자를 아예 안 넘겨 생성자 기본값
            #   (False)이 쓰였다. reproduce.py 는 `not args.allow_self_retrieval`
            #   로 True 를 넘긴다 — HPO 와 재현이 다른 검색 규칙을 쓴 것이다.
            #   proto_dev 에서는 retrieval 이 예측 경로 밖이라 예측에 영향이
            #   없지만(실측 최대차 0.000e+00), 실제 불일치이므로 맞춘다.
            exclude_self_retrieval=(not args.allow_self_retrieval),
            fusion_mode=args.fusion_mode,
            # [v3] L_nbr — reproduce.py와 같은 값이어야 HPO가 같은
            #   아키텍처를 기준으로 탐색한다.
            nbr_lambda=args.nbr_lambda,
            # [조건 O] centroid 갱신 방식. reproduce.py 와 같아야 한다.
            use_ema_codebook=not args.gradient_codebook,
            # dead reinit 은 patience 를 학습 epoch 수보다 크게 두어 끈다
            # (reproduce.py 와 같은 방식 — 별도 분기를 만들지 않는다).
            #   reproduce.py 와 같은 값(1e9)을 써야 조건이 일치한다.
            **({"dead_reinit_patience": 10 ** 9} if args.disable_dead_reinit else {}),
            disable_retrieval_branch=args.disable_retrieval_branch,
            # ⚠ [감사 수정] reproduce.py 는 `not args.no_context_emb`(기본 True)
            #   를 쓰는데 여기는 `args.use_context_emb`(기본 False)를 썼다.
            #   --use_context_emb 는 deprecated(효과 없음)라고 적혀 있었으나
            #   실제로는 이 값이 그대로 전달되고 있었다.
            #   proto_dev/query_only_linear 에서는 head 를 안 타서 영향이
            #   없지만, concat 에서는 출력이 달라진다(실측 최대차 0.454).
            use_context_emb=(not args.no_context_emb),
            # [필수 수정 — 이전엔 아예 빠져 있었음] categorical/numeric feature
            # 인코딩. 이게 없으면 cat_col_idx=None이 돼서 cat_combine/
            # num_embedding 설정과 무관하게 raw-encoding 경로로 빠짐 — HPO가
            # reproduce.py와 다른 아키텍처를 기준으로 하이퍼파라미터를 찾게
            # 되는 불일치가 있었음 (reproduce.py는 최종 학습에서 categorical
            # embedding을 쓰는데, 그 best_params는 이걸 안 쓰던 시절 HPO
            # 결과였음). 채택 확정된 기본값(onehot+PLR lite)을 HPO도 그대로
            # 씀 — jasmine 등에서 봤던 "구조 바뀌었는데 하이퍼파라미터는
            # 헌 것" 문제의 근본 원인.
            cat_col_idx=list(dataset.X_cat),
            num_col_idx=list(dataset.X_num),
            cat_cardinalities=list(dataset.X_cat_cardinality),
            cat_combine=args.cat_combine,
            cat_embed_dim=args.cat_embed_dim,
            num_embedding=args.num_embedding,
            num_bin_edges=num_bin_edges,
            # plr_n_frequencies/plr_freq_scale/plr_out_dim은 여기서 안 넘김 —
            # num_embedding="plr_lite"면 model_kwargs(get_search_space가 trial마다
            # 탐색한 값, params_to_model_kwargs를 거쳐 옴)에 이미 들어있어서
            # 여기 또 넘기면 "같은 키워드 인자 중복" 에러가 남. plr_lite가
            # 아니면 애초에 안 쓰이는 값이라 안 넘겨도 무방(TabERA 기본값이
            # 대신 채워지지만 어차피 참조 안 됨).
        )

        wrapper = TabERAWrapper(model, params, tasktype,
                                  device=str(device), **HPO_TRAINING_SCHEDULE)
        wrapper._data_id = args.openml_id   # 에폭 tqdm에 data_id 표시
        # [v3] L_nbr용 raw feature kNN graph. nbr_lambda=0이면 즉시 반환한다.
        #   ⚠ 없으면 _nbr_graph가 None이라 L_nbr이 조용히 계산되지 않는다.
        #   프로세스 캐시가 있어 trial마다 다시 만들지 않는다.
        model.set_nbr_graph(X_train)
        wrapper.fit(X_train, y_train, X_val, y_val)

        # ── 평가: logits 1회 계산 → preds/probs 동시 추출 ──────
        wrapper.model.eval()
        with torch.no_grad():
            val_logits  = wrapper._forward_batched(X_val)
            test_logits = wrapper._forward_batched(X_test)
        preds_val,  probs_val  = get_preds_and_probs(val_logits,  tasktype)
        preds_test, probs_test = get_preds_and_probs(test_logits, tasktype)

        # regression: y_std 역정규화 후 metric 계산 (MultiTab 원본과 동일)
        if tasktype == "regression":
            val_metrics  = calculate_metric(y_val  * y_std, preds_val  * y_std, probs_val,  tasktype, "val")
            test_metrics = calculate_metric(y_test * y_std, preds_test * y_std, probs_test, tasktype, "test")
        else:
            val_metrics  = calculate_metric(y_val,  preds_val,  probs_val,  tasktype, "val")
            test_metrics = calculate_metric(y_test, preds_test, probs_test, tasktype, "test")

        for k, v in val_metrics.items():
            trial.set_user_attr(k, v)
        for k, v in test_metrics.items():
            trial.set_user_attr(k, v)

        # MultiTab 원본과 동일한 출력
        print(device, env_info, args.openml_id,
              data_info.get(str(args.openml_id))["name"], "tabera", savepath)
        print(val_metrics)
        print(test_metrics)
        now      = datetime.datetime.now()
        duration = now - trial.datetime_start
        print(f"### Optimization time for trial {trial.number}: {duration.total_seconds():.0f} secs")
        trial.set_user_attr("training_time", duration.total_seconds())

        # 최적화 목표: regression → rmse_val 최소화 / classification → acc_val 최대화
        result = val_metrics["rmse_val"] if tasktype == "regression" else val_metrics["acc_val"]

        # ── centroid margin 진단 반영 (percentile 기반, 문턱값 없음) ──
        # [배경] routing_scale은 forward(예측)에는 영향이 없지만(STE라서
        # hard_assignment는 양수 스케일에 불변), 학습 중 STE backward
        # gradient의 뾰족함에는 직접 영향을 준다. 실측 결과(credit-g:
        # routing_scale=1.49, margin_percentile≈0% vs socmob: 19.8,
        # ≈100% / SpeedDating: 13.77, ≈100%) — routing_scale이 낮게 나온
        # trial일수록 학습된 centroid 라우팅 구조가 무작위 수준(또는 그보다
        # 나쁨)에 머물 수 있음이 확인됨.
        #
        # [설계 원칙] search_space.py의 routing_scale 탐색 범위(1.0~20.0)는
        # 그대로 둔다 — 하한을 강제하지 않고, Optuna가 정확도와 함께 이
        # 부작용까지 스스로 학습해서 피하도록 objective만 조정한다.
        #
        # [z-score 문턱값 대신 percentile을 직접 쓰는 이유] 처음엔
        # "z_margin < -2.0(또는 2.0)이면 penalty"처럼 z-score에 문턱값을
        # 두는 방식을 썼는데:
        #   1. z-score는 정규분포 근사가 깔려 있고, "몇 z 이상이면 봐줄지"
        #      자체가 다시 임의의 선택이 됨 — 처음 -2.0(actively bad만
        #      잡음)으로 뒀다가 mfeat-zernike(z_margin=-0.15, "무작위와
        #      구분 안 됨"인데 -2.0보다는 커서 penalty가 전혀 안 걸림)
        #      에서 구멍이 확인돼 2.0으로 올렸는데, 그 2.0도 근거는
        #      "reproduce.py 진단 라벨(사람이 읽는 용도로 고른 2-시그마
        #      관행)을 그대로 재사용한 것"뿐이라 여전히 매직넘버였음.
        #   2. penalty 자체는 어차피 연속값(badness에 비례)이라, "유의미
        #      하다고 확신할 수 있는가"라는 이진 판단(z 문턱값이 필요한
        #      이유)이 애초에 필요 없음 — "이게 무작위보다 나은가"라는
        #      순위 정보만 있으면 됨.
        # 그래서 이미 계산해둔 50개 null 샘플 대비 실측 margin의 순위를
        # 직접 백분위(margin_percentile, supervised.py에서 계산)로 써서
        # penalty를 100% 연속으로 만든다 — percentile=100%(모든 무작위
        # 시도보다 나음)면 penalty=0, 50%(딱 무작위 평균만큼)면 penalty=
        # cap의 절반, 0%(무작위보다 항상 나쁨)면 penalty=cap. 어떤 z를
        # 기준으로 "봐줄지" 결정하는 문턱값이 코드 어디에도 없음.
        # penalty_cap(0.05)은 credit-g/mfeat-zernike/jasmine 실측 기준
        # 잡은 값 — 더 많은 데이터셋의 margin_percentile이 아래 user_attr에
        # 계속 쌓이므로, 나중에 이 상한 자체도 데이터로 재조정 가능.
        diag = wrapper.centroid_geometry_diag
        if diag is not None:
            trial.set_user_attr("centroid_z_top1",            diag["z_top1"])
            trial.set_user_attr("centroid_z_margin",           diag["z_margin"])
            trial.set_user_attr("centroid_margin_percentile",  diag["margin_percentile"])
            # [순수 로깅, penalty 미반영] "결과(최종 스냅샷)"가 아니라 "과정
            # 전체의 안정성"을 보는 지표 — credit-g trial #47(margin_percentile=
            # 1.0인데 학습 내내 reinit이 안 멈췄던 사례)처럼, 최종 스냅샷만
            # 보는 위 세 지표로는 안 잡히는 케이스가 있다는 게 실측 확인됨.
            # 다만 이 두 지표가 실제로 나쁜 결과(재현성, test 성능)와
            # 상관관계가 있는지 여러 trial에 걸쳐 먼저 확인한 뒤에 penalty
            # 반영 여부를 결정한다 — study.trials_dataframe()에서 이 두
            # 컬럼과 test 성능/margin_percentile 간 상관관계를 분석할 것.
            if "reinit_per_epoch" in diag:
                trial.set_user_attr("centroid_reinit_per_epoch", diag["reinit_per_epoch"])
            if "active_ratio_std" in diag:
                trial.set_user_attr("centroid_active_ratio_std", diag["active_ratio_std"])
            penalty_cap  = 0.05
            penalty_frac = penalty_cap * (1.0 - diag["margin_percentile"])
            if penalty_frac > 0.0:
                if tasktype == "regression":
                    result = result * (1.0 + penalty_frac)   # rmse는 클수록 나쁨 → 키움
                else:
                    result = result * (1.0 - penalty_frac)   # acc는 클수록 좋음 → 줄임
                trial.set_user_attr("centroid_penalty_frac", penalty_frac)

        # [메모리 정리] trial마다 새 모델/옵티마이저를 만드는데, PyTorch가
        # 이전 trial에서 쓴 GPU 메모리를 내부 캐시로 들고 있으면 다음 trial의
        # collapse 안전장치(supervised.py)가 "여유 메모리 0"으로 오판할 수
        # 있음 (실측 확인됨). trial 사이에 명시적으로 반납.
        # [추가] del만으로는 부족할 수 있음 — autograd 그래프(tensor↔grad_fn)가
        # 참조 순환을 만들면 CPython의 단순 참조 카운팅으로는 즉시 회수가
        # 안 되고, 순환 GC(gc.collect())가 돌아야 풀림. 작은 데이터셋
        # (id=41027, N=35,855)에서는 trial당 찌꺼기가 작아 눈에 안 띄었지만,
        # 더 큰 데이터셋(id=41150, N=104,050)+무거운 하이퍼파라미터 조합에서
        # trial을 거듭할수록(7번째 trial쯤) GPU가 서서히 채워지다 못해
        # collapse 안전장치가 오작동하는 현상으로 실측 확인됨.
        del model, wrapper
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return result

    # ── 콜백  (MultiTab 원본과 동일) ──────────────────────
    def stop_when_reached_optimal(study, trial):
        if study.best_value >= 1.0:
            study.stop()

    if tasktype == "regression":
        study.optimize(objective, n_trials=remaining_trials,
                       callbacks=[lambda study, trial: joblib.dump(study, fname)])
    else:
        study.optimize(objective, n_trials=remaining_trials,
                       callbacks=[stop_when_reached_optimal,
                                  lambda study, trial: joblib.dump(study, fname)])

    # ── 학습 시간 집계  (MultiTab 원본과 동일) ─────────────
    total_training_time = sum([
        trial.user_attrs.get("training_time", 0)
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ])
    study.set_user_attr("total_training_time", total_training_time)

    # ── 결과 저장  (MultiTab 원본과 동일) ─────────────────
    print("#############################################")
    print(env_info)
    print(study.best_trial.user_attrs)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(savepath, f"data={args.openml_id}{_ablation_tag}..seed={args.seed}..model=tabera.csv"), index=False)
    joblib.dump(study, fname)
    print(fname)
    print("#############################################")