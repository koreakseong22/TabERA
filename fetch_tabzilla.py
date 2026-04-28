# TabERA — TabZilla Benchmark Runner
# 사용법:
#   1. python fetch_tabzilla.py        # TabZilla 36개 데이터셋 ID 조회 및 저장
#   2. .\run_tabzilla.ps1              # 자동 생성된 스크립트로 벤치마크 실행

import openml
import json
import os
import subprocess

# ── TabZilla Hard Datasets Suite (OpenML study id=379) ────────────
TABZILLA_STUDY_ID = 379
CACHE_FILE = "tabzilla_datasets.json"

def fetch_tabzilla_datasets():
    """OpenML에서 TabZilla 36개 데이터셋 정보 조회"""
    print("TabZilla Hard Datasets 조회 중...")
    suite = openml.study.get_suite(TABZILLA_STUDY_ID)
    task_ids = suite.tasks
    print(f"  총 {len(task_ids)}개 task 확인")

    datasets = []
    for i, task_id in enumerate(task_ids):
        try:
            task = openml.tasks.get_task(task_id, download_data=False)
            dataset = openml.datasets.get_dataset(
                task.dataset_id, download_data=False,
                download_qualities=True, download_features_meta_data=False
            )
            qualities = dataset.qualities
            n = int(qualities.get('NumberOfInstances', 0))
            f = int(qualities.get('NumberOfFeatures', 0))
            c = int(qualities.get('NumberOfClasses', 0))
            task_type = str(task.task_type)

            # tasktype 판별
            if c == 2:
                tasktype = 'binclass'
            elif c > 2:
                tasktype = 'multiclass'
            else:
                tasktype = 'unknown'

            datasets.append({
                'task_id':    task_id,
                'dataset_id': task.dataset_id,
                'name':       dataset.name,
                'n':          n,
                'f':          f,
                'classes':    c,
                'tasktype':   tasktype,
            })
            print(f"  [{i+1:2d}/{len(task_ids)}] {dataset.name:30s} "
                  f"N={n:7,d} F={f:3d} C={c:2d} {tasktype}")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(task_ids)}] task_id={task_id} 오류: {e}")

    with open(CACHE_FILE, 'w', encoding='utf-8') as fp:
        json.dump(datasets, fp, indent=2, ensure_ascii=False)
    print(f"\n{CACHE_FILE}에 저장 완료")
    return datasets


def filter_local(datasets, max_n=50000, max_f=500):
    """로컬 실행 가능한 데이터셋 필터링"""
    filtered = [d for d in datasets
                if d['n'] <= max_n
                and d['f'] <= max_f
                and d['tasktype'] in ('binclass', 'multiclass')
                and d['n'] > 0]
    filtered.sort(key=lambda x: x['n'])
    return filtered


def generate_ps1(datasets, n_trials=100, gpu_id=0, seed=1):
    """PowerShell 실행 스크립트 생성"""
    lines = [
        "# TabERA × TabZilla Benchmark",
        f"# {len(datasets)}개 데이터셋 (N≤50,000)",
        "# 사용법: .\\run_tabzilla.ps1",
        "",
        f"$gpu_id = {gpu_id}",
        f"$n_trials = {n_trials}",
        f"$seed = {seed}",
        "",
        "$datasets = @(",
    ]
    for d in datasets:
        lines.append(f"    {d['dataset_id']},  "
                     f"# {d['name']:30s} N={d['n']:7,d} F={d['f']:3d} {d['tasktype']}")
    lines += [
        ")",
        "",
        "$total = $datasets.Count",
        "$done = 0; $failed = @(); $skipped = @()",
        "",
        'Write-Host "===== TabERA x TabZilla =====" -ForegroundColor Yellow',
        'Write-Host "총 $total개 데이터셋"',
        "",
        "foreach ($id in $datasets) {",
        "    $done++",
        '    $fname = ".\\optim_logs\\seed=$seed\\data=$id..model=tabera.pkl"',
        "    if (Test-Path $fname) {",
        '        Write-Host "[$done/$total] SKIP  id=$id" -ForegroundColor Gray',
        "        $skipped += $id; continue",
        "    }",
        '    Write-Host "[$done/$total] START id=$id" -ForegroundColor Cyan',
        "    python optimize.py --gpu_id $gpu_id --openml_id $id --n_trials $n_trials --seed $seed",
        "    if ($LASTEXITCODE -eq 0) {",
        '        Write-Host "[$done/$total] DONE  id=$id" -ForegroundColor Green',
        "    } else {",
        '        Write-Host "[$done/$total] FAIL  id=$id" -ForegroundColor Red',
        "        $failed += $id",
        "    }",
        "}",
        "",
        'Write-Host ""',
        'Write-Host "===== 완료 =====" -ForegroundColor Yellow',
        'Write-Host "완료: $($done - $failed.Count - $skipped.Count)개"',
        'Write-Host "스킵: $($skipped.Count)개"',
        'Write-Host "실패: $($failed.Count)개"',
        'if ($failed.Count -gt 0) {',
        '    Write-Host "실패 ID: $($failed -join \', \')" -ForegroundColor Red',
        "}",
    ]
    ps1_content = "\n".join(lines)
    with open("run_tabzilla.ps1", "w", encoding="utf-8") as fp:
        fp.write(ps1_content)
    print("run_tabzilla.ps1 생성 완료")
    return ps1_content


if __name__ == "__main__":
    # 캐시 있으면 재사용
    if os.path.exists(CACHE_FILE):
        print(f"{CACHE_FILE} 캐시 로드...")
        with open(CACHE_FILE, encoding='utf-8') as fp:
            datasets = json.load(fp)
    else:
        datasets = fetch_tabzilla_datasets()

    all_datasets = datasets

    print(f"\n===== 전체 TabZilla 데이터셋 ({len(all_datasets)}개) =====")
    print(f"{'dataset_id':>12} {'이름':30s} {'N':>8} {'F':>4} {'task':10s}")
    print("─" * 68)

    for d in all_datasets:
        print(f"  {d['dataset_id']:>10d} {d['name']:30s} "
              f"{d['n']:>8,d} {d['f']:>4d} {d['tasktype']}")

    generate_ps1(all_datasets)

    print("\n실행 방법:")
    print("  1. python fetch_tabzilla.py")
    print("  2. ./run_tabzilla.ps1")
