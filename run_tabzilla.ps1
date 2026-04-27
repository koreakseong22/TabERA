# TabERA × TabZilla Benchmark
# 28개 데이터셋 (N≤50,000)
# 사용법: .\run_tabzilla.ps1

$gpu_id = 0
$n_trials = 100
$seed = 1

$datasets = @(
    10,  # lymph                          N=    148 F= 19 multiclass
    7,  # audiology                      N=    226 F= 70 multiclass
    51,  # heart-h                        N=    294 F= 14 binclass
    25,  # colic                          N=    368 F= 27 binclass
    334,  # monks-problems-2               N=    601 F=  7 binclass
    11,  # balance-scale                  N=    625 F=  5 multiclass
    470,  # profb                          N=    672 F= 10 binclass
    29,  # credit-approval                N=    690 F= 16 binclass
    40981,  # Australian                     N=    690 F= 15 binclass
    54,  # vehicle                        N=    846 F= 19 multiclass
    31,  # credit-g                       N=  1,000 F= 21 binclass
    1494,  # qsar-biodeg                    N=  1,055 F= 42 binclass
    934,  # socmob                         N=  1,156 F=  6 binclass
    1493,  # one-hundred-plants-texture     N=  1,599 F= 65 multiclass
    14,  # mfeat-fourier                  N=  2,000 F= 77 multiclass
    22,  # mfeat-zernike                  N=  2,000 F= 48 multiclass
    1067,  # kc1                            N=  2,109 F= 22 binclass
    41143,  # jasmine                        N=  2,984 F=145 binclass
    46,  # splice                         N=  3,190 F= 61 multiclass
    1043,  # ada_agnostic                   N=  4,562 F= 49 binclass
    1489,  # phoneme                        N=  5,404 F=  6 binclass
    40536,  # SpeedDating                    N=  8,378 F=121 binclass
    # ---------------------------------------------------------------- # 아래부터는 재실험 필요.
    4538,  # GesturePhaseSegmentationProcessed N=  9,873 F= 33 multiclass
    1459,  # artificial-characters          N= 10,218 F=  8 multiclass
    846,  # elevators                      N= 16,599 F= 19 binclass
    1486,  # nomao                          N= 34,465 F=119 binclass
    41027,  # jungle_chess_2pcs_raw_endgame_complete N= 44,819 F=  7 multiclass
    151  # electricity                    N= 45,312 F=  9 binclass
)

$total = $datasets.Count
$done = 0; $failed = @(); $skipped = @()

Write-Host "===== TabHERA x TabZilla =====" -ForegroundColor Yellow
Write-Host "총 $total개 데이터셋"

foreach ($id in $datasets) {
    $done++
    $fname = ".\optim_logs\seed=$seed\data=$id..model=tabhera.pkl"
    if (Test-Path $fname) {
        Write-Host "[$done/$total] SKIP  id=$id" -ForegroundColor Gray
        $skipped += $id; continue
    }
    Write-Host "[$done/$total] START id=$id" -ForegroundColor Cyan
    python optimize.py --gpu_id $gpu_id --openml_id $id --n_trials $n_trials --seed $seed
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$done/$total] DONE  id=$id" -ForegroundColor Green
    } else {
        Write-Host "[$done/$total] FAIL  id=$id" -ForegroundColor Red
        $failed += $id
    }
}

Write-Host ""
Write-Host "===== 완료 =====" -ForegroundColor Yellow
Write-Host "완료: $($done - $failed.Count - $skipped.Count)개"
Write-Host "스킵: $($skipped.Count)개"
Write-Host "실패: $($failed.Count)개"
if ($failed.Count -gt 0) {
    Write-Host "실패 ID: $($failed -join ', ')." -ForegroundColor Red
}