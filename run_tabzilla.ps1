# TabERA × TabZilla Benchmark — Multi-Seed Run
# 사용법: .\run_tabzilla.ps1
# seed=1~10 × 전체 데이터셋을 순차 실행합니다.
# 이미 완료된 (seed, dataset) 조합은 자동 스킵합니다.

$gpu_id   = 0
$n_trials = 100
$seeds    = 1..10   # seed 10개 (논문 관례)

$datasets = @(
    7,     # audiology                         N=    226 F= 70 multiclass
    10,    # lymph                             N=    148 F= 19 multiclass
    11,    # balance-scale                     N=    625 F=  5 multiclass
    14,    # mfeat-fourier                     N=  2,000 F= 77 multiclass
    22,    # mfeat-zernike                     N=  2,000 F= 48 multiclass
    25,    # colic                             N=    368 F= 27 binclass
    29,    # credit-approval                   N=    690 F= 16 binclass
    31,    # credit-g                          N=  1,000 F= 21 binclass
    46,    # splice                            N=  3,190 F= 61 multiclass
    51,    # heart-h                           N=    294 F= 14 binclass
    54,    # vehicle                           N=    846 F= 19 multiclass
    151,   # electricity                       N= 45,312 F=  9 binclass
    334,   # monks-problems-2                  N=    601 F=  7 binclass
    470,   # profb                             N=    672 F= 10 binclass
    846,   # elevators                         N= 16,599 F= 19 binclass
    934,   # socmob                            N=  1,156 F=  6 binclass
    1043,  # ada_agnostic                      N=  4,562 F= 49 binclass
    1067,  # kc1                               N=  2,109 F= 22 binclass
    1459,  # artificial-characters             N= 10,218 F=  8 multiclass
    1468,  # cnae-9                            N=  1,080 F=857 multiclass
    1486,  # nomao                             N= 34,465 F=119 binclass
    1489,  # phoneme                           N=  5,404 F=  6 binclass
    1493,  # one-hundred-plants-texture        N=  1,599 F= 65 multiclass
    1494,  # qsar-biodeg                       N=  1,055 F= 42 binclass
    4134,  # Bioresponse                       N=  3,751 F=1777 binclass
    4538,  # GesturePhaseSegmentationProcessed N=  9,873 F= 33 multiclass
    23512, # higgs                             N= 98,050 F= 29 binclass
    40536, # SpeedDating                       N=  8,378 F=121 binclass
    40981, # Australian                        N=    690 F= 15 binclass
    41027, # jungle_chess_2pcs_raw_endgame     N= 44,819 F=  7 multiclass
    41143, # jasmine                           N=  2,984 F=145 binclass
    41150, # MiniBooNE                         N=130,064 F= 51 binclass
    41159  # guillermo                         N= 20,000 F=4297 binclass
)

$total   = $datasets.Count * $seeds.Count
$done    = 0
$skipped = @()
$failed  = @()

Write-Host ""
Write-Host "===== TabERA x TabZilla  [seed 1~10] =====" -ForegroundColor Yellow
Write-Host "  Seeds   : $($seeds -join ', ')"
Write-Host "  Datasets: $($datasets.Count)개"
Write-Host "  Total   : $total runs  (이미 완료된 건 자동 스킵)"
Write-Host ""

foreach ($seed in $seeds) {

    Write-Host "━━━  Seed = $seed / 10  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" `
               -ForegroundColor Magenta

    foreach ($id in $datasets) {
        $done++

        $fname = ".\optim_logs\seed=$seed\data=$id..model=tabera.pkl"

        if (Test-Path $fname) {
            Write-Host "  [$done/$total] SKIP  seed=$seed  id=$id" `
                       -ForegroundColor Gray
            $skipped += "seed=$seed/id=$id"
            continue
        }

        Write-Host "  [$done/$total] START seed=$seed  id=$id" `
                   -ForegroundColor Cyan

        python optimize.py `
            --gpu_id    $gpu_id   `
            --openml_id $id       `
            --n_trials  $n_trials `
            --seed      $seed

        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [$done/$total] DONE  seed=$seed  id=$id" `
                       -ForegroundColor Green
        } else {
            Write-Host "  [$done/$total] FAIL  seed=$seed  id=$id" `
                       -ForegroundColor Red
            $failed += "seed=$seed/id=$id"
        }
    }

    Write-Host ""
}

$succeeded = $total - $failed.Count - $skipped.Count
Write-Host "===== 완료 =====" -ForegroundColor Yellow
Write-Host "  완료  : $succeeded"
Write-Host "  스킵  : $($skipped.Count)"
Write-Host "  실패  : $($failed.Count)"
if ($failed.Count -gt 0) {
    Write-Host "  실패 목록:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
}
Write-Host ""
Write-Host "다음 단계: python aggregate_results.py --seeds 1 2 3 4 5 6 7 8 9 10" `
           -ForegroundColor Yellow