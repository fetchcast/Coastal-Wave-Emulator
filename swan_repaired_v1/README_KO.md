# SWAN 벤치마크 공통 수정본 v1

첨부한 train.py 및 두 개의 동일한 legacy 파일을 기준으로 작성했습니다.
기존 `~/swan/train.py`, legacy 원본, `runs/v2_focused_all`, `runs/v2_followup`에는 쓰지 않습니다.
예전 `patch_train_fraction.py`와 `followup_launcher.py`를 먼저 실행할 필요가 없습니다.
분할 패치 기능은 이번 수정본에 이미 통합되어 있습니다.

## 먼저 실행할 명령

기존 벤치마크의 작업 배정을 중단하고 필요한 체크포인트를 확보한 뒤 실행하세요.
이 패키지는 기존 프로세스를 종료하지 않습니다. 선택한 GPU가 사용 중이면 대기합니다.

```bash
cd /home/jovyan/swan
unzip -q SWAN_Repaired_Benchmark_v1.zip
nohup bash swan_repaired_v1/START_REPAIRED.sh --gpus 0,1,2 > repaired_run.log 2>&1 &
tail -f repaired_run.log
```

기본 명령은 순서대로 다음을 수행합니다.

1. CPU 회귀 검사 실행.
2. FNO, TNO, FFNO 각각 실제 자료로 optimizer 2회 갱신 및 소규모 평가.
3. 세 검사가 모두 통과하면 같은 세 모델의 대표 설정 학습.
4. 대표 설정 학습과 최종 평가가 끝나면 종료. 대형 후속 실험은 자동 실행하지 않음.

대표 설정은 세 모델 모두 공간 모드 24×24, 폭 64, 깊이 4, seed 42입니다.
TNO의 시간 모드는 4입니다. 이 설정은 구현 검증용이며 최종 최적 구성이라고 주장하지 않습니다.
모델별 파라미터 수는 같지 않습니다. 실제 파라미터 수가 training_audit.json에 기록됩니다.
처음 세 검사만 실행하려면 `--stage smoke`를 추가합니다.

```bash
bash swan_repaired_v1/START_REPAIRED.sh --stage smoke --gpus 0,1,2
```

같은 명령으로 재시작하면 검증된 완료 작업은 건너뛰고, 미완료 작업은 마지막 검증 주기의
전체 상태 체크포인트에서 재개합니다. 해당 주기 이후의 미저장 작업은 다시 실행합니다.
시작 시 코드와 입력 자료 지문이 기존 계획과 달라졌으면 중단합니다.
새 설정으로 비교할 때는 `--root /home/jovyan/swan/runs/repaired_v2`처럼 새 결과 폴더를 지정하세요.

## 파일과 경로

| 파일 | 역할 |
|---|---|
| train_repaired.py | 전체 아키텍처 코드와 수정된 FNO/TNO 스펙트럼 층, worker |
| legacy_repaired.py | 전체 자료 전처리·손실·평가 코드 |
| repair_support.py | 분할·방향 보정·sampler·학습 루프·상태 저장 |
| run_repaired.py | 격리된 실행 계획, GPU 배정, 결과 검증 |
| selftest.py | CPU 회귀 검사, 선택적 합성 NetCDF 통합 검사 |
| bench_epoch_report.py | 기존/신규 결과의 학습량 읽기 전용 보고 |
| START_REPAIRED.sh | 검사 후 실행하는 시작점 |

서버 기본 위치는 `/home/jovyan/swan`입니다. 아래 서버 자료는 기존 것을 그대로 사용합니다.

- wavm-Waves_2019_2020_v2.nc
- bnd_features.py, boundspec_segments.py
- bnd_2019_v2, bnd_2020_v2
- 기존 관측소 CSV. 없으면 해당 관측소 그림은 기존 동작대로 생략합니다.

서버 루트는 `--server-root`, NC 경로는 `--data`로 지정할 수 있습니다.
BND 경로는 `SWAN_BND_DIR_2019`, `SWAN_BND_DIR_2020` 환경변수로 지정할 수 있습니다.
서버의 기존 Python 환경에서 실행하며 패키지를 자동 설치하거나 업그레이드하지 않습니다.
필요한 주요 패키지는 torch, numpy, pandas, xarray, netCDF4, scipy, scikit-learn,
matplotlib, tqdm입니다. GPU 실행에는 사용 중인 CUDA에 맞는 PyTorch가 필요합니다.

출력 기본 위치는 `/home/jovyan/swan/runs/repaired_v1`입니다.
`repaired_run.log`에는 작업 시작·완료가 기록됩니다. 각 작업의 상세 로그는 해당 결과 폴더의
`attempt_*.log`입니다. 처음에는 NC의 유한값·시간축 검사와 BND 로딩 때문에 GPU가 대기할 수 있습니다.
성공한 NC 검사는 자료 지문과 time_steps를 기준으로 캐시합니다.

```bash
python3 swan_repaired_v1/run_repaired.py --report
python3 swan_repaired_v1/bench_epoch_report.py --root /home/jovyan/swan/runs/repaired_v1
```

기존 벤치마크를 읽기만 하려면 다음을 사용합니다.

```bash
python3 swan_repaired_v1/bench_epoch_report.py --root /home/jovyan/swan/runs/v2_focused_all
```

## 학습·평가 규칙의 변경

- FNO는 두 개의 부호 영역, TNO는 네 개의 부호 영역을 처리합니다. 격자가 작거나 홀수일 때
  영역을 중복 덮어쓰지 않도록 제한합니다. FFNO의 스펙트럼 층은 변경하지 않았습니다.
- 방향 손실은 sin/cos 벡터를 정규화한 뒤 원형 오차를 계산합니다. 모호한 채널 축 추정도 제거했습니다.
  원형 손실은 벡터 크기를 강제하지 않습니다. 작은 크기에서의 파향 불안정성을 확인할 수 있도록
  예측 벡터 길이와 원시 출력도 저장합니다.
- 파향 지도는 순환 색상표를 사용합니다. 파향 오차 지도는 0–180도의 최소 각도 차이입니다.
- 분할은 168시간, q=5, seed=42, embargo=sequence length로 고정합니다.
  분할 실패나 잘못된 분율은 예외로 종료합니다. 다른 분할·무작위 분할로 넘어가지 않습니다.
- 25%와 50%는 계층별 학습 블록의 목표 분율입니다. 블록 반올림으로 실제 표본 분율이 다를 수 있으며
  실제 인덱스와 개수를 저장합니다. 검증·시험 집합은 바뀌지 않습니다.
- 기본 방향 정책 `train_auto`는 모든 비교 실행에서 공통인 중첩 25% 학습 블록의 정답 시점만
  사용합니다. 검증·시험 정답은 변환 선택에 사용하지 않습니다. 분율은 0.25 이상만 지원합니다.
  채널 계약은 [Hs, Tm, sin, cos]이며 값의 음수 여부로 채널을 뒤집지 않습니다.
- 자료 메타데이터로 방향 관례가 확인됐다면 `--bnd-transform refl+270` 같은 고정 변환을 쓸 수 있습니다.
  기본값으로 270도 반사를 추정해서 강제하지 않았습니다. 자동 선택 결과도 물리적 관례의 증명은 아닙니다.
- 최종 모델은 **검증 Hs MAE가 최소인 EMA 체크포인트**입니다. 학습 종료 후 이를 실제로 불러와
  시험 평가합니다. 가중 검증 손실은 계속 기록하지만 체크포인트 선택 기준으로 사용하지 않습니다.
- train MAE와 validation MAE는 같은 EMA 가중치로 평가합니다. Train MAE에서는 peak 중복 추출을
  사용하지 않습니다. 훈련 중의 가중 train loss는 여전히 curriculum·학습 중 가중치의 영향을 받으므로
  그 값과 validation loss의 차이를 그대로 일반화 간격으로 해석하지 마세요.
- 학습량과 검증 시점은 성공한 optimizer 갱신 횟수 기준입니다. 기본 early stopping은 꺼져 있습니다.
  기준 갱신 간격은 `ceil(전체 학습집합의 peak sampler 길이 / 유효 배치 크기)`입니다.
  기본 유효 배치 크기는 모든 실행에서 4입니다. 대표 학습은 이 간격의 30배만큼 갱신합니다.
- 로그의 cycle은 이 검증 간격을 뜻하며, 줄인 자료를 한 번 순회한 native epoch와 다릅니다.
  실제 순회 횟수, 배치 수, optimizer 시도·성공·생략 횟수를 별도 기록합니다.
- D 실험도 같은 전체 자료 기준으로 갱신 예산과 검증 간격을 계산합니다.
  epoch 수를 단순히 1/분율로 늘리지 않습니다. Curriculum과 log_vars 동결 구간도 같은 갱신 기준입니다.
- Peak sampler는 상위 5% 집합에서 두 배 개수를 복원 추출합니다. Epoch별 curriculum은 공간 손실
  가중치만 바꾸며 sampler 길이를 바꾸지 않습니다. 동률에 따른 실제 peak 개수가 적용됩니다.
- 마지막 불완전 누적 묶음은 실제 배치 개수로 나눕니다. AMP로 optimizer 갱신이 생략되면
  scheduler와 EMA를 진행시키지 않습니다.
- AdamW의 log_vars에는 weight decay를 적용하지 않습니다. OneCycleLR의 그룹별 max_lr를 지정해
  본체 대비 0.1배 학습률을 유지합니다.
- 재개 파일에는 optimizer·scheduler·EMA·scaler와 난수 상태, sampler 순열·위치를 저장합니다.
  로딩 오류가 나면 반쯤 복원된 상태로 새 학습을 시작하지 않고 종료합니다.
- NaN 배치를 조용히 제외하거나 가짜 0 표본으로 대체하지 않습니다. 원본 NC의 wet cell을 먼저
  검사합니다. 육지의 정의되지 않은 방향은 공간 마스크로 제외하고 0으로 채웁니다.
- 정적 2D depth와 시간에 따라 주어진 depth를 모두 명시적으로 처리합니다.
- 단일 GPU/job 방식만 지원합니다. 여러 GPU에는 서로 다른 job을 배정합니다.

기존 benchmark의 숫자는 새 코드의 대조군으로 자동 재사용하지 않습니다.
전처리 또는 손실을 바꾸면 모든 최종 비교군에 같은 정책을 적용해야 합니다.
D 실험은 갱신 횟수가 같더라도 학습 분율에 따라 정규화 통계와 peak 분포가 달라질 수 있습니다.
따라서 데이터 다양성 하나만 완전히 분리한 인과 실험이라고 과장하면 안 됩니다.

## 후속 실험 원안과 메모리 제한

새 패치에 제안된 R/D/L/B 22개 구성은 유지하고, 수정된 공통 코드로 학습하는 100% 기준 실행
5개를 추가했습니다. 이 27개는 기본 실행에 포함되지 않습니다.

```bash
python3 swan_repaired_v1/run_repaired.py --stage followup --plan
```

FNO의 `w384 d8 m48`은 수정 후 스펙트럼 상태만으로도 EMA 평가 시 약 243 GiB가 필요합니다.
이는 활성값과 다른 파라미터를 제외한 계산입니다. 현재 한 장 B200에서는 해당 설정을 실행할 수
없으므로 원안의 전체 후속 실행은 사전 검사에서 차단됩니다. 구성 자체를 조용히 축소하지 않았습니다.
대표 학습을 확인한 뒤 메모리에 맞는 최종 구성을 결정해야 합니다.

사용자가 확정한 별도 계획은 JSON job 목록을 `--plan-file`로 지정할 수 있습니다.
설정 변경 시 새 결과 루트를 쓰고, 해당 루트의 smoke와 pilot을 통과해야 후속 단계가 실행됩니다.
`--stage followup`만으로 임의의 큰 실험이 시작되지 않도록 이 검증 절차를 넣었습니다.
논문 일정상 현재 권장하는 실행은 위 기본 명령의 smoke + pilot까지입니다.

## 검증 범위

CPU PyTorch에서 다음을 검사했습니다.

- 홀수·짝수 격자에서 2D/3D Fourier 영역 복원과 역전파.
- FNO/TNO/FFNO 전체 모델의 작은 입력에 대한 출력·역전파.
- 방향 오차 15도에 손실 0이 되던 반례와 359/1도 원형 오차.
- 학습 분율의 중첩, 제외 처리, 검증·시험 인덱스 보존, 잘못된 분율 차단.
- 보정 자료 외의 파향 정답을 바꿔도 방향 선택 결과가 유지되는지.
- Sampler와 dropout을 포함한 중단·재개와 연속 실행의 일치.
- 실제 optimizer hook 카운트, 강제로 한 번 생략한 갱신, 학습률 그룹 비율.
- 합성 NetCDF로 세 모델의 실제 worker, 최종 best-EMA 평가, 요약 저장.

합성 NetCDF 통합 검사는 BND-off입니다. 실제 bnd_features.py와 boundspec_segments.py는
이 첨부에 없어 해당 서버 helper를 실행해 검증하지 못했습니다. 방향 보정 함수 자체는 별도
합성 검사로 확인했습니다. 서버 GPU/실제 자료 검증은 자동 smoke 단계에서 수행됩니다.
CPU 재개 일치가 모든 CUDA 실행의 비트 단위 동일성을 보장하지는 않습니다.

```bash
python3 swan_repaired_v1/selftest.py --extended
```

그 밖의 아키텍처 구조, 해안 가중, Hs/Tm 손실·TV 항, 자료 경로 기본값은 첨부본을 유지했습니다.
기존 한글 주석·일부 docstring은 영어로 정리했으며 실행 동작을 바꾸기 위한 변경은 아닙니다.
