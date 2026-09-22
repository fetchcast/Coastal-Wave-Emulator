# SWAN Parallel v4

FNO·FFNO의 완료된 세 시드를 먼저 2021년에 평가하고, 나머지 7개 아키텍처를 수정된 공통 학습 기준으로 실행하는 전체 추가 패키지입니다. 기존 repaired trainer 파일은 변경하지 않습니다. 기존 A/B/C 결과는 보존하고 미완료 B/C는 같은 job과 같은 체크포인트 경로에서 이어갑니다.

## 중요 변경

- v3의 전체 C 완료 대기를 없애고, 모델별 3개 시드가 준비되면 평가합니다.
- GPU 0/1은 기존 B/C, 2/3은 평가, 4/5/6/7은 기본 모델 학습에 고정 배정합니다. 두 제어기가 같은 GPU를 선점하는 경쟁을 피하기 위해 v3를 한 번 종료하고 v4가 제어를 맡습니다.
- **현재 학습은 마지막 저장 체크포인트부터 재개됩니다. 저장 이후 일부 업데이트는 반복될 수 있습니다. 무중단 인계는 아닙니다.**
- v3가 이미 2021년 평가 단계로 넘어간 경우 자동 인계를 거부합니다. 그때는 프로세스를 임의로 죽이지 말고 상태와 로그를 확인해야 합니다.
- FNO/FFNO/TNO 선택 규칙은 원래의 pilot seed42+A+B 중 validation Hs MAE 최소값입니다. 2021년 결과는 선택에 사용하지 않습니다.
- 다른 7개는 고정 후보를 seed42/43/44로 학습합니다. 추가 탐색은 하지 않습니다. **21개 정규 학습과 7개 2업데이트 사전 점검**입니다.
- 2021 평가기에서 모델별 추가 설정과 UNet feature 목록을 전달하도록 수정했습니다. preflight 체크포인트와 평가 모델의 state_dict 구조를 meta device에서 strict 검사한 뒤 정규 학습을 시작합니다.
- 모든 모델이 같은 content-keyed 2021 캐시를 공유하고, 캐시 생성은 한 프로세스만 수행합니다. 추가 디스크 약 33 GB 이상이 필요합니다. 학습 체크포인트 저장 공간은 별도로 필요합니다.
- 모델별 태풍 CSV와 그림을 먼저 생성합니다. 통합 CSV는 완료된 모델이 늘어날 때 갱신합니다.

## 실행

ZIP을 `/home/jovyan/swan`에 업로드한 뒤 아래 명령을 한 번 실행합니다.

```bash
cd /home/jovyan/swan
unzip SWAN_Parallel_Typhoon_Baselines_v4.zip
nohup bash /home/jovyan/swan/swan_parallel_v4/START.sh \
  > /home/jovyan/swan/iclr_parallel_v4.log 2>&1 &
```

`START.sh`에 인계와 실행이 포함되어 있습니다. 별도의 `kill`, 기존 START 재실행, 기존 파일 수정은 필요하지 않습니다. 실행 전 상태만 확인하려면 아래 명령을 사용합니다. 이 명령은 학습 작업을 시작하거나 기존 프로세스를 종료하지 않습니다.

```bash
python3 /home/jovyan/swan/swan_parallel_v4/run.py
```

현재 설치된 v3의 Python 파일 또는 repaired trainer의 해시가 제공 당시와 다르면 중단합니다. 이 검사를 제거하지 말고 오류와 수정된 파일을 확인하세요.

### 모니터링

```bash
python3 /home/jovyan/swan/swan_parallel_v4/status.py --watch 10
```

```bash
tail -n 80 /home/jovyan/swan/iclr_parallel_v4.log
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/original.log
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/baselines.log
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/eval_fno.log
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/eval_ffno.log
```

이제 원래 `iclr_bc_typhoon_v3.log`는 업데이트되지 않습니다. 기존 B/C 작업 결과는 원래 경로에 저장되지만 제어 로그는 위 original.log에 기록됩니다. 시작 시 데이터 및 캐시 준비로 GPU 사용률이 바로 올라가지 않을 수 있습니다.

## GPU 배정

| GPU | 작업 |
|---|---|
| 0,1 | 기존 TNO B 잔여 작업 → 기존 선택 규칙 → TNO C seed43/44 |
| 2,3 | FNO/FFNO 우선 2021 평가 → 준비된 다른 모델의 2021 평가 |
| 4,5,6,7 | 나머지 7개 모델의 사전 점검 → 고정 후보 21개 학습 |

FNO/FFNO는 각각 3개 시드를 순차 평가하며 두 모델이 병렬 진행됩니다. 7개 기본 모델은 seed42 작업을 먼저 큐에 넣고 이후 43/44 순서로 실행합니다. 각 모델의 3개 시드가 끝나면 평가 큐에 들어갑니다. 학습 전용 GPU가 먼저 끝나도 이번 버전은 GPU 배정을 자동 변경하지 않습니다. 자원 충돌을 피하기 위한 고정 배정이며, 모든 단계에서 8개가 계속 사용된다는 뜻은 아닙니다.

## 고정 후보 설정

첨부 원고에 명시된 구조를 사용하고, 원고에 생략된 세부 값은 현재 구현의 기본값을 명시적으로 고정했습니다. 이들은 **추가 탐색을 거치지 않은 고정 기준선**입니다. 과거 7개 모델 실행을 완전히 복제했다거나, 현재 3개 Fourier 계열과 동일한 탐색 예산으로 최적화했다고 보고하면 안 됩니다.

| 모델 | 고정 설정 |
|---|---|
| ConvNeXt-LSTM | dims 96/192/384, depths 2/2/2, recurrent hidden 256 |
| Conv-Swin | base width48, attention dim256, depth6, heads8, window8 |
| UNet-LSTM | features64/128/256/512/1024, recurrent hidden768 |
| UNet-FFNO | features128/256/512/1024/2048, spectral width256, depth4, modes16/16 |
| Swin | embed72, depths2/2/6/2, heads3/6/12/24, patch4, window8 |
| ConvLSTM | width128, depth2 |
| ViT | embed384, depth6, heads6, patch16 |

원고에 생략된 ConvNeXt stage depths, Swin stage depths/heads, attention window와 head 수 등은 현재 구현의 기본값을 명시적으로 고정했습니다.

공통 학습은 2019–2020 데이터, 경계 ON, 입력 길이12, 최대 학습률1e-4, weight decay1e-4, 30 cycles, early stopping 비활성화, microbatch1×accumulation4입니다. 원래 3개 계열의 모델별 설정은 변경하지 않습니다. 유효 배치와 성공 업데이트 예산은 비교하되 미니배치 크기 자체가 모두 같은 것은 아닙니다. 업데이트 예산·시간 분할·방향 보정 일치를 검사하고, 최종 성능과 실제 파라미터 수를 기록합니다.

설정은 `plans.py`와 실행 후 `baseline_plan.json`에 있습니다. 실행한 뒤 설정을 바꾸면 frozen-plan 검사가 거부합니다. OOM이 발생해도 width/depth를 자동 축소하지 않습니다. 사전 점검이 실패하면 기본 모델 학습 lane이 중단되고, TNO 및 가능한 평가 작업은 계속됩니다. `status.json`의 errors와 supervisor/attempt 로그를 확인하세요.

기존 7개 체크포인트는 학습·분할·정규화·수정 코드의 일치가 증명되지 않았으므로 자동 재사용하지 않습니다. 이번 패키지로 완료한 작업은 같은 명령 재실행 시 재사용됩니다. 이전 기본 모델 기록을 확인해 동일성이 입증되면 별도 감사 후 재사용을 추가할 수 있습니다.

## 결과 위치

```text
/home/jovyan/swan/runs/iclr_parallel_v4/
  status.json
  controller_config.json
  baseline_plan.json
  results.csv
  available_event_metrics_by_seed.csv
  available_event_metrics_seed_summary.csv
  baselines/
    preflight/
    fixed/
    evaluator_layout_checks.json
  evaluation/
    fno/
    ffno/
    tno/
    convnext_lstm/
    ...
  shared_2021/
  logs/
```

- `results.csv`: 3개 시드가 준비된 모델의 2019–2020 기존 테스트 지표 및 validation MAE, 파라미터 수, 학습 기록. 재개 실행의 wallclock은 총비용으로 해석하기 전에 누적 여부를 확인해야 합니다.
- `evaluation/fno/event_metrics_seed_summary.csv`: FNO의 연간/태풍별 평균과 seed SD. FFNO와 다른 모델도 동일합니다.
- `evaluation/<model>/figures/`: 태풍별 최대 Hs 시계열과 공간 오차 그림. 공간 지도는 seed42이며, 수치는 3개 시드를 모두 집계합니다.
- `available_event_metrics_seed_summary.csv`: **완료된 모델만** 합친 표입니다. 행이 없는 모델의 성능을 0으로 처리하지 않습니다.
- `completed.json`: 10개 모델, 각 3시드의 평가까지 완료.
- `partial_completed.json`: 일부 실패. 성공 결과는 보존됩니다.

2021은 같은 해역의 미학습 연도이며, 다른 해역 일반화 검증이 아닙니다. 태풍 사건과 고파랑 임계값은 v3에서 고정한 기준을 유지합니다. 겹치는 사건 창은 합집합 지표에서 한 번만 집계합니다. 비사건 구간을 잔잔한 날이라고 부르지 않습니다. Tm 이상값은 제거하거나 잘라내지 않고 QC 표에 기록합니다.

모델별 공간 그림의 색 범위는 그 그림 안에서 참값/예측을 공유합니다. 다른 모델 폴더의 그림끼리는 색 범위가 다를 수 있으므로 색의 진하기로 모델 순위를 비교하지 마세요. 수치 CSV를 사용하세요.

## 기존 분석 도구와의 관계

`SWAN_Evidence_Analysis_v1/analyze.py`는 원래 단일 폴더의 9개 실행을 가정합니다. **새 v4 결과에 그대로 실행하지 마세요.** 이번 패키지는 모델별 및 통합 정확도/태풍 표를 직접 만듭니다. 10개 모델에 대한 추가 통계 검정이나 추론 속도 실측은 이번 변경 범위에 포함하지 않았습니다. 기존 v3 전체 평가가 자동으로 다시 실행되어 FNO/FFNO를 중복 평가하는 일도 없습니다.

## 재시작

문제를 해결한 뒤 동일한 START 명령을 실행하면 완료된 학습·평가를 재사용하고, 미완료 작업은 저장 지점부터 재개합니다. 이미 v4가 실행 중이면 controller lock으로 두 번째 실행을 거부합니다. 첫 인계 때 기존 작업이 90초 안에 종료되지 않으면 새 작업을 시작하지 않습니다. 예상하지 못한 GPU 프로세스가 남아 있어도 중단합니다. 임의의 다른 프로세스를 종료하지 않습니다.

## 검증 범위

```bash
python3 -m unittest discover -s swan_parallel_v4 -p 'test_*.py' -v
```

고정 job 생성, 모델별 constructor 설정 전달, 임의 모델의 시드 집계, 미완료 결과 제외, 격리된 임시 프로세스의 인계, 합성 8748시간×3시드 비Fourier 모델의 태풍 CSV/그림 생성 테스트를 포함합니다. 실제 B200 학습과 실제 2021 자료 평가는 서버에서 실행해야 합니다. 실행 전 소스 해시 검사와 GPU preflight는 실제 환경 문제를 확인하기 위한 절차입니다.
