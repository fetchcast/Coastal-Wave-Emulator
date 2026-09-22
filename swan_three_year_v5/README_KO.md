# SWAN Three-Year v5

v4를 유지하면서 별도 폴더에서 2019–2021년 자료로 FNO·FFNO를 새로 학습하고, 기존 2년 모델과 새 모델을 동일한 2022년 자료로 평가합니다. 원본 trainer와 기존 결과는 수정하지 않습니다. 실제 서버 GPU 학습은 사용자가 실행해야 합니다.

## 실험 범위

| 항목 | 설정 |
|---|---|
| 기존 모델 | v3에서 선택하고 완료한 FNO·FFNO, 각각 seed 42/43/44 |
| 새 모델 | 동일 구조·학습률·배치 설정의 FNO·FFNO, 각각 seed 42/43/44, 처음부터 학습 |
| 개발 자료 | 기존 2019–2020 자료에 2021년 추가 |
| 분할 | 기존 시간 간격·embargo 및 주별 분할 방식을 새 자료에 적용 |
| 학습 예산 | 기존 선택 모델과 같은 성공 업데이트 수와 검증 간격; 새 자료 30회 순회가 아님 |
| 평가 | 2022년 전체와 사전 고정한 JMA 태풍/열대저기압 사건; 2년/3년 모델 총 12개 |
| 추가 탐색 | 없음; TNO와 나머지 7개 모델은 이 v5의 새 학습 대상에서 제외 |
| 기본 GPU | v4 완료 후 0–3에서 최대 4개 학습 병렬; 평가 최대 2개 병렬 |

2019–2021년 모두를 100% 학습에 투입하지 않습니다. 세 해 안에서 학습·검증·내부 테스트를 나누며, 2022년 전체는 외부 시간 일반화 평가용으로 남깁니다. 정규화와 경계 방향 보정은 기존 trainer의 학습 전용 규칙으로 새로 산출합니다. 2021년이 새 모델 학습에 들어가므로, v4의 2021년 미학습 평가와 v5의 내부 2021년 결과를 같은 표에서 직접 비교하지 마십시오.

새 분할과 정규화도 바뀌므로 결과는 '학습 자료 확장 설정의 효과'입니다. 자료량만의 인과 효과, 다른 해역 일반화 또는 통계적 우월성을 자동으로 입증하지 않습니다. 같은 업데이트 수에서는 표본당 노출 횟수가 감소합니다. 학습 부족 여부는 2022년 결과를 보기 전에 새 검증 곡선으로 판단해야 합니다. 예산을 바꾸려면 별도 사전 고정 실험으로 분리하십시오.

## 1. 설치 및 경로 확인

```bash
cd /home/jovyan/swan
unzip SWAN_Three_Year_v5.zip
cat swan_three_year_v5/config.json
```

첫 PREPARE 실행 전에 config.json의 경로를 확인하십시오.

- 기존 자료: `/home/jovyan/swan/wavm-Waves_2019_2020_v2.nc`
- 2021 자료: `/home/jovyan/swan/swan_2021_nc_v2/wavm-Waves.nc`
- 경계 자료: 서버 루트의 `bnd_2019_v2`, `bnd_2020_v2`, `bnd_2021_v2`
- **가정한 2022 경로**: `/home/jovyan/swan/swan_2022_nc_v2/wavm-Waves.nc`, `/home/jovyan/swan/bnd_2022_v2`. 실제 경로가 다르면 첫 실행 전에 수정하십시오.
- 결과: `/home/jovyan/swan/runs/iclr_three_year_v5`

원본 `swan_repaired_v1` 및 `swan_bc_typhoon_v3`의 선택/결과 파일이 필요합니다. 데이터의 시간축·격자·마스크·변수 단위·차원을 검사합니다. 2021년은 전체 달력 연도가 필요하며 다음 해 1월 1일 끝점은 제외합니다. 2019–2020은 기존 모델의 time_steps만큼만 보존합니다. 중간 결측 시간은 보간하지 않고 기존 시간 간격 필터로 처리합니다.

동일 서버의 기존 학습 환경을 사용하십시오. numpy, netCDF4, xarray, pandas, matplotlib, scipy와 기존 trainer의 PyTorch/CUDA 등 의존성이 필요합니다. 잘 동작하는 PyTorch를 재설치하지 마십시오.

## 2. 지금 가능한 CPU 준비

```bash
cd /home/jovyan/swan
nohup bash swan_three_year_v5/PREPARE.sh > iclr_three_year_v5_prepare.log 2>&1 &
tail -f iclr_three_year_v5_prepare.log
```

기존 FNO·FFNO의 선택 모델 3시드가 모두 완료되어 있어야 합니다. 이 단계는 2022 자료를 읽지 않고 GPU 학습을 시작하지 않습니다. 다만 큰 NetCDF를 읽고 쓰므로 v4와 저장 장치 I/O를 공유합니다. I/O 병목이면 v4 완료 뒤 실행하십시오.

압축된 통합 NetCDF를 새로 만들며, 비압축 예상 용량과 여유 2 GiB를 확보한 뒤 진행합니다. 261×256 격자, float32 8변수, 약 2.6만 프레임이면 비압축 약 52 GiB이고, 학습 캐시·체크포인트·평가 캐시를 위한 추가 여유가 필요합니다. 생성 도중 중단된 임시 파일은 재실행 시 다시 작성합니다. 완료된 통합 파일은 출처와 수정 시각 검증 후 재사용합니다.

첫 준비 시 설정을 동결합니다. 이후 config.json을 임의로 바꾸면 중단하도록 되어 있습니다. 변경 실험은 별도 result_root와 패키지 사본으로 구성하십시오.

## 3. v4 완료 후 새 학습

```bash
cd /home/jovyan/swan
nohup bash swan_three_year_v5/START_TRAIN.sh > iclr_three_year_v5_train.log 2>&1 &
python3 swan_three_year_v5/status.py --watch 10
tail -f iclr_three_year_v5_train.log
```

기본값은 `runs/iclr_parallel_v4/completed.json`을 확인한 뒤 실행하며, 아직 없으면 학습을 시작하지 않고 종료합니다. 자동으로 기다리다 시작하는 큐가 아니므로 v4 완료 후 다시 실행하십시오. PREPARE 결과는 재사용됩니다.

먼저 모델별 2업데이트 smoke test를 실행합니다. 통과하면 총 6개 학습을 최대 4개씩 진행합니다. OOM이나 검증 오류가 발생하면 모델 용량을 몰래 줄이지 않고 중단합니다. 같은 명령 재실행 시 완료 결과를 재사용하고, 미완료 학습은 기존 trainer의 저장된 체크포인트에서 재개합니다.

GPU 8개를 추가로 **같은 서버에서 물리 인덱스 8–15로** 제공한다면, 첫 PREPARE 전에 config.json의 gpus를 그 목록으로 바꾸고 require_v4_complete를 false로 설정하여 v4와 병행할 수 있습니다. 다른 노드의 GPU를 자동으로 통합하지 않습니다. 기본값의 GPU 0–7과 동시 실행은 거부합니다.

## 4. 2022 완성 및 새 학습 완료 후 평가

```bash
cd /home/jovyan/swan
nohup bash swan_three_year_v5/EVALUATE_2022.sh > iclr_three_year_v5_eval.log 2>&1 &
python3 swan_three_year_v5/status.py --watch 10
tail -f iclr_three_year_v5_eval.log
```

2022년 전체 시간축과 격자, 경계 입력을 검사합니다. 경계 보간의 최대 간격은 6시간입니다. 처음 12시간은 입력 문맥으로 쓰므로 평가 대상은 8,748시간입니다. v3의 JMA best-track 원본이 있으면 재사용하고 없으면 evaluator의 다운로드 절차를 사용합니다. 사건 목록과 모든 후보는 예측 결과를 보기 전에 고정합니다.

태풍/열대저기압 기준은 wet grid에서 400 km 이내, 진입~이탈 ±24시간입니다. JMA grade 5를 경험한 태풍은 별도로 표시합니다. 고파랑 기준은 참값 Hs ≥3 m, ≥5 m입니다. 사건 시간창 밖이 반드시 잔잔한 시기는 아닙니다.

기존 2년 모델은 기존 정규화·체크포인트를 그대로, 새 모델은 새 학습 정규화를 사용합니다. 물리 단위로 복원한 평가 참값과 시간축이 12개 실행에서 일치하는지 검사합니다. 통합 보고서에 3시드 평균·표본 표준편차와 3년−2년 차이를 기록합니다. 유의성 p값은 생성하지 않습니다. 부호가 있는 peak bias·timing error는 음수라는 이유만으로 개선이 아닙니다.

## 결과 위치

`runs/iclr_three_year_v5/`:

- `controls/`: 기존 선택 후보와 체크포인트/정규화 정보
- `data/prepared.json`: 시간축·결측·입력 파일 출처
- `plan.json`, `protocol.json`: 고정 학습 계획
- `training/`: 6개 학습 결과와 validation history
- `results.csv`: 새 학습 결과 요약
- `training_completed.json`: 6개 학습 완료
- `events_2022.json`: 고정 사건 목록
- `evaluation/{two_year,three_year}_{fno,ffno}/`: 사건별 CSV, 시간별 CSV, 공간/시계열 그림
- `comparison_by_seed.csv`, `comparison_summary.csv`: 같은 2022년 기준 기존/새 모델 비교
- `completed.json`: 전체 비교 완료

## 구현 변경과 검증 범위

패키지 안 trainer 사본에 2021 경계 경로를 추가하고 provenance 버전을 분리했습니다. 모델 구조·손실·스펙트럼 연산자는 수정하지 않았습니다. `trainer_changes.json`에 원본/수정 SHA256을 기록했습니다. 기존 서버 파일을 패치하지 않습니다.

```bash
python3 -m unittest discover -s swan_three_year_v5 -p 'test_v5.py' -v
```

배포 전 CPU 테스트: 연도 끝점 제외, 중복 시간 거부, 결측 시간 기록, 2022년 평가 달력, JMA 2022 필터, 실제 합성 NetCDF 3개 연도 병합/재사용, 동일 참값의 12개 평가 비교 및 불일치 거부. 실제 B200 학습·실자료 2022 평가는 로컬에서 실행하지 않았으며 서버 smoke test와 데이터 검사를 통과해야 합니다.
