# SWAN 사건 진단 v1

기존 v4/v4.1 평가 결과를 읽는 별도 후처리 패키지입니다. 기존 학습 프로세스, 체크포인트, 후보 선정, 정규화, 학습·평가 결과를 변경하지 않습니다.

## 수행 범위

기본 실행은 FNO·FFNO·TNO, seed 42·43·44, 2021년 루핏·오마이스·찬투입니다. 사건 창은 기존 events_2021.json을 그대로 사용합니다. 모델은 기존 selected_2021.json에서 읽으며 재선정하지 않습니다. 새 학습은 없습니다.

기존 v4는 사건별 실제 최대파고 시각의 공간 배열만 snapshots/*.npz로 저장했습니다. 이 파일을 확인하여 재사용하고, 나머지 시간은 기존 prepared 캐시와 최종 체크포인트로 재추론합니다. 정규화된 캐시 targets.npy는 정답이고 예측 배열이 아닙니다. hourly.csv도 공간 예측을 복원할 수 없습니다.

새 배열은 Hs만 물리 단위 m로 저장합니다. Tm 품질 문제를 해결하거나 방향장을 재평가하는 코드는 아닙니다. 2019–2020년은 저장된 split_indices.npz를 통한 사용 이력 점검을 제공합니다. 역사 사건의 재추론은 이 버전에 포함하지 않았습니다. 사용 이력을 확인하지 않고 학습 기간 전체를 독립 시험으로 분류하지 않습니다.

## 실행

서버 /home/jovyan/swan에 ZIP을 올립니다.

```bash
cd /home/jovyan/swan
unzip SWAN_Event_Diagnostics_v1.zip
python3 swan_event_diagnostics_v1/run.py inspect
nohup bash swan_event_diagnostics_v1/START.sh --gpus 2,3 > /home/jovyan/swan/iclr_event_diagnostics_v1.log 2>&1 &
tail -f /home/jovyan/swan/iclr_event_diagnostics_v1.log
```

기본 GPU는 2,3입니다. 지정 GPU에 기존 계산 프로세스가 있으면 오류로 중단하며 종료시키지 않습니다. 다른 실행 제어기와 GPU를 자동 협상하지 않습니다. 이 GPU를 다른 캠페인에 동시에 배정하지 마십시오. 현재 UNet-LSTM 학습 GPU 4는 사용하지 않습니다. tail -f의 Ctrl+C는 nohup 작업을 종료하지 않습니다.

기본 출력은 /home/jovyan/swan/runs/iclr_event_diagnostics_v1 입니다. 원본 평가 캐시가 삭제됐다면 오류로 중단합니다. 이 패키지는 2021년 전체 캐시를 자동 재생성하지 않습니다.

진행 확인

```bash
cat /home/jovyan/swan/runs/iclr_event_diagnostics_v1/status.json
tail -n 30 /home/jovyan/swan/runs/iclr_event_diagnostics_v1/logs/tno_s42.log
nvidia-smi
```

첫 실행은 기존 체크포인트 SHA256 계산 때문에 시작이 늦을 수 있습니다. 기본 예측 배열의 비압축 데이터 크기는 약 1.5 GB입니다. 그림과 작업 여유분을 포함해 3 GB 이상 여유를 권합니다. 실행 시간은 해당 서버의 추론 속도에 따라 달라집니다.

중단 후 동일 명령을 다시 실행하면, provenance가 같은 저장 프레임을 재사용합니다. 손상된 프레임이나 원본과 다른 체크포인트·정규화는 오류로 중단합니다. 코드나 모델·사건 목록을 바꿀 때는 --output으로 새 폴더를 사용합니다. 진행 중인 작업에 동일 명령을 겹쳐 실행하지 않습니다.

완료된 9개 모델을 모두 분석하려면 처음부터 다음과 같이 실행합니다. 세 모델 기본 실행과 동시에 실행하지 마십시오.

```bash
nohup bash swan_event_diagnostics_v1/START.sh \
  --gpus 2,3 \
  --models fno,ffno,tno,conv_swin,swin,convnext_lstm,convlstm,u_ffno,vit \
  --output /home/jovyan/swan/runs/iclr_event_diagnostics_all9_v1 \
  > /home/jovyan/swan/iclr_event_diagnostics_all9_v1.log 2>&1 &
```

## 출력

- all_event_metrics.csv, summary_event_metrics.csv: 사건별 전체 MAE/RMSE/편향, 실제 최대값 지점·시각의 오차, 최대값 시간 차이. summary는 시드 평균과 표본 표준편차, n_seeds를 표시합니다.
- all_wave_bins.csv, summary_wave_bins.csv: 실제 Hs 구간별 오차, 표본 수, 표본 비중, 전체 MAE 기여도. 구간은 음수, 0–1, 1–2, 2–3, 3–4, 4–5, 5–6, 6 m 이상입니다. 왼쪽 경계 포함, 오른쪽 제외입니다.
- common_bin_standardized_mae.csv: 모든 선택 사건에 표본이 있는 공통 파고 구간만 남기고, 사건을 합친 공통 표본 비중으로 MAE를 재계산합니다. retained_sample_fraction과 common_bins를 함께 보십시오. 실제 연간 성능이나 인과 효과를 나타내지 않는 탐색적 비교입니다. 기존 단순 MAE를 대체하지 않습니다.
- all_detection.csv, summary_detection.csv: 실제/예측 3 m·5 m 초과 여부의 TP/FP/FN/TN, recall, precision, false_alarm_ratio, false_positive_rate, CSI. 실제와 예측 모두 >= 경계를 사용합니다. 분모가 0이면 공란이며 0점으로 바꾸지 않습니다.
- all_phases.csv: 실제 영역 최대파고 시각의 ±12시간, 그 이전, 그 이후를 구분합니다. 이 구간 이름은 물리적 발달기·감쇠기를 보장하지 않습니다.
- models/{model}_s{seed}/events/{id}/hourly_diagnostics.csv: 공간 MAE와 고정된 실제 최대파고 격자의 시계열.
- 같은 폴더의 time_series.png 및 spatial_comparison.png: 기본적으로 seed 42 그림만 생성합니다. 모든 시드 수치는 계산합니다. --plot-all-seeds로 모든 그림을 생성합니다.
- spatial_errors.npz: 사건 평균 공간 MAE와 편향.
- frames/{target_index}.npz: pred_hs, true_hs, time, units. 같은 사건 폴더 mask.npy의 True 격자만 평가에 사용합니다. 육지 값은 해석하지 않습니다. 시간은 기존 평가와 동일한 UTC입니다.

공간 그림의 축은 격자 행·열입니다. 경위도 지도나 해안 거리별 통계가 아닙니다. 모델 예측 최대값은 실제 최대값과 위치가 다를 수 있으므로, 고정 지점의 시계열과 구분합니다. 최대값 시각이 평가 창 끝에 걸리면 시간 차이 해석도 창의 영향을 받습니다.

각 프레임은 원본 hourly.csv의 Hs MAE와 대조합니다. float32 및 물리 단위 변환의 반올림을 고려해 atol=2e-6 m, rtol=2e-4를 적용합니다. 이를 넘으면 오류로 중단합니다. 이것은 시공간 예측의 모든 값이 원래 계산과 완전히 같다는 증명은 아닙니다.

2021년 결과를 확인한 뒤 만든 진단은 사후 분석입니다. 새 지표에 맞춰 모델을 다시 선택하거나 튜닝했다면 2021년을 그대로 최종 독립 시험이라고 주장할 수 없습니다. 공간 격자·시간을 독립 표본으로 취급하는 유의성 검정은 수행하지 않습니다.

## 2019–2020년 사용 이력 점검

CPU에서 실행합니다. 저장 인덱스는 입력 시작점이므로 seq_length를 더해 목표 시각을 찾습니다. 학습 입력·목표 시간과의 겹침, 시간 공백도 기록합니다.

```bash
python3 swan_event_diagnostics_v1/audit_training_period.py \
  --output /home/jovyan/swan/runs/historical_exposure_audit_v1
```

출력 {model}_s{seed}_timeline.csv를 통해 원하는 태풍 기간의 목표 시각이 train/val/test/excluded 중 무엇인지 확인할 수 있습니다. train은 학습용 분할 포함 여부이지 샘플러가 해당 시간대를 몇 회 사용했는지를 뜻하지 않습니다. 학습 입력인 외력 시각 노출과 학습 목표 파고 노출도 구분합니다.

공식 자료 또는 기존 고정 기준으로 역사 사건 창을 정한 뒤 historical_events.template.csv에 id,name,start,end 열을 작성하면 사건별 집계를 할 수 있습니다. 시각 형식은 YYYY-MM-DDTHH:00:00, UTC이며 끝 시각도 포함합니다. 임의 날짜를 예시로 넣지 않았습니다.

```bash
python3 swan_event_diagnostics_v1/audit_training_period.py \
  --events-csv /home/jovyan/swan/historical_events.csv \
  --output /home/jovyan/swan/runs/historical_event_exposure_v1
```

역사 사건의 진단을 위한 준비 단계입니다. 역사 자료 전처리와 경계 입력을 현재 평가 캐시에 혼합하거나, 2021년 캐시를 역사 사건에 재사용하지 않습니다.

## 결과 공유

다음 명령은 CSV·JSON·PNG·로그만 압축합니다. 큰 예측 배열과 체크포인트는 제외합니다.

```bash
python3 swan_event_diagnostics_v1/collect_report.py \
  --output /home/jovyan/swan/SWAN_Event_Diagnostics_Report_v1.zip
```

9개 모델 실행이었다면 --root /home/jovyan/swan/runs/iclr_event_diagnostics_all9_v1 를 추가합니다.

## 검증 및 의존성

기존 SWAN 환경의 Python, NumPy, pandas, PyTorch, matplotlib를 사용합니다. 역사 시간축 점검에는 xarray가 필요합니다. 패키지 설치·업그레이드를 자동 실행하지 않습니다.

```bash
python3 -m unittest discover -s swan_event_diagnostics_v1 -p test_diagnostics.py -v
```

합성 자료에서 파고 구간 기여도 합, 임계값 혼동행렬, 최대파고 값, 시드 1개일 때 표준편차 공란, split 목표 시각 변환, 시간 공백 거부, 그림 생성을 검증했습니다. 실제 B200·전체 체크포인트를 이용한 추론은 제작 환경에서 실행하지 못했습니다. 원본 evaluate_2021.py와 campaign.py는 기존 v4.1에서 복사한 보조 모듈이며 원본 서버 파일을 수정하지 않습니다.
