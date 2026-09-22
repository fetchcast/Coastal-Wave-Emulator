# SWAN ICLR Campaign v2 — 2021 독립 평가 추가

## 실행

ZIP을 /home/jovyan/swan에 업로드하고:

```bash
cd /home/jovyan/swan
unzip -q SWAN_ICLR_Campaign_v2.zip
nohup bash swan_iclr_campaign_v2/START.sh > iclr_campaign_v2.log 2>&1 &
```

v1이 실행 중이면 기다립니다. v1을 강제로 종료할 필요 없습니다.
v1이 종료되면 기존 코드 그대로 완료 결과를 검사하고 미완료 작업을 재개합니다.
v1과 v2의 기존 학습 결과 루트는 동일한 /home/jovyan/swan/runs/iclr_expanded_v1 입니다.
학습 계획과 campaign.py 파일을 변경하지 않았으므로 기존 코드 지문과 단계별 계획을 유지합니다.
새 2021년 평가 결과는 별도 /home/jovyan/swan/runs/iclr_2021_v2 에 저장합니다.
v1 원본 Python이나 데이터는 덮어쓰지 않습니다.

```bash
python3 /home/jovyan/swan/swan_iclr_campaign_v2/status_v2.py --watch 10
```

조회 Ctrl+C는 학습/평가를 멈추지 않습니다.

```bash
tail -n 50 /home/jovyan/swan/iclr_campaign_v2.log
```

평가 로그:

```bash
tail -n 30 /home/jovyan/swan/runs/iclr_2021_v2/models/fno_s42/evaluate.log
```

재시작도 최초 nohup 명령과 같습니다. 동일 v2 중복 실행은 잠금으로 막습니다.
독립 평가의 시간별 CSV에 정상 저장된 시각은 재시작 때 건너뜁니다.
강제 종료로 CSV 마지막 줄이 손상되면 명확한 오류로 멈춥니다. 로그 확인 후 해당 CSV를 보존/이동하고 그 모델만 재평가하세요.

## 기본 경로

- 기존 코드: /home/jovyan/swan/swan_repaired_v1
- 기존 학습 자료: /home/jovyan/swan/wavm-Waves_2019_2020_v2.nc
- 2021 자료: /home/jovyan/swan/swan_2021_nc_v2/wavm-Waves.nc
- 2021 경계: /home/jovyan/swan/bnd_2021_v2
- 기존 경계 보조 함수: /home/jovyan/swan/bnd_features.py 및 boundspec_segments.py

첨부 조사 보고서에서 2021 자료의 격자/해륙 마스크와 기존 자료의 일치를 확인했습니다.
8761개 시각은 2021-01-01 00시부터 2022-01-01 00시까지입니다.
필요한 모든 모델 입력 변수가 있습니다. 실행 시 다시 격자와 시간축을 확인합니다.

## 학습과 모델 선택

v1의 추가 본 학습 최대68개 + 구조 사전 확인29개를 유지합니다.
기존 pilot9개 재사용. 자세한 구조 탐색은 README_training_v1_KO.md 참조.
2021년 자료를 학습, 학습률 선택, 체크포인트 선택에 쓰지 않습니다.
2019–2020 검증 MAE로 선택된 모델별 구성의 3개 seed(총9개)만 평가합니다.
60-cycle 또는 분율 실험 결과로 최종 구성을 다시 고르지 않습니다.
2021 평가 성능을 본 다음 구성 선택을 변경하면 독립 시험으로 해석할 수 없으므로 피하세요.

## 2021 평가 처리

- 입력 길이12와 다음 시각 표적이라는 기존 규칙을 유지합니다.
- 2022-01-01 00시 레코드를 제외하고 2021년 8760개 프레임을 사용합니다.
- 2020년 입력을 이어 붙이지 않으므로 최초12시간은 입력 문맥으로만 사용합니다.
- 평가 표적은 2021-01-01 12시부터 2021-12-31 23시까지 8748개입니다.
- 저장된 normalization.json만 읽고 2021 자료로 정규화 범위를 다시 추정하지 않습니다.
- 저장된 direction_manifest.json의 chosen 변환을 경계 sin/cos에 적용합니다.
- 2021 표적 방향과의 비교로 방향 변환을 재선택하는 함수는 호출하지 않습니다.
- SWAN 방향 표적은 기존 학습과 같은 native convention입니다. 표시용 nautical 변환을 새로 적용하지 않습니다.
- 첫2021 프레임의 depth로 깊이 경사 특성을 한 번 계산하는 기존 전처리 규칙을 유지합니다.
- 원래 모델 wrapper를 생성하고 EMA 체크포인트를 strict=True로 읽습니다.
- 추론은 기존 최종 평가처럼 FP32/no autocast, batch1입니다.
- 기존 평가 마스크 kcs>0, wet-cell 균등 가중을 사용합니다.
- 예측이나 입력의 비유한 값은 오류로 처리하며 묵시적으로 표본을 제외하지 않습니다.
- 추가 spin-up 제외 기간은 설정하지 않았습니다. 시뮬레이션의 초기조건 영향을 별도로 판단해야 합니다.

경계 파일은37개지만 공간 매핑에는 기존 SEGMENTS 정의의 세그먼트만 사용합니다.
새 경계 세그먼트를 임의로 추가하지 않습니다. 사용하지 않은 이름은 캐시 complete.json에 기록됩니다.
각 사용 세그먼트의 시간축, 표적 시간 범위, 유한값을 검사합니다.
내부 시간 간격은 기존 방식의 시간 선형보간을 쓰되, 원관측 시각을 유지한 합집합 축에서 보간합니다.
외삽이나 0으로 채우기는 허용하지 않습니다. 마지막 시각의 보간에 필요한 2022 경계 레코드는 사용 가능하며 2022년을 평가한다는 뜻은 아닙니다.
기본 허용 경계 원자료 최대 간격은6시간입니다. 초과하면 중단하며 자동으로 완화하지 않습니다.
경계 원자료가 정확히 시간별이면 보간 없이 동일한 값이 사용됩니다.

## 저장과 동시 실행

2021년 자료를 16프레임씩 읽어 memmap으로 캐시합니다.
캐시는 정규화/방향변환/자료지문이 같을 때9개 모델 간 공유합니다.
서로 다른 정규화 또는 방향변환이면 별도 캐시가 필요합니다.
캐시1개당 약33 GB(십진 GB)가 필요합니다. 전처리 시작 전 여유 공간을 확인합니다.
캐시에는 입력과 표적이 있으며 원본 NetCDF를 병합하거나 수정하지 않습니다.
준비 과정은 CPU에서 순차 실행합니다. 실제 평가는 기본2개 GPU 동시 실행으로 디스크 경합을 줄입니다.
GPU 후보는0–7이며 이미 점유된 GPU는 기다립니다. 별도 런처를 같은 GPU에 동시에 시작하지 마세요.

동시 평가를4개로 늘리려면 시작할 때:

```bash
nohup bash /home/jovyan/swan/swan_iclr_campaign_v2/START.sh --eval-workers 4 > /home/jovyan/swan/iclr_campaign_v2.log 2>&1 &
```

위 명령을 기존 v2 실행과 동시에 중복 실행하지 마세요.
학습이 완료되어 있고 평가만 실행하려면 --eval-only 사용. 선택된9개 체크포인트가 없으면 중단합니다.

## 결과

/home/jovyan/swan/runs/iclr_2021_v2 아래:

- selected_2021.json: 평가 전에 동결한9개 모델, 정규화 출처, 방향 변환.
- summary_2021.csv: 모델/seed별 연간 지표.
- models/{model}_s{seed}/result.json: 연간 및 월별 지표, 프레임 수, 메모리, 출처.
- models/{model}_s{seed}/hourly.csv: 시간별 Hs/Tm/방향 오차와 실제 최대/평균 Hs.
- models/{model}_s{seed}/evaluate.log: 모델별 진행 상황.
- cache/{signature}/complete.json: 시간/경계 검사, 사용하지 않은 경계 세그먼트 등.
- completed.json: 독립 평가9개 완료.

hs_mean_frame_rmse는 시간마다 공간 RMSE를 구한 뒤 평균합니다. 기존 legacy rmse_m와 집계 방식이 같습니다.
hs_pooled_rmse는 시간·공간을 합친 제곱오차 평균의 제곱근입니다. 두 값을 혼용하지 마세요.
Tm도 동일한 두 집계 방식을 제공합니다. 방향은 최소 원형 각도차(도 단위)로 평가합니다.
방향 벡터 반경이1e-6보다 작은 예측 비율을 별도 보고합니다. 해당 방향은 해석에 주의해야 합니다.
월별과 시간별 결과는 사건 분석에 쓸 수 있으나 사건 선정/그림 자동 생성은 포함하지 않았습니다.
실행 시간은 입출력 포함이므로 추론 속도 벤치마크로 사용하지 마세요.

## 오류와 재시작

학습 오류 처리는 v1과 같습니다. 명시적 CUDA OOM만 자원 제외하고 계속합니다.
독립 평가는9개가 모두 필요하므로 어떤 평가가 실패하면 중단합니다. 실패를 성공이나 모델 제외로 처리하지 않습니다.
v1에 코드/계획 변경 오류가 발생하면 v2도 중단합니다. 지문을 지우거나 임의 수정하지 마세요.
v1 실행 중 v2를 시작하면30초마다 대기 메시지를 출력합니다. v1 종료 이후 정상 완료 여부를 확인/재개합니다.

## 검증 범위

CPU 모의 테스트10개 통과: 기존 실행 계획/선택/재시작/실패 격리와 연도 끝점 제외,
시퀀스 문맥, 시간누락 거부, 저장된 방향변환, 경계 보간 및 외삽 거부, 물리/원형 오차 계산.
실제 서버 NetCDF와 GPU를 사용한 전체 평가를 이 환경에서 실행하지는 않았습니다.

```bash
cd /home/jovyan/swan/swan_iclr_campaign_v2
python3 -m unittest -v test_campaign.py test_2021.py
```
