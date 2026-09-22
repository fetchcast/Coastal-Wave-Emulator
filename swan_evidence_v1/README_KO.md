# SWAN evidence analysis v1

기존 swan_bc_typhoon_v3 결과를 읽는 별도 패키지입니다. 학습·후보 선택·평가 기준·기존 결과 파일을 변경하지 않습니다. 새 출력 폴더가 이미 있으면 덮어쓰지 않고 종료합니다. Python 3.10+와 numpy, GPU 측정은 기존 SWAN 학습 환경이 필요합니다.

## 가능한 분석과 한계

| 항목 | 입력 / 실행 조건 | 출력과 해석 |
|---|---|---|
| 통계 비교 | v3의 최종 9개 모델에 대한 2021년 평가 완료 | 시드 평균·표준편차, 쌍체 부호 무작위화 검정 및 전체 검정에 Holm 보정. 탐색적 결과 |
| 태풍 | v3의 고정 JMA 사건 정의와 hourly.csv | 사건별·연간·TC 합집합·태풍 일생 기준 합집합·사건 외 구간. Hs MAE/RMSE, 참값 3/5m 이상, 피크 진폭·시각·참 피크 위치 오차 |
| 시간 의존성 | 동일 9개 hourly.csv | 연간 Hs MAE 차이의 24/72/168시간 이동블록 bootstrap 구간. 학습된 모델과 해당 연도에 조건부인 민감도 분석 |
| 다른 해역 | 별도 해역 실제 평가 결과와 명세 | 현재 자료에는 없으므로 NOT_EVALUATED. 제공된 외부 결과 집계 도구만 포함 |
| 추론 속도 | 준비된 v3 signature/cache/checkpoint, 빈 GPU | FP32 batch=1 실제 forward와 캐시 입력→CPU 출력 지연시간 |
| SWAN 가속률 | 위 측정 + 동일 물리 작업의 SWAN solver-core 실행시간 | 중앙 지연시간에서 추정한 제한적 core/cached 비율. 전체 파이프라인 가속률은 아님 |

이 패키지는 새 해역 데이터 전처리/모델 이식이나 SWAN 실행을 대신하지 않습니다. 그 부분은 새 격자, 좌표·수심·경계 입력 의미, 강제 자료, 정답과 기존 모델의 호환성을 확인한 뒤 구현해야 합니다. 2021년은 원래 해역의 미학습 연도 평가이며 공간 일반화가 아닙니다.

## 1. 설치와 CPU 분석

서버 `/home/jovyan/swan`에 ZIP을 올리고:

```bash
cd /home/jovyan/swan
unzip SWAN_Evidence_Analysis_v1.zip
python3 swan_evidence_v1/analyze.py \
  --eval-root /home/jovyan/swan/runs/iclr_typhoon_2021_v3 \
  --output /home/jovyan/swan/runs/evidence_analysis_v1
```

평가가 아직 진행 중이면 WAITING 상태 파일만 작성합니다. 완료 뒤 **새 출력 폴더**로 다시 실행하세요(예: `evidence_analysis_v1_final`). 이 도구는 평가 완료를 자동 대기하거나 GPU 평가를 시작하지 않습니다. 기존 v3가 학습 및 평가를 계속 수행합니다. v3의 `analysis_completed.json`이 필요합니다. 데이터 9개 모두 완전한 8748시간인지, 시드·정답·선택 config가 일치하는지 확인합니다. 불일치하면 조용히 제외하지 않고 오류로 종료합니다.

출력:
- `typhoon_and_annual_summary.csv`: 모델별 평균/시드 표준편차.
- `typhoon_and_annual_by_seed.csv`: 원래 사건별 상세 결과(피크 지표 포함).
- `seed_comparisons.csv`: 모델 A-B 차이, 시드 변동, 탐색적 p값. 음수는 A 오차가 작음.
- `temporal_block_sensitivity.csv`: 연간 MAE 차이의 조건부 bootstrap 95% 구간.
- `STATUS.json`, `provenance.json`: 제약, 입력 SHA256와 코드 기록.

통계 해석:
1. 3개 시드의 양측 정확 부호 무작위화 검정은 8개 경우만 있으며 최소 p값은 0.25입니다. 유의성을 확보하려고 시간/격자를 독립 표본으로 늘리지 않습니다.
2. 동일 시드 번호는 비교의 짝을 정의할 뿐, 모형 사이 교환가능성을 보장하지 않습니다. seed 42는 후보 선택에도 쓰였으므로 모두 탐색적 통계입니다. seed 43/44만 쓰면 반복 수가 더 적습니다.
3. Holm 보정은 출력의 모든 scope×metric×model pair에 적용합니다. 검정 수가 많아 보수적입니다. 비유의는 동등함을 뜻하지 않습니다.
4. 블록 bootstrap은 시드별 손실 차이를 평균한 시간열을 재표집합니다. 예측 평균 앙상블이 아니고, 시드 모집단이나 다른 연도에 대한 CI도 아닙니다. 계절성·긴 자기상관 때문에 명목 95% 보장을 주장하지 마세요. 24/72/168시간에서 결과가 얼마나 달라지는지 함께 보고합니다.
5. 사건 수가 적거나 사건 창이 겹칠 수 있으므로 사건×시드×시간을 독립 반복으로 취급하지 않습니다. 본 버전은 사건별 seed SD를 제공하며 독립 사건 모집단 CI를 만들지 않습니다.
6. `outside_TC_windows`는 잔잔한 날과 동의어가 아닙니다. `typhoon_lifetime_union`은 태풍 강도에 도달했던 사건의 전체 정의 창이며 매 시각 태풍 강도였다는 뜻이 아닙니다.
7. 피크 시각은 도메인 최대 Hs 기준이라 최대 위치가 이동할 수 있습니다. 관측 부이별 피크 검증과 구분하세요. SWAN 정답 상대 오차는 실해양 관측 상대 오차가 아닙니다.

## 2. 추론 속도: 실제로 빈 GPU에서 순서대로 실행

아래 예시는 물리 GPU 0을 사용합니다. GPU 0이 학습 중이면 실행을 거부합니다. 학습을 종료하지 말고 빈 GPU 번호로 바꾸거나 학습 완료 뒤 실행하세요. 각 모델의 v3 `signature.json`은 해당 모델의 2021 평가가 시작되면 생깁니다. 이 파일·캐시가 아직 없으면 평가 준비를 기다립니다.

```bash
cd /home/jovyan/swan
for model in fno ffno tno; do
  python3 swan_evidence_v1/benchmark_inference.py \
    --signature "/home/jovyan/swan/runs/iclr_typhoon_2021_v3/models/${model}_s42/signature.json" \
    --gpu 0 \
    --output "/home/jovyan/swan/runs/inference_evidence_v1/${model}.json" || break
done
```

동일 GPU에서 순차 측정합니다. 각 모델은 별도 프로세스라 GPU 메모리가 해제됩니다. 8개 연중 시점의 실제 입력, warmup 10회, 반복 40회, batch=1, FP32, TF32 off, CUDA 동기화를 사용합니다. CPU 입력 배열 8개를 먼저 올려두므로 추가 호스트 메모리가 필요합니다. seed42 체크포인트만으로 실행시간을 비교하며, 정확도 통계의 3개 시드와 구분합니다. 이 프로파일링은 학습/평가 정밀도를 변경하지 않습니다.

- `forward_gpu`: GPU 이벤트 시간, 전송 제외.
- `cached_host_to_host`: CPU 배열 복사+H2D+forward+D2H 포함.
- 둘 다 데이터 파일 읽기, 기상/경계 전처리, 물리 단위 복원, 출력 저장 제외.
- 모델 로딩/초기 CUDA 전송 시간은 `startup_seconds`로 별도 기록.
- 중앙값, p10/p90, 원시 반복 시간, GPU/torch/CUDA, 피크 할당 메모리와 체크포인트 해시 저장.
- GPU 초기 점검은 한 시점의 점검입니다. 측정 도중 다른 작업을 시작하지 마세요.

모델 간 지연시간 비교:

```bash
python3 swan_evidence_v1/compare_speed.py \
  /home/jovyan/swan/runs/inference_evidence_v1/fno.json \
  /home/jovyan/swan/runs/inference_evidence_v1/ffno.json \
  /home/jovyan/swan/runs/inference_evidence_v1/tno.json \
  --output /home/jovyan/swan/runs/inference_evidence_v1/comparison.csv
```

SWAN과 비교하려면 `swan_baseline.template.json`을 복사하여 **실측** wall_seconds, 동일 작업의 output_frames, 하드웨어/명령/물리 작업 명세를 입력합니다. `emulator_input_shape`는 측정 JSON의 input_shape와 같아야 합니다. 동일 도메인·입력·출력 계약을 직접 확인했을 때만 `same_domain_forcing_output_contract`를 true로 설정하세요. SWAN 내부 적분 스텝 수가 아니라 동일한 물리 출력 프레임 수를 사용합니다. 병렬 SWAN이면 CPU 코어 수/MPI ranks를 반드시 기재합니다.

위 compare 명령에 `--swan /absolute/path/swan_baseline.json`을 추가하면 `SWAN core wall seconds / (output_frames × cached surrogate median seconds)`를 계산합니다. **추정 core/cached 비율**이며 전체 배포 가속률로 쓰면 안 됩니다. 전체 가속률은 양쪽에서 같은 기간·출력을 실제 끝까지 실행한 동일 범위의 wall time이 추가로 필요합니다. 훈련 시간은 어느 추론 가속률에도 들어가지 않습니다.

## 3. 다른 해역: 자료가 확보된 경우에만

`regions.template.json`의 regions는 비어 있고 기본적으로 실행이 거부됩니다. 새 해역 평가가 실제 완료되면 `region_example.schema.json` 구조를 regions 배열에 넣습니다. `metrics_csv`에는 다음 열이 필요합니다:

```text
model,seed,scope,hs_mae,checkpoint_sha256
```

각 scope에 FNO/FFNO/TNO×seed42/43/44가 있어야 합니다. Hs MAE는 m 단위, 체크포인트는 기존 고정 모델 그대로여야 합니다. 체크포인트 해시는 manifest의 모델/시드별 해시와 일치해야 합니다. 데이터 제외·좌표 재표본화·경계 입력 계약·훈련 정규화 고정은 사용자가 실제 검증해 기록하는 명세이며 이 집계기가 자료 누수를 독립적으로 증명하지 않습니다. 새 해역에서 미세조정했다면 zero-shot 일반화로 보고하지 않습니다.

```bash
python3 swan_evidence_v1/geography.py \
  --manifest /absolute/path/regions.json \
  --output /home/jovyan/swan/runs/geographic_evidence_v1
```

여러 해역에 대한 결과가 생겨도 모든 해역으로 일반화된다고 주장할 수는 없습니다.

## 검증과 참고

```bash
python3 -m unittest discover -s swan_evidence_v1 -p 'test_*.py' -v
```

CPU 통계 핵심 테스트 및 합성 9개 연간 자료 통합 실행을 검증했습니다. 실제 서버 B200의 GPU 벤치마크 및 실제 2021 자료 실행은 여기서 수행하지 않았습니다.

- Paired randomization / exchangeability: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
- CUDA asynchronous execution and timing: https://docs.pytorch.org/docs/stable/notes/cuda.html
