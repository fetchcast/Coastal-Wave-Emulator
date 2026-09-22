# SWAN ICLR 확대 실험 v1

기존 swan_repaired_v1 학습 코드와 완료된 pilot 9개를 사용하는 별도 자동 실행 패키지입니다.
기존 학습 코드와 결과를 덮어쓰지 않습니다. Python 주석과 docstring은 영어입니다.

## 실행

ZIP을 /home/jovyan/swan 에 올린 후:

```bash
cd /home/jovyan/swan
unzip -q SWAN_ICLR_Campaign_v1.zip
nohup bash swan_iclr_campaign_v1/START.sh > iclr_campaign_v1.log 2>&1 &
```

GPU 0–7 중 비어 있는 GPU에 한 작업씩 배정합니다. 다른 프로세스가 사용 중인 GPU는 기다립니다.
같은 GPU를 대상으로 다른 런처를 동시에 시작하지 마세요. 이 패키지 자체의 중복 실행은 잠금으로 막습니다.

```bash
python3 /home/jovyan/swan/swan_iclr_campaign_v1/status.py --watch 10
```

조회만 종료하려면 Ctrl+C. 한 번만 조회하려면 --watch 10 생략.

```bash
tail -n 50 /home/jovyan/swan/iclr_campaign_v1.log
```

계획만 확인하려면:

```bash
bash /home/jovyan/swan/swan_iclr_campaign_v1/START.sh --plan
```

--plan은 모델을 학습하지 않으며 서버 자료 지문 검사도 하지 않습니다. 실제 실행 시 검사합니다.
중단 후에는 최초 nohup 명령을 다시 실행합니다. 기존 런처가 종료됐는지 먼저 확인하세요.
완료된 결과는 검증 후 건너뛰며, 미완료 작업은 원래 worker의 저장된 cycle 체크포인트에서 재개합니다.
마지막 저장 이후 업데이트는 다시 수행할 수 있습니다.

## 필요한 기존 파일

- /home/jovyan/swan/swan_repaired_v1/{run_repaired.py,train_repaired.py,repair_support.py,legacy_repaired.py}
- /home/jovyan/swan/runs/repaired_timegap_v1/plan_pilot.json 및 protocol.json
- /home/jovyan/swan/runs/repaired_timegap_extra_v1/extra_plan.json
- 기존 9개 실행의 run_summary.json 및 best checkpoint
- 기존 NetCDF와 경계 입력 등 원래 학습에 사용한 자료

첨부로 검토한 네 학습 코드의 SHA-256을 expected_hashes.json에 기록했습니다.
서버 코드/자료가 pilot과 다르면 실행을 중단합니다. 해시를 수동 변경해서 통과시키지 마세요.
이 패키지는 기존 Python 환경을 사용하며 별도 pip 설치는 하지 않습니다.

## 자동 단계

1. preflight: A 후보 29개 각각 성공 업데이트 2회 및 전체 학습/검증 평가, 최종 평가를 실행합니다.
   원래 pilot과 같은 stage를 사용하므로 축소된 smoke 평가가 아닙니다. 전체 자료를 읽어 비용이 발생합니다.
   사전 확인 결과의 성능은 후보 선택과 results.csv에서 제외합니다. 통과해도 본 학습 OOM 가능성은 남습니다.
2. A: 구조 탐색 29개, seed 42, 30 full-data-equivalent cycles.
   FNO/FFNO: 폭 64/128/256 × 깊이 4/6/8 × 모드24에서 기존64/4/24 제외(각8개).
   FNO/FFNO: 폭64/128/256, 깊이4, 모드48 추가(각3개).
   TNO: 위 모드24 격자에서 기존 기준과 폭256 깊이6/8 제외(6개), 폭64 깊이4 모드48 추가(1개).
   TNO 시간 모드는 4 고정. 신규 TNO 구조 후보는 activation checkpointing 사용.
3. B: 모델별 검증 상위 2개 구성에 max_lr=5e-5와2e-4 적용(최대12개).
   기존 max_lr=1e-4 결과도 최종 후보에 포함합니다.
4. C: A/B 및 기존 기준의 seed42 결과에서 모델별 검증 최상위 선택 후 seed43/44(최대6개).
   기존 기준이 선택되면 이미 완료된 seed43/44 재사용.
5. D: 선택 구성에서 seed42, 60 cycles(3개). 새로 초기화하는 별도 학습입니다.
   OneCycleLR 전체 길이 및 검증 기회도 바뀌므로 순수한 추가 업데이트 효과로 해석하지 마세요.
6. E: 선택 구성에서 train_fraction=.25/.5 × seed42/43/44(18개), 30 full-data-equivalent cycles.
   원래 학습 코드의 중첩 부분집합, 업데이트 기반 검증을 사용합니다.
   seed 반복은 모델 학습 seed 반복이며 독립적인 부분집합 추출 seed 3개가 아닙니다.

추가 본 학습 최대68개 + 두 업데이트 사전 확인29개입니다. 기존 pilot9개는 다시 학습하지 않습니다.
전체 조합 전수탐색이 아닙니다. 모델별 탐색 수는 FNO11/FFNO11/TNO7로 다르므로 동일 탐색 비용이라고 주장하지 마세요.
실제 시간이 길어질 수 있습니다. 학회 마감에 맞춘 자동 종료 기능은 없으며 모든 단계가 끝날 때까지 계속됩니다.

## 조건과 결과 해석

- 선택은 repair_audit.best_val_hs_mae만 사용. 시험 RMSE는 정렬/선택에 사용하지 않음.
- 기존 worker가 각 실행의 최종 시험 평가도 수행하지만 캠페인 자동 선택에서는 읽지 않음.
- 신규 구조 작업은 nominal effective batch=4를 유지. microbatch와 accumulation은 기록된 설정 참조.
- 누적 끝의 잔여 묶음은 작을 수 있으므로 모든 실제 업데이트가 정확히4개 표본이라고 주장하지 않음.
- 기존 pilot 설정이 선택되면 후속 작업에서 해당 microbatch/checkpointing 설정을 그대로 유지.
- 자원 한계로 빠진 후보도 기록. 동일 파라미터 수/동일 GPU시간 비교가 아님.
- TNO 폭256 깊이8은 후보에서 제외. GPU8개 메모리가 하나로 합쳐지는 방식이 아님.
- 이 패키지는 새 분할 실험, 7년 자료 확장, 별도의 논문용 그림 생성 코드를 추가하지 않음.

## 실패 처리

명시적인 CUDA OOM은 *.resource_skip.json에 기록하고 다른 작업을 계속 실행합니다.
메모리 하한이 GPU예산85%를 넘으면 실행 전에 같은 방식으로 제외합니다.
자동으로 폭/깊이를 낮추거나 다른 batch 설정으로 결과를 대체하지 않습니다.
알 수 없는 오류/자료 불일치/분할 불일치는 전체 캠페인을 중단합니다.
완료된 작업은 유지되며 오류 원인 해결 후 재시작 가능합니다.
OOM 제외 기록은 재시작 때 재사용됩니다. 설정을 변경하려면 별도 결과 루트를 사용하세요.
campaign_completed.json은 캠페인 종료를 뜻하며 모든 후보가 성공했다는 뜻은 아닙니다.

## 결과 위치

/home/jovyan/swan/runs/iclr_expanded_v1

- status.json: 현재 단계, 활성 GPU, 대기 수. 중단된 경우 마지막 기록이므로 시간과 로그를 함께 확인.
- results.csv: 기존9개와 완료된 본 학습의 성능/파라미터/시간/체크포인트 경로.
- selection.json: 검증 기준 선택 결과와 실제 job.
- preflight,A,B,C,D,E: 각 단계 계획, 원래 worker 출력, 자원 제외 기록.
- campaign_completed.json: 모든 단계 처리 종료.

고급 경로 변경은 START.sh --server-root ... --package ... --data ... --root ... --gpus ... 사용.
원래 pilot 루트는 server-root 아래 고정된 두 경로를 사용합니다.

## 검증 범위

CPU 모의 테스트로 29개 구성과 ID, 원본 설정 보존, 시험 지표 비사용 선택,
계획 변경 거부, OOM 후 다른 작업 실행, 재시작 완료/제외 재사용을 확인했습니다.
첨부 코드 해시가 pilot 계획의 해시와 일치했습니다.
실제 CUDA 학습, 메모리 사용, 8GPU 동시 실행은 이 환경에서 수행하지 않았습니다.

```bash
cd /home/jovyan/swan/swan_iclr_campaign_v1
python3 -m unittest -v test_campaign.py
```
