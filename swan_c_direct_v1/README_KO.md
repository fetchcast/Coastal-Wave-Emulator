# SWAN A → C 실행 전환

기존 v2의 기본 경로 및 인수 없이 시작한 run_v2.py를 대상으로 합니다.
훈련 코드는 변경하지 않습니다. A와 기존 파일럿 seed42의 검증 Hs MAE로
계열별 최우수 설정을 정하고, seed43/44를 새 결과 폴더에서 학습합니다.
B/D/E는 생략합니다. TNO를 작은 설정으로 강제 변경하지 않습니다.

## 설치와 점검

ZIP을 /home/jovyan/swan에 놓고 실행합니다.

```bash
cd /home/jovyan/swan
unzip SWAN_C_Direct_v1.zip
python3 swan_c_direct_v1/run_c_direct.py
```

위 명령은 점검만 수행합니다. 학습 코드, 기존 실행기 코드, 프로토콜,
기존 파일럿, A 계획을 검사합니다. 지원하지 않는 설정이면 중단합니다.

## 적용

```bash
cd /home/jovyan/swan
nohup python3 -u swan_c_direct_v1/run_c_direct.py --apply > iclr_c_direct_v1.log 2>&1 &
```

이 명령은 기존 v2 상위 실행기에 SIGTERM을 보내 정상 종료 절차를 요청합니다.
하위 학습 프로세스의 종료를 확인하고 기존 campaign.lock과 v2.lock을 확보한 뒤
남은 A를 기존 결과 폴더에서 재개합니다. 최신 체크포인트 저장 이후의 업데이트는
다시 수행할 수 있습니다. 학습이 한 순간도 중단되지 않는 방식은 아닙니다.
기존 체크포인트와 결과는 삭제하거나 덮어 초기화하지 않습니다.

프로세스가 60초 내 종료되지 않으면 새 학습은 시작하지 않습니다.
로그를 확인한 후 같은 적용 명령으로 재시도할 수 있습니다.
다른 경로에서 실행한 별도 GPU 작업까지 제어하지 않습니다.
알 수 없는 작업이나 사용자 지정 기존 런처는 자동 종료하지 않습니다.

새 실행기가 원래 GPU 번호를 보장하지는 않습니다. 장비가 동일한 경우
남은 A를 사용 가능한 GPU에 배정하고 FNO/FFNO의 C를 함께 시작합니다.
TNO는 해당 계열 A가 모두 완료되어야 C를 시작합니다.
기존 기준 설정이 선정되면 이미 완료된 seed43/44를 재사용합니다.

B가 이미 실행 중이었다면 전환 때 종료되며 그 결과는 보존하지만 이번 선정에는
사용하지 않습니다. 모든 계열에서 pilot42+A만 사용한다는 규칙을 적용합니다.

## 상태 확인

```bash
python3 /home/jovyan/swan/swan_c_direct_v1/status.py --watch 10
tail -n 40 /home/jovyan/swan/iclr_c_direct_v1.log
```

상태는 runs/iclr_c_direct_v1/status.json에서 읽습니다.
기존 status_v2.py의 상태는 전환 이후 갱신되지 않으므로 사용하지 않습니다.
C_completed는 기존 파일럿을 재사용한 반복도 포함합니다.

## 결과와 다음 단계

- 기존 A: runs/iclr_expanded_v1/A
- 새 C: runs/iclr_c_direct_v1/C
- 계열별 고정 선정: selection_fno.json, selection_ffno.json, selection_tno.json
- 9개 선정 결과: selected_9.json, results.csv
- 완료 기록: completed.json

이 패키지는 실행 제어만 변경합니다. 2021년 평가와 태풍 사건 정의는 실행하지
않습니다. 2021년 평가에는 새 선정 기록 및 C 경로를 읽는 후속 연결이 필요합니다.
기존 START.sh나 run_v2.py를 다시 실행하면 원래 B/D/E 계획으로 돌아갈 수 있으므로
전환 뒤에는 기존 런처를 실행하지 마세요. 새 제어부는 같은 명령으로 재시작합니다.

최우수 설정은 단일 탐색 seed로 고르고 고정합니다. 추가 seed 성능으로 설정을
다시 고르지 않습니다. 동일 계산 예산 벤치마크라고 주장할 수 없으며,
실제 탐색 시간과 범위를 보고해야 합니다.

## 검증 범위

CPU 모의 시험 4개로 계열별 완료 대기, seed 변경 시 설정 보존, A/C 동시 배정,
선정 기록 변경 거부를 검사했습니다. 실제 서버의 GPU 및 프로세스 인계는
이 환경에서 실행하지 못했습니다. 원본 모델 구현과 데이터 처리 코드는 그대로입니다.
