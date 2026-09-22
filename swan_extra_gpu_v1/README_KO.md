# GPU 3–7 추가 실행 및 진행 조회

기존 GPU 0–2의 학습은 종료하거나 다시 시작하지 마세요.
이 패키지는 기존 swan_repaired_v1의 Python 파일을 덮어쓰지 않습니다.
SWAN_TimeGap_Update_v1 적용 후 세 smoke가 성공하고 pilot 계획이 생성된 상태에서 사용합니다.

## 실행

ZIP을 /home/jovyan/swan에 올리고 다음을 실행합니다.

```bash
cd /home/jovyan/swan
unzip -q SWAN_Extra_GPU_v1.zip
nohup bash swan_extra_gpu_v1/START_EXTRA_GPUS.sh > extra_gpu_run.log 2>&1 &
```

시작 로그만 보려면 다음을 실행합니다.

```bash
tail -f /home/jovyan/swan/extra_gpu_run.log
```

전체 진행률을 10초마다 확인하려면 다음을 실행합니다.

```bash
python3 /home/jovyan/swan/swan_extra_gpu_v1/status_repaired.py --watch 10
```

Ctrl+C는 조회만 종료하며 학습을 종료하지 않습니다. 한 번만 조회하려면 --watch 10을 생략합니다.
조회 코드는 각 로그의 끝부분만 읽으며 큰 학습 로그 전체를 메모리에 올리지 않습니다.
Successful updates의 진행률과 마지막 검증 주기의 best Hs MAE를 보여줍니다.
표시된 ETA는 향후 검증·최종 평가 비용을 충분히 반영하지 않을 수 있습니다.
NOT COMPLETED는 실패 판정이 아닙니다. 대기·학습·평가 중인 작업도 여기에 해당합니다.

## 추가 작업

| 최초 배정 GPU | 모델 | Seed |
|---|---|---|
| 3 | TNO | 43 |
| 4 | FNO | 43 |
| 5 | FFNO | 43 |
| 6 | TNO | 44 |
| 7 | FFNO | 44 |
| 선택 GPU 중 먼저 비는 한 장 | FNO | 44 |

위 배정은 GPU 3–7이 모두 비어 있을 때의 순서입니다. 점유된 GPU는 기다립니다.
새 작업은 총 6개이며 최대 5개가 동시에 실행됩니다. 기존 seed 42 세 작업을 합하면 총 9개입니다.
모든 작업은 기존 pilot 계획을 복사하므로 폭 64, 깊이 4, 공간 모드 24와 학습량을 유지합니다.
변경되는 항목은 seed와 이를 표시하는 config ID뿐입니다. 자료 분할 seed는 그대로입니다.
실행 전 계획만 보려면 START_EXTRA_GPUS.sh 뒤에 --plan을 붙이세요.

## 결과와 재시작

기존 결과는 /home/jovyan/swan/runs/repaired_timegap_v1에 유지됩니다.
추가 결과는 /home/jovyan/swan/runs/repaired_timegap_extra_v1에 저장됩니다.
두 결과 폴더를 모두 status_repaired.py가 조회합니다.

동일 명령으로 다시 실행하면 완료 결과는 검증 후 건너뛰고, 미완료 작업은 기존 학습 코드의
체크포인트 규칙에 따라 재개합니다. 실행 중인 동일 추가 런처가 있으면 잠금으로 차단합니다.
기존 코드나 자료 지문이 달라지면 실행하지 않습니다. protocol.json을 수동으로 바꾸지 마세요.
추가 작업 실패 시 추가 런처가 소유한 작업만 종료하며 기존 GPU 0–2의 런처는 건드리지 않습니다.

공통 자료 검사 성공 기록은 동일한 자료 지문을 확인한 뒤 재사용합니다.
각 학습 작업의 자료 로딩·전처리는 여전히 필요하므로 시작 직후에는 GPU 사용률이 낮을 수 있습니다.
동시 작업 증가로 CPU·메모리 대역폭·저장장치 경쟁이 커질 수 있어 실행당 속도가 같다고 보장하지 않습니다.
이 패키지는 2021년 평가나 대형 모델 탐색을 추가하지 않습니다.

## 검증

모의 실행에서 seed만 변경되는지, 기존 계획이 변경되지 않는지, 실패한 smoke를 거부하는지,
검사 캐시 재사용, GPU 0–2 지정 거부, 전체 실행 경로와 진행 로그 파싱을 확인했습니다.
실제 GPU 5개 동시 실행은 여기서 수행하지 않았으며, 서버에서 통과한 학습 코드를 그대로 호출합니다.
