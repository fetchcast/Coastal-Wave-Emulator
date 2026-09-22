# v4.1 GPU 배정 변경

기존 v4의 GPU 2·3을 기본 모델 학습에 추가합니다. 모델·학습률·시드·업데이트 수·평가 기준은 변경하지 않습니다. v5는 실행하지 않습니다.

- GPU 0·1: 기존 TNO B/C 완료 후 2021년 평가.
- GPU 2–7: 기존 기본 모델 21개 작업을 최대 6개 병렬로 실행.
- 결과 폴더는 기존 runs/iclr_parallel_v4 및 기존 C 폴더 그대로 사용합니다.
- 기존 v4 제어부에만 SIGTERM을 보내 정상 종료를 요청하고 자식 프로세스 종료를 확인합니다. 종료 확인 실패 시 새 작업을 시작하지 않습니다.
- 진행 중인 학습은 마지막 저장 체크포인트에서 재개합니다. 마지막 저장 이후의 업데이트는 반복될 수 있습니다. 중단 시점의 메모리 상태를 그대로 옮기는 기능은 아닙니다.
- 완료된 결과는 재사용합니다. 실행 중이던 평가는 재시작될 수 있습니다.
- 평가 시작은 original 학습 프로세스가 종료될 때까지 기다립니다. TNO가 거의 완료된 현재 상황에 맞는 배정입니다.
- 모든 GPU의 지속적인 사용을 보장하지 않습니다. 남은 작업 수나 GPU 메모리 조건에 따라 일부는 대기합니다.
- 예전 START.sh를 동시에 실행하지 마십시오. 상태 조회는 기존 v4 status.py도 계속 사용할 수 있습니다.

## 실행

ZIP을 /home/jovyan/swan 에 업로드한 뒤:

```bash
cd /home/jovyan/swan
unzip SWAN_Parallel_v41_GPU23.zip
nohup bash swan_parallel_v41/START.sh > iclr_parallel_v41.log 2>&1 &
tail -f iclr_parallel_v41.log
```

정상 인계 후 실제 학습 로그:

```bash
python3 /home/jovyan/swan/swan_parallel_v41/status.py
tail -n 40 /home/jovyan/swan/runs/iclr_parallel_v4/logs/baselines.log
nvidia-smi
```

복사된 기본 trainer는 없으며 서버의 기존 swan_repaired_v1을 그대로 사용합니다. 기존 v4 파일은 수정하지 않습니다. 새 제어부 출처는 controller_config_v41.json에 별도로 기록합니다. 기존 테스트와 GPU 배정/평가 순서 테스트를 CPU에서 실행했으며 실제 서버 GPU 인계는 현지 실행 시 확인해야 합니다.
