# 자원·강건성 진단 (소스 버전 v2.1.1)

정식 설명은 [README.md](README.md)입니다.

기본 실행은 재학습하지 않습니다. 기존 2021년 평가가 완료된 모델을 대상으로
사건 예측 재사용/복원 → 추론 메모리·속도 → 입력 교란 평가 순서로 실행합니다.
사용 중인 GPU는 건너뛰며, 다른 학습을 종료하지 않습니다.

```bash
cd /home/jovyan/swan
unzip SWAN_Resource_Checks_2_1_1.zip
python3 swan_resource_checks/run.py inspect
nohup bash swan_resource_checks/START.sh --seeds 42 \
  > /home/jovyan/swan/resource_checks_2_1_1.log 2>&1 &
python3 swan_resource_checks/run.py status
```

먼저 seed 42로 시작하는 명령입니다. 통계적 우월성 판정용 결과가 아닙니다.
강건성 기본 표본은 사건 기간 6시간 간격입니다. 전체 시간은 `--stride 1`과
새 `--output` 경로를 지정하십시오. 기존 사건 진단 자체는 모든 시간을 사용합니다.
12시간은 새 작업 배정 한도이며, 시작한 작업은 완료까지 계속됩니다.

추가 학습은 새 출력 경로로 첫 실행할 때 `--lr-pilots`를 지정해야 합니다.
기본 아키텍처별 학습률 3개를 같은 짧은 스케줄로 비교하는 탐색이며,
장기 학습 성능이나 공정한 전체 탐색을 대신하지 않습니다.

측정 메모리는 **추론 최대 할당량/예약량 GiB**입니다. 실제 학습 메모리는
이번 측정에 포함되지 않습니다. 수정 입력에 대한 정답은 원래 SWAN 결과이므로
강건성 결과를 물리 응답 정확성으로 해석하지 마십시오. 2022년은 건드리지 않습니다.
