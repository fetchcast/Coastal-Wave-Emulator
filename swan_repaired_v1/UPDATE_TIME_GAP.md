# 시간 공백 수정

제공된 시간축에는 2019-12-31 00시에서 2020-01-01 00시 사이 23개 시간 기록이 없습니다.
시간순서가 올바르고 간격이 정수 시간인 자료를 허용하되, 입력 시작부터 목표 시점까지 공백을
가로지르는 시퀀스는 모든 분할과 방향 보정 후보에서 제외합니다. 30분 간격, 중복, 역순, NaT는 거부합니다.
원본 NC, 모델, 손실, 학습 예산은 변경하지 않습니다. 블록은 기존의 168개 저장 표본 단위를 유지하며,
공백을 포함한 블록을 달력상 정확한 168시간이라고 해석하면 안 됩니다.

제공된 공백은 기존 경계 제외 구간에 해당하므로, 재현 검사에서 분할 인덱스가 모두 유지됐습니다.
전체 분율은 9770/1980/1980, 50%는 5150/1980/1980, 25%는 2640/1980/1980입니다.
공백을 블록 내부에 둔 별도 검사에서는 해당 시퀀스만 제외됐습니다. ns/us 단위, 비정상 시간축,
기존 CPU 검사 10개도 확인했습니다. 실제 서버 자료의 나머지 전처리와 GPU 학습은 아직 미검증입니다.

ZIP은 기존 swan_repaired_v1 폴더 위에 압축 해제합니다. 코드 지문이 달라졌으므로 이전 실패 결과는
보존하고 새 결과 폴더를 사용합니다. 이전 protocol.json을 편집하거나 검사 기능을 끄지 마세요.

```bash
cd /home/jovyan/swan
unzip -o -q SWAN_TimeGap_Update_v1.zip
nohup bash swan_repaired_v1/START_REPAIRED.sh --gpus 0,1,2 --root /home/jovyan/swan/runs/repaired_timegap_v1 > repaired_timegap_run.log 2>&1 &
tail -f repaired_timegap_run.log
```

새 결과 상태 확인 명령

```bash
python3 swan_repaired_v1/run_repaired.py --root /home/jovyan/swan/runs/repaired_timegap_v1 --report
```
