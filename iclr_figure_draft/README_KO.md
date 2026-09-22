# ICLR 비용·검증 성능 그림 초안

cost_accuracy_A.pdf, .svg, .png는 동일한 그림입니다.
29개 A 설정과 3개 파일럿의 seed 42만 사용했습니다. B와 C는 포함하지 않았습니다.
별표는 각 모델 계열의 최저 검증 Hs MAE 설정입니다. 이는 최종 선택 표시가 아닙니다.

재생성:
python3 plot_cost_accuracy.py results_A_source.csv --output .

영문 캡션 초안:
Validation accuracy across the architecture search. Each point represents one configuration trained with seed 42 for 76,950 successful optimizer updates. The panels relate the best validation Hs MAE to parameter count (a) and recorded training wall time (b). Stars mark the lowest validation MAE for each model family among the 29 stage-A configurations and three pilot anchors. Stage-B learning-rate trials and stage-C seed repetitions are not included. Training wall times reflect the recorded runs and are not controlled measurements of inference latency.

해석 범위:
- 같은 업데이트 수가 같은 계산량을 뜻하지 않습니다.
- 1시드 탐색 결과로 통계적 우월성을 주장할 수 없습니다.
- 파라미터 수와 기록된 학습시간은 다른 비용 척도입니다.
- 현재 그림으로 태풍·2021년 성능 또는 추론 가속률을 주장하지 않습니다.
- 제공된 CSV에는 B 결과가 없어 FFNO의 새 최종 선택 lr=5e-5 결과가 반영되지 않았습니다.
