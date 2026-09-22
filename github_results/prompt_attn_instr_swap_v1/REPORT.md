## google_robot_open_drawer
- dedup scenes in manifest: 66; scenes with all five arms: 66
- success rate (Wilson 95% CI):
  - vanilla     20/ 66 = 0.303  CI [0.2055, 0.4222]  mean_m None  mean_steps 113.0
  - correct     31/ 66 = 0.4697  CI [0.3543, 0.5884]  mean_m 36.79  mean_steps 113.0
  - paraphrase  31/ 66 = 0.4697  CI [0.3543, 0.5884]  mean_m 36.32  mean_steps 113.0
  - swapped     36/ 66 = 0.5455  CI [0.4262, 0.6598]  mean_m 35.63  mean_steps 113.0
  - random      19/ 66 = 0.2879  CI [0.1927, 0.4064]  mean_m 35.95  mean_steps 113.0
- paired (scene-level), rescue/harm:
  - correct   vs swapped  : correct-win=6, correct-lose=11, net=-5, McNemar p=0.3323
  - correct   vs paraphrase: correct-win=11, correct-lose=11, net=+0, McNemar p=1.0
  - correct   vs random   : correct-win=17, correct-lose=5, net=+12, McNemar p=0.0169
  - correct   vs vanilla  : correct-win=17, correct-lose=6, net=+11, McNemar p=0.0347
  - paraphrase vs swapped  : paraphrase-win=9, paraphrase-lose=14, net=-5, McNemar p=0.4049
## google_robot_close_drawer
- dedup scenes in manifest: 66; scenes with all five arms: 66
- success rate (Wilson 95% CI):
  - vanilla     36/ 66 = 0.5455  CI [0.4262, 0.6598]  mean_m None  mean_steps 113.0
  - correct     51/ 66 = 0.7727  CI [0.6583, 0.8571]  mean_m 33.63  mean_steps 113.0
  - paraphrase  50/ 66 = 0.7576  CI [0.6419, 0.8449]  mean_m 34.46  mean_steps 113.0
  - swapped     50/ 66 = 0.7576  CI [0.6419, 0.8449]  mean_m 33.61  mean_steps 113.0
  - random      41/ 66 = 0.6212  CI [0.5006, 0.7285]  mean_m 33.01  mean_steps 113.0
- paired (scene-level), rescue/harm:
  - correct   vs swapped  : correct-win=6, correct-lose=5, net=+1, McNemar p=1.0
  - correct   vs paraphrase: correct-win=6, correct-lose=5, net=+1, McNemar p=1.0
  - correct   vs random   : correct-win=14, correct-lose=4, net=+10, McNemar p=0.0309
  - correct   vs vanilla  : correct-win=17, correct-lose=2, net=+15, McNemar p=0.0007
  - paraphrase vs swapped  : paraphrase-win=6, paraphrase-lose=6, net=+0, McNemar p=1.0
## google_robot_pick_coke_can
- dedup scenes in manifest: 100; scenes with all five arms: 100
- success rate (Wilson 95% CI):
  - vanilla     23/100 = 0.23  CI [0.1584, 0.3215]  mean_m None  mean_steps 80.0
  - correct     44/100 = 0.44  CI [0.3467, 0.5377]  mean_m 26.1  mean_steps 80.0
  - paraphrase  35/100 = 0.35  CI [0.2636, 0.4475]  mean_m 25.3  mean_steps 80.0
  - swapped     42/100 = 0.42  CI [0.328, 0.5179]  mean_m 24.99  mean_steps 80.0
  - random      34/100 = 0.34  CI [0.2546, 0.4372]  mean_m 25.32  mean_steps 80.0
- paired (scene-level), rescue/harm:
  - correct   vs swapped  : correct-win=17, correct-lose=15, net=+2, McNemar p=0.8601
  - correct   vs paraphrase: correct-win=18, correct-lose=9, net=+9, McNemar p=0.1221
  - correct   vs random   : correct-win=24, correct-lose=14, net=+10, McNemar p=0.1433
  - correct   vs vanilla  : correct-win=31, correct-lose=10, net=+21, McNemar p=0.0015
  - paraphrase vs swapped  : paraphrase-win=10, paraphrase-lose=17, net=-7, McNemar p=0.2478
## google_robot_move_near
- dedup scenes in manifest: 77; scenes with all five arms: 77
- success rate (Wilson 95% CI):
  - vanilla     46/ 77 = 0.5974  CI [0.4858, 0.6998]  mean_m None  mean_steps 80.0
  - correct     48/ 77 = 0.6234  CI [0.5117, 0.7233]  mean_m 42.12  mean_steps 80.0
  - paraphrase  44/ 77 = 0.5714  CI [0.4601, 0.676]  mean_m 40.5  mean_steps 80.0
  - swapped     48/ 77 = 0.6234  CI [0.5117, 0.7233]  mean_m 40.51  mean_steps 80.0
  - random      43/ 77 = 0.5584  CI [0.4474, 0.6639]  mean_m 40.06  mean_steps 80.0
- paired (scene-level), rescue/harm:
  - correct   vs swapped  : correct-win=12, correct-lose=12, net=+0, McNemar p=1.0
  - correct   vs paraphrase: correct-win=13, correct-lose=9, net=+4, McNemar p=0.5235
  - correct   vs random   : correct-win=13, correct-lose=8, net=+5, McNemar p=0.3833
  - correct   vs vanilla  : correct-win=12, correct-lose=10, net=+2, McNemar p=0.8318
  - paraphrase vs swapped  : paraphrase-win=9, paraphrase-lose=13, net=-4, McNemar p=0.5235

## Pooled (scene-level across the four tasks)
- valid scenes with all five arms: 309
  - vanilla     125/ 309 = 0.4045  CI [0.3513, 0.4601]  mean_m None  mean_steps 94.1
  - correct     174/ 309 = 0.5631  CI [0.5074, 0.6173]  mean_m 33.99  mean_steps 94.1
  - paraphrase  160/ 309 = 0.5178  CI [0.4622, 0.573]  mean_m 33.4  mean_steps 94.1
  - swapped     176/ 309 = 0.5696  CI [0.5139, 0.6236]  mean_m 32.97  mean_steps 94.1
  - random      137/ 309 = 0.4434  CI [0.389, 0.4991]  mean_m 32.91  mean_steps 94.1
- paired (scene-level), rescue/harm:
  - correct   vs swapped  : correct-win=41, correct-lose=43, net=-2, McNemar p=0.9132
  - correct   vs paraphrase: correct-win=48, correct-lose=34, net=+14, McNemar p=0.1507
  - correct   vs random   : correct-win=68, correct-lose=31, net=+37, McNemar p=0.0003
  - correct   vs vanilla  : correct-win=77, correct-lose=28, net=+49, McNemar p=0.0
  - paraphrase vs swapped  : paraphrase-win=34, paraphrase-lose=50, net=-16, McNemar p=0.1011

## Offline same-state branch checks
- google_robot_open_drawer: states=189, clean_equal=True, m_equal=True, mean_m=39.28042328042328
- google_robot_close_drawer: states=198, clean_equal=True, m_equal=True, mean_m=32.85858585858586
- google_robot_pick_coke_can: states=300, clean_equal=True, m_equal=True, mean_m=27.043333333333333
- google_robot_move_near: states=231, clean_equal=True, m_equal=True, mean_m=41.39393939393939
