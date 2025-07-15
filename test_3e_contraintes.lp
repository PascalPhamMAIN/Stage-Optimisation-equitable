\ Model affectation couple
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xd5db99d40fb91aed
Minimize
  l[0] + l[1] + l[2] + l[3] + l[4]
Subject To
 R0: x_affect[0,2] + x_affect[0,4] >= 1
 R1: x_affect[0,2] + x_affect[0,4] <= 1
 R2: x_affect[1,2] + x_affect[1,3] >= 1
 R3: x_affect[1,2] + x_affect[1,3] <= 1
 R4: x_affect[2,0] + x_affect[2,4] >= 1
 R5: x_affect[2,0] + x_affect[2,4] <= 1
 R6: x_affect[3,1] + x_affect[3,3] >= 1
 R7: x_affect[3,1] + x_affect[3,3] <= 1
 R8: x_affect[4,0] + x_affect[4,1] >= 1
 R9: x_affect[4,0] + x_affect[4,1] <= 1
 R10: x_affect[2,0] + x_affect[4,0] >= 1
 R11: x_affect[2,0] + x_affect[4,0] <= 1
 R12: x_affect[3,1] + x_affect[4,1] >= 1
 R13: x_affect[3,1] + x_affect[4,1] <= 1
 R14: x_affect[0,2] + x_affect[1,2] >= 1
 R15: x_affect[0,2] + x_affect[1,2] <= 1
 R16: x_affect[1,3] + x_affect[3,3] >= 1
 R17: x_affect[1,3] + x_affect[3,3] <= 1
 R18: x_affect[0,4] + x_affect[2,4] >= 1
 R19: x_affect[0,4] + x_affect[2,4] <= 1
 sync_0_4: x_affect[0,4] - x_obj[0,4] = 0
 sync_1_2: x_affect[1,2] - x_obj[1,2] = 0
 sync_2_0: x_affect[2,0] - x_obj[2,0] = 0
 sync_3_3: x_affect[3,3] - x_obj[3,3] = 0
 sync_4_1: x_affect[4,1] - x_obj[4,1] = 0
 z_0: - 722 x_obj[0,2] - 237 x_obj[0,4] + z[0] = 0
 z_1: - 291 x_obj[1,2] - 391 x_obj[1,3] + z[1] = 0
 z_2: - 598 x_obj[2,0] - 82 x_obj[2,4] + z[2] = 0
 z_3: - 296 x_obj[3,1] - 418 x_obj[3,3] + z[3] = 0
 z_4: - 276 x_obj[4,0] - 913 x_obj[4,1] + z[4] = 0
 l_sorted_0: - z[0] + l[0] >= 0
 l_sorted_1: - z[0] - z[1] + l[1] >= 0
 l_sorted_2: - z[0] - z[1] - z[3] + l[2] >= 0
 l_sorted_3: - z[0] - z[1] - z[3] - z[4] + l[3] >= 0
 l_sorted_4: - z[0] - z[1] - z[2] - z[3] - z[4] + l[4] >= 0
 l_sorted_0: - z[4] + l[0] >= 0
 l_sorted_1: - z[2] - z[4] + l[1] >= 0
 l_sorted_2: - z[2] - z[3] - z[4] + l[2] >= 0
 l_sorted_3: - z[1] - z[2] - z[3] - z[4] + l[3] >= 0
 l_sorted_1: - z[0] - z[4] + l[1] >= 0
 l_sorted_2: - z[0] - z[1] - z[4] + l[2] >= 0
 l_sorted_0: - z[1] + l[0] >= 0
Bounds
 x_affect[0,2] = 1
 x_affect[0,4] = 0
 x_affect[1,2] = 0
 x_affect[1,3] = 1
 x_affect[2,0] = 0
 x_affect[2,4] = 1
 x_affect[3,1] = 1
 x_affect[3,3] = 0
 x_affect[4,0] = 1
 x_affect[4,1] = 0
 x_obj[0,2] = 0.5
 x_obj[0,4] = 0
 x_obj[1,2] = 0
 x_obj[1,3] = 0.5
 x_obj[2,0] = 0
 x_obj[2,4] = 0.5
 x_obj[3,1] = 0.5
 x_obj[3,3] = 0
 x_obj[4,0] = 0.5
 x_obj[4,1] = 0
End
