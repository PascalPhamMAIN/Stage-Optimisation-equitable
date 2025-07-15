\ Model L-Dominance Vertex Cover
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x25803d57377869cb
Minimize
  l[0] + l[1] + l[2] + l[3] + l[4] + l[5]
Subject To
 sync_0_0: x_vc[0,0] - x_obj[0,0] = 0
 sync_0_1: x_vc[0,1] - x_obj[0,1] = 0
 sync_1_0: x_vc[1,0] - x_obj[1,0] = 0
 sync_1_1: x_vc[1,1] - x_obj[1,1] = 0
 sync_2_0: x_vc[2,0] - x_obj[2,0] = 0
 sync_2_1: x_vc[2,1] - x_obj[2,1] = 0
 sync_3_0: x_vc[3,0] - x_obj[3,0] = 0
 sync_3_1: x_vc[3,1] - x_obj[3,1] = 0
 sync_4_0: x_vc[4,0] - x_obj[4,0] = 0
 sync_4_1: x_vc[4,1] - x_obj[4,1] = 0
 sync_5_0: x_vc[5,0] - x_obj[5,0] = 0
 sync_5_1: x_vc[5,1] - x_obj[5,1] = 0
 vc_0_1_0: x_vc[0,0] + x_vc[1,0] >= 1
 vc_0_1_1: x_vc[0,1] + x_vc[1,1] >= 1
 vc_0_3_0: x_vc[0,0] + x_vc[3,0] >= 1
 vc_0_3_1: x_vc[0,1] + x_vc[3,1] >= 1
 vc_0_5_0: x_vc[0,0] + x_vc[5,0] >= 1
 vc_0_5_1: x_vc[0,1] + x_vc[5,1] >= 1
 vc_1_2_0: x_vc[1,0] + x_vc[2,0] >= 1
 vc_1_2_1: x_vc[1,1] + x_vc[2,1] >= 1
 vc_1_3_0: x_vc[1,0] + x_vc[3,0] >= 1
 vc_1_3_1: x_vc[1,1] + x_vc[3,1] >= 1
 vc_1_4_0: x_vc[1,0] + x_vc[4,0] >= 1
 vc_1_4_1: x_vc[1,1] + x_vc[4,1] >= 1
 vc_2_4_0: x_vc[2,0] + x_vc[4,0] >= 1
 vc_2_4_1: x_vc[2,1] + x_vc[4,1] >= 1
 vc_3_4_0: x_vc[3,0] + x_vc[4,0] >= 1
 vc_3_4_1: x_vc[3,1] + x_vc[4,1] >= 1
 vc_3_5_0: x_vc[3,0] + x_vc[5,0] >= 1
 vc_3_5_1: x_vc[3,1] + x_vc[5,1] >= 1
 z_0: - 315 x_obj[0,0] - 689 x_obj[0,1] + z[0] = 0
 z_1: - 982 x_obj[1,0] - 761 x_obj[1,1] + z[1] = 0
 z_2: - 898 x_obj[2,0] - 121 x_obj[2,1] + z[2] = 0
 z_3: - 708 x_obj[3,0] - 881 x_obj[3,1] + z[3] = 0
 z_4: - 178 x_obj[4,0] - 843 x_obj[4,1] + z[4] = 0
 z_5: - 953 x_obj[5,0] - 98 x_obj[5,1] + z[5] = 0
Bounds
 x_vc[0,0] <= 1
 x_vc[0,1] <= 1
 x_vc[1,0] <= 1
 x_vc[1,1] <= 1
 x_vc[2,0] <= 1
 x_vc[2,1] <= 1
 x_vc[3,0] <= 1
 x_vc[3,1] <= 1
 x_vc[4,0] <= 1
 x_vc[4,1] <= 1
 x_vc[5,0] <= 1
 x_vc[5,1] <= 1
 x_obj[0,0] <= 1
 x_obj[0,1] <= 1
 x_obj[1,0] <= 1
 x_obj[1,1] <= 1
 x_obj[2,0] <= 1
 x_obj[2,1] <= 1
 x_obj[3,0] <= 1
 x_obj[3,1] <= 1
 x_obj[4,0] <= 1
 x_obj[4,1] <= 1
 x_obj[5,0] <= 1
 x_obj[5,1] <= 1
End
