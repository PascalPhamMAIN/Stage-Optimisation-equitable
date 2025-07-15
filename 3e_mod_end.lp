\ Model L-Dominance Vertex Cover
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x8a3d0c9ce4a5548b
Minimize
  0 l[0] + 0 l[1] + 0 l[2] + 0 l[3] + 0 l[4] + 0 l[5] + 70 Constant
Subject To
 sync_5_0: x_vc[5,0] - x_obj[5,0] = 0
 sync_5_1: x_vc[5,1] - x_obj[5,1] = 0
 vc_0_1_0: x_vc[0,0] + x_vc[1,0] >= 1
 vc_0_1_1: x_vc[0,1] + x_vc[1,1] >= 1
 vc_0_2_0: x_vc[0,0] + x_vc[2,0] >= 1
 vc_0_2_1: x_vc[0,1] + x_vc[2,1] >= 1
 vc_0_3_0: x_vc[0,0] + x_vc[3,0] >= 1
 vc_0_3_1: x_vc[0,1] + x_vc[3,1] >= 1
 vc_1_2_0: x_vc[1,0] + x_vc[2,0] >= 1
 vc_1_2_1: x_vc[1,1] + x_vc[2,1] >= 1
 vc_1_4_0: x_vc[1,0] + x_vc[4,0] >= 1
 vc_1_4_1: x_vc[1,1] + x_vc[4,1] >= 1
 vc_1_5_0: x_vc[1,0] + x_vc[5,0] >= 1
 vc_1_5_1: x_vc[1,1] + x_vc[5,1] >= 1
 vc_2_3_0: x_vc[2,0] + x_vc[3,0] >= 1
 vc_2_3_1: x_vc[2,1] + x_vc[3,1] >= 1
 vc_2_4_0: x_vc[2,0] + x_vc[4,0] >= 1
 vc_2_4_1: x_vc[2,1] + x_vc[4,1] >= 1
 vc_2_5_0: x_vc[2,0] + x_vc[5,0] >= 1
 vc_2_5_1: x_vc[2,1] + x_vc[5,1] >= 1
 vc_3_4_0: x_vc[3,0] + x_vc[4,0] >= 1
 vc_3_4_1: x_vc[3,1] + x_vc[4,1] >= 1
 vc_3_5_0: x_vc[3,0] + x_vc[5,0] >= 1
 vc_3_5_1: x_vc[3,1] + x_vc[5,1] >= 1
 vc_4_5_0: x_vc[4,0] + x_vc[5,0] >= 1
 vc_4_5_1: x_vc[4,1] + x_vc[5,1] >= 1
 z_0: - 315 x_obj[0,0] - 193 x_obj[0,1] + z[0] = 0
 z_1: - 953 x_obj[1,0] - 578 x_obj[1,1] + z[1] = 0
 z_2: - 184 x_obj[2,0] - 429 x_obj[2,1] + z[2] = 0
 z_3: - 636 x_obj[3,0] - 362 x_obj[3,1] + z[3] = 0
 z_4: - 631 x_obj[4,0] - 914 x_obj[4,1] + z[4] = 0
 z_5: - 515 x_obj[5,0] - 338 x_obj[5,1] + z[5] = 0
 l_0: z[5] <= 5
Bounds
 x_vc[0,0] = 1
 x_vc[0,1] = 1
 x_vc[1,0] = 1
 x_vc[1,1] = 1
 x_vc[2,0] = 1
 x_vc[2,1] = 1
 x_vc[3,0] = 1
 x_vc[3,1] = 1
 x_vc[4,0] = 1
 x_vc[4,1] = 1
 x_vc[5,0] <= 1
 x_vc[5,1] <= 1
 x_obj[0,0] = 0.5
 x_obj[0,1] = 0.5
 x_obj[1,0] = 0.5
 x_obj[1,1] = 0.5
 x_obj[2,0] = 0.5
 x_obj[2,1] = 0.5
 x_obj[3,0] = 0.5
 x_obj[3,1] = 0.5
 x_obj[4,0] = 0.5
 x_obj[4,1] = 0.5
 x_obj[5,0] <= 1
 x_obj[5,1] <= 1
 Constant = 1
End
