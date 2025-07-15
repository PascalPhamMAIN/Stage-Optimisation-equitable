\ Model linear program
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x7a954f9c38f0c1c6
Maximize
  alpha_k[0] + 8 alpha_k[1] + 9 alpha_k[2] + alpha_k[3]
Subject To
 c1: alpha_k[0] + alpha_k[1] + alpha_k[2] + alpha_k[3] = 2
Bounds
 alpha_k[0] <= 1
 alpha_k[1] <= 1
 alpha_k[2] <= 1
 alpha_k[3] <= 1
End
