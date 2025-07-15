\ Model dual L_k
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x1583886019caf28d
Minimize
  b_k[0] + b_k[1] + b_k[2] + b_k[3] + 2 r_k
Subject To
 contraintes[0]: b_k[0] + r_k >= 1
 contraintes[1]: b_k[1] + r_k >= 8
 contraintes[2]: b_k[2] + r_k >= 9
 contraintes[3]: b_k[3] + r_k >= 1
Bounds
End
