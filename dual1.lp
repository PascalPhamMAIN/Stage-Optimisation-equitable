\ Model linear program
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x1fc7b21e380d8cc2
Maximize
  2 x + y
Subject To
 c1: 3 x + y <= -2
 c2: 2 x + 2 y >= -10
Bounds
 -infinity <= x <= 0
 -infinity <= y <= 0
End
