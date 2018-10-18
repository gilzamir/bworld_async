//Primeiro calcule os valores máximos e mínimos:
max = máximo(R,G,B), min = mínimo(R,G,B)
//depois os valores de saturação e brilho:
V = max , S = (max - min) / max

//ai passe a calcular as cores ou H:
if S = 0 /* H passa a ser irrelevante, a cor no HSV será : (0,0,V) */
else
	R1 = (R-min) / (max-min)
	G1 = (G-min) / (max-min)
	B1 = (B-min) / (max-min)
if R1 = max , H = G1 - B1
else if	G1 = max , H = 2 + B1 - R1
else if B1 = max , H = 4 + R1 - G1

//(converte-se H em graus)
H = H*60

//usa-se H variando de 0 a 360Â° , S e V variando entre 0 e 1
if H < 0 , H=H+360

// a cor no HSV será : (H,S,V)*/

