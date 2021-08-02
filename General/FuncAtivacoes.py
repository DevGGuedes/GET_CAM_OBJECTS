import numpy as np

def stepFunction(soma):
	if(soma >= 1):
		return 1
	return 0

#função sigmoid 
def sigmoidFunction(soma):
	return 1/ (1 + np.exp(-soma))

#função tangente hiperbolica(Hyperbolic tanget)
def tahnFunction(soma):
	return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

#função ReLU (rectifed linear units)
def reluFunction(soma):
	if soma >= 0:
		return soma
	return 0

#função linear
def linearFunction(soma):
	return soma

#exp = exponencial
#função softmax
def softmaxFunction(x):
	ex = np.exp(x)
	return ex / ex.sum()


step = stepFunction(-1)

#valor da some retorna 0358 para usar no sigmoid
sigmoid = sigmoidFunction(0.358)
print(sigmoid)

tahn = tahnFunction(-0.358)
print(tahn)

relu = reluFunction(0.358)
print(relu)

linear = linearFunction(0.358)
print(linear)

valores = [5.0, 2.0, 1.3]
#retorna probabilidades de acordo com o vetor
print(softmaxFunction(valores))