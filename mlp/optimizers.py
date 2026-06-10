import numpy as np


class SGD:
    def __init__(self, taxa_aprendizado=0.01):
        self.taxa_aprendizado = taxa_aprendizado

    def atualizar(self, rede, grads_W, grads_b):
        for i in range(rede.num_layers):
            rede.weights[i] -= self.taxa_aprendizado * grads_W[i]
            rede.biases[i]  -= self.taxa_aprendizado * grads_b[i]


def treinar(rede, otimizador, funcao_perda, X, y, epocas=20, tamanho_lote=64):
    historico = {'perda': [], 'acuracia': []}
    N = X.shape[0]

    for epoca in range(epocas):
        indices = np.random.permutation(N)
        X_embaralhado = X[indices]
        y_embaralhado = y[indices]

        perda_epoca = 0.0
        num_lotes = 0

        for inicio in range(0, N, tamanho_lote):
            X_lote = X_embaralhado[inicio:inicio + tamanho_lote]
            y_lote = y_embaralhado[inicio:inicio + tamanho_lote]

            y_previsto = rede.forward(X_lote)
            perda_lote = funcao_perda(y_lote, y_previsto)
            perda_epoca += perda_lote

            grads_W, grads_b = rede.backward(y_lote)
            otimizador.atualizar(rede, grads_W, grads_b)
            num_lotes += 1

        perda_media = perda_epoca / num_lotes
        predicoes = rede.predict(X)
        acuracia = np.mean(predicoes == np.argmax(y, axis=1))

        historico['perda'].append(perda_media)
        historico['acuracia'].append(acuracia)

        if (epoca + 1) % 5 == 0 or epoca == 0:
            print(f'Epoca {epoca+1:>3}/{epocas}  perda: {perda_media:.4f}  acuracia: {acuracia:.4f}')

    return historico
