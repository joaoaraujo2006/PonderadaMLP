# MLP do Zero — Classificação de Dígitos MNIST

Implementação de um Multi-Layer Perceptron do zero, usando apenas NumPy. Sem PyTorch, TensorFlow ou qualquer framework de deep learning — cada operação de forward pass, backpropagation e atualização de pesos foi escrita manualmente.

---

## Como rodar

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar os experimentos
jupyter notebook notebooks/experimentos.ipynb
```

---

## Arquitetura escolhida

**Configuração final:** `[784 → 128 → 64 → 10]`

| Camada | Neurônios | Ativação |
|--------|-----------|----------|
| Entrada | 784 | — |
| Oculta 1 | 128 | ReLU |
| Oculta 2 | 64 | ReLU |
| Saída | 10 | Softmax |

**Por que essas escolhas:**

- **784 entradas:** imagens 28×28 achatadas em vetor
- **ReLU nas camadas ocultas:** não sofre com o problema de gradiente que desaparece como a sigmoid, e sua derivada é simples (0 ou 1), o que torna o backprop eficiente
- **Softmax na saída:** converte logits em probabilidades somando 1, combinado com cross-entropy produz um gradiente limpo: `ŷ - y`
- **He initialization:** pesos inicializados com `N(0, √(2/n))` — projetado especificamente para ReLU evitar saturação nas primeiras épocas
- **128 e 64 neurônios:** capacidade suficiente para aprender as features do MNIST sem overfitting excessivo

---

## Resultados

**Acurácia final no teste:** `XX.XX%` ← preencher após rodar o notebook

### Curva de loss e acurácia

![Curvas de treinamento](results/curvas_mnist.png)

### Tabela comparativa de experimentos

| Configuração | Acurácia Teste |
|---|---|
| Baseline `[784,128,64,10]` lr=0.01 | XX.XX% |
| Maior learning rate lr=0.1 | XX.XX% |
| Rede maior `[784,256,128,10]` lr=0.01 | XX.XX% |
| Lote menor batch=32 | XX.XX% |

---

## Decisões e dificuldades

### Qual foi a decisão técnica mais difícil que tomei?

<!-- Escreva aqui em primeira pessoa. Exemplo de resposta honesta:
A decisão mais difícil foi a inicialização dos pesos. Na primeira tentativa inicializei tudo com zero
e a rede não aprendia nada — todos os neurônios calculavam o mesmo gradiente e evoluíam de forma idêntica.
Isso se chama "symmetry breaking problem". Migrei para He initialization (pesos aleatórios escalados por
√(2/n)) e a rede passou a convergir normalmente. -->

### O que tentei que não funcionou?

<!-- Escreva aqui. Exemplo:
Tentei usar learning rate 0.1 direto no MNIST. A loss explodiu na segunda época — os gradientes ficaram
grandes demais e os pesos divergiram. Abaixar para 0.01 resolveu. Também tentei sem normalizar as entradas
(÷255) e a convergência foi muito mais lenta e instável. -->

### Se fosse refazer do zero, o que faria diferente?

<!-- Escreva aqui. Exemplo:
Implementaria o gradient check logo após o backprop, antes de qualquer treinamento. Passei tempo procurando
um bug que estava na multiplicação element-wise da derivada da ReLU — `*` em vez de `@` — e um gradient
check teria me dado a resposta em segundos. -->

---

## Estrutura do repositório

```
.
├── README.md
├── mlp/
│   ├── __init__.py
│   ├── network.py      ← MLP com forward e backpropagation
│   ├── activations.py  ← ReLU e Softmax com derivadas
│   ├── losses.py       ← Cross-entropy
│   └── optimizers.py   ← SGD com mini-batches
├── notebooks/
│   └── experimentos.ipynb
├── results/
│   └── curvas_mnist.png
└── requirements.txt
```
