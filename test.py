import numpy as np
import matplotlib.pyplot as plt

# Eixo de "capacidade" do modelo (p. ex., número de parâmetros normalizado)
capacity = np.linspace(0.1, 3.0, 300)

# Erro de treino: tende a cair monotonicamente com mais capacidade
train_err = 0.35 / (capacity + 0.3) + 0.02

# Erro de teste: desce, sobe perto da fronteira de interpolação (~1.0) e volta a descer
base = 0.35 / (capacity + 0.3)                                # tendência geral de queda
bump = 0.4 * np.exp(-((capacity - 1.0) ** 2) / 0.02)          # pico perto da interpolação
test_err = base + bump

plt.figure(figsize=(7, 4))
plt.plot(capacity, train_err, label="train error")
plt.plot(capacity, test_err, label="test error")
plt.xlabel("Capacidade do modelo (normalizada)")
plt.ylabel("Erro")
plt.title("Curva qualitativa de double descent")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
