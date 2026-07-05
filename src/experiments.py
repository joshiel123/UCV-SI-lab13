import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 👇 esto va arriba de todo
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Dataset XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

configuraciones = [2, 4, 8]

resultados = []

print("\nRESULTADOS DE CONFIGURACIONES MLP")
print("=" * 40)

for n in configuraciones:

    modelo = MLPClassifier(
        hidden_layer_sizes=(n,),
        max_iter=2000,
        random_state=1
    )

    modelo.fit(X, y)

    pred = modelo.predict(X)

    acc = accuracy_score(y, pred)

    resultados.append((n, acc))

    print(f"Neuronas: {n}")
    print(f"Predicciones: {pred}")
    print(f"Exactitud: {acc}")
    print("-" * 40)

print("\nTABLA COMPARATIVA")
print("=" * 40)

for n, acc in resultados:
    print(f"{n} neuronas -> Exactitud: {acc}")

mejor = max(resultados, key=lambda x: x[1])

print("\nMejor configuración:")
print(f"{mejor[0]} neuronas con exactitud {mejor[1]}")