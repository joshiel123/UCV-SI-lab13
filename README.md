# UCV-SI-Lab13: Resolución del problema XOR con Perceptrón Multicapa (MLP)

Este proyecto implementa y evalúa una red neuronal artificial del tipo **Perceptrón Multicapa (MLP)** utilizando la biblioteca `scikit-learn` en Python para resolver el problema clásico de la compuerta lógica **XOR**.

---

## 📂 Estructura del Proyecto

El repositorio está estructurado de la siguiente manera:

*   **[src/dataset.py](file:///c:/Users/User/Repositorios/UCV-SI-lab13/src/dataset.py)**: Define la función `load_data()` que retorna las entradas y salidas de la compuerta XOR en formato de arreglos de NumPy.
*   **[src/model.py](file:///c:/Users/User/Repositorios/UCV-SI-lab13/src/model.py)**: Contiene la función `create_model()`, encargada de instanciar un clasificador `MLPClassifier` configurado con una arquitectura base (4 neuronas ocultas, optimizador Adam y un máximo de 2000 iteraciones).
*   **[src/experiments.py](file:///c:/Users/User/Repositorios/UCV-SI-lab13/src/experiments.py)**: Realiza pruebas comparativas utilizando tres configuraciones distintas de neuronas en la capa oculta: 2, 4 y 8 neuronas.
*   **[src/main.py](file:///c:/Users/User/Repositorios/UCV-SI-lab13/src/main.py)**: Ejecuta el entrenamiento y evaluación rápida del modelo con la configuración por defecto de 4 neuronas.
*   **[tests/test_mlp.py](file:///c:/Users/User/Repositorios/UCV-SI-lab13/tests/test_mlp.py)**: Suite de pruebas unitarias (`pytest`) que aseguran la carga correcta del dataset, la creación del modelo, la estabilidad del entrenamiento y la precisión mínima exigida.

---

## 🚀 Instrucciones de Configuración y Ejecución

### 1. Requisitos Previos
Asegúrate de contar con Python 3.11+ instalado.

### 2. Instalación de Dependencias
Para activar el entorno virtual y cargar los paquetes necesarios (`numpy`, `scikit-learn`, `pytest`, `coverage`), ejecuta los siguientes comandos en tu terminal de Windows:

```powershell
# Activar entorno virtual
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Ejecución del Flujo Principal
Para entrenar el modelo por defecto y ver sus predicciones en consola:

```powershell
python src/main.py
```

### 4. Ejecución de los Experimentos de Arquitectura
Para comparar el rendimiento de las diferentes cantidades de neuronas ocultas:

```powershell
python src/experiments.py
```

### 5. Ejecución de Pruebas Unitarias
Para correr los tests automatizados:

```powershell
python -m pytest
```

---

## 📊 Resultados de los Experimentos

A continuación se detallan los resultados obtenidos al ejecutar el script de experimentos `src/experiments.py` con tres configuraciones de neuronas ocultas:

| Configuración (Neuronas Ocultas) | Predicciones para XOR `[[0,0], [0,1], [1,0], [1,1]]` | Exactitud (Accuracy) |
| :---: | :---: | :---: |
| **2 neuronas** | `[0, 0, 0, 0]` | **50.0% (0.5)** |
| **4 neuronas** | `[0, 1, 1, 0]` | **100.0% (1.0)** |
| **8 neuronas** | `[0, 1, 1, 0]` | **100.0% (1.0)** |

### Análisis de Resultados
*   **¿Cuál configuración funciona mejor?**
    Las configuraciones de **4 y 8 neuronas** logran resolver perfectamente el problema XOR con una **exactitud del 100% (1.0)**, prediciendo los valores correctos `[0, 1, 1, 0]`. La configuración de **2 neuronas** falla, logrando una exactitud de solo el **50%** al predecir la clase mayoritaria `0` para todas las muestras.
*   **¿Por qué sucede esto?**
    1.  **2 neuronas ocultas**: Teóricamente, el número mínimo para resolver XOR es 2 neuronas en la capa oculta. Sin embargo, en la práctica y bajo la inicialización aleatoria de pesos (fijada con `random_state=1`), el solver (Adam) a menudo se estanca en mínimos locales o requiere condiciones de optimización sumamente precisas que no se alcanzan en 2000 iteraciones con hiperparámetros por defecto. Esto causa que la red colapse a una frontera lineal que clasifica todos los ejemplos como `0`.
    2.  **4 y 8 neuronas ocultas**: Al añadir más neuronas a la capa oculta, aumentamos la dimensionalidad del espacio intermedio y la cantidad de parámetros libres (pesos y sesgos). Esto provee un "paisaje de pérdida" mucho más suave y redundante donde el optimizador encuentra fácilmente un camino hacia el mínimo global (exactitud del 100%).
    3.  **Comparativa entre 4 y 8 neuronas**: Aunque ambas logran la exactitud perfecta, la configuración de **4 neuronas** es la óptima y más eficiente. Ofrece el equilibrio ideal entre capacidad de representación (suficiente para resolver la no linealidad) y simplicidad computacional, evitando la redundancia innecesaria y el sobreajuste que traería consigo la configuración de 8 neuronas en problemas más complejos.

---

## 🧠 Cuestionario Teórico

### 1. ¿Por qué el perceptrón simple no puede resolver XOR?
El perceptrón simple es un clasificador puramente lineal. Matemáticamente, computa una combinación lineal de sus entradas y aplica una función de paso (escalón): 
$$\hat{y} = f(\mathbf{w}^T\mathbf{x} + b)$$
Esto restringe su frontera de decisión a un hiperplano (una línea recta en un plano 2D). 

Al mapear el problema XOR en un gráfico bidimensional:
*   $(0,0) \rightarrow 0$ (Clase A)
*   $(1,1) \rightarrow 0$ (Clase A)
*   $(0,1) \rightarrow 1$ (Clase B)
*   $(1,0) \rightarrow 1$ (Clase B)

Es geométricamente imposible trazar una única línea recta que separe los puntos de la Clase A de los de la Clase B (están posicionados de forma diagonalmente opuesta). Al no ser linealmente separable, el perceptrón simple es incapaz de converger a una solución correcta.

### 2. ¿Qué papel cumplen las capas ocultas?
Las capas ocultas actúan como un **transformador de características no lineal**. Su función principal es tomar los datos de entrada originales y proyectarlos a un nuevo espacio geométrico de mayor o diferente dimensión, donde las clases se vuelvan linealmente separables.

Al combinar las salidas de múltiples neuronas ocultas a través de funciones de activación no lineales (como ReLU, Sigmoide o Tangente Hiperbólica), la red puede deformar el espacio de entrada. La neurona en la capa de salida simplemente tiene que realizar una separación lineal clásica sobre esta representación transformada para producir la clasificación final correcta.

### 3. ¿Qué sucede si se aumentan demasiado las neuronas?
Si bien un número mayor de neuronas aumenta la capacidad de la red para aprender relaciones complejas, un exceso de las mismas acarrea graves inconvenientes:
1.  **Sobreajuste (Overfitting)**: La red adquiere una capacidad de memorización tan alta que aprende el ruido y los detalles específicos de los datos de entrenamiento, perdiendo su poder de generalización. Su rendimiento en datos nuevos (no vistos) decaerá drásticamente.
2.  **Costo Computacional Elevado**: El número de conexiones (pesos y sesgos) crece de forma cuadrática o exponencial respecto a la cantidad de neuronas. Esto demanda mayor uso de memoria (RAM/VRAM), ralentiza los tiempos de entrenamiento e incrementa el consumo de energía.
3.  **Inestabilidad en la Optimización**: Espacios de parámetros extremadamente grandes pueden hacer que el gradiente se desvanezca o explote, o que el optimizador tarde mucho más tiempo en encontrar soluciones estables debido a la abundancia de valles planos y puntos de silla en la función de pérdida.