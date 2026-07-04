from dataset import load_data
from model import create_model
from sklearn.metrics import accuracy_score

X, y = load_data()

modelo = create_model()
modelo.fit(X, y)

pred = modelo.predict(X)

print("Predicciones:", pred)
print("Exactitud:", accuracy_score(y, pred))