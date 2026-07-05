from sklearn.neural_network import MLPClassifier


def create_model():
    """
    Crea y devuelve el modelo MLP configurado para XOR.
    """
    return MLPClassifier(
        hidden_layer_sizes=(4,),
        max_iter=2000,
        random_state=1
    )