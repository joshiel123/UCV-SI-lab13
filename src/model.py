from sklearn.neural_network import MLPClassifier


def create_model():
    return MLPClassifier(
        hidden_layer_sizes=(4,),
        max_iter=2000,
        learning_rate_init=0.01,
        random_state=1
    )