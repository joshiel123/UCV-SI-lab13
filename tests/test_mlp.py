from src.dataset import load_data
from src.model import create_model


def test_xor_dataset_shape():
    X, y = load_data()

    assert X.shape == (4, 2)
    assert y.shape == (4,)


def test_model_can_fit():
    X, y = load_data()

    model = create_model()
    model.fit(X, y)

    assert hasattr(model, "predict")


def test_model_predicts_valid_output():
    X, y = load_data()

    model = create_model()
    model.fit(X, y)

    pred = model.predict(X)

    # Validación básica de forma
    assert len(pred) == len(y)

    # XOR solo debe producir 0 o 1
    assert set(pred).issubset({0, 1})


def test_model_learns_xor():
    X, y = load_data()

    model = create_model()
    model.fit(X, y)

    pred = model.predict(X)

    # El modelo debería poder resolver XOR o acercarse mucho
    assert sum(pred == y) >= 3  # al menos 3 de 4 correctas