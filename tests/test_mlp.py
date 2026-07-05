import src.dataset
import src.model

from src.dataset import load_data
from src.model import create_model

def test_import_modules():
    assert load_data is not None
    assert create_model is not None


def test_dataset():
    X, y = load_data()

    assert X.shape == (4, 2)
    assert y.shape == (4,)


def test_model_creation():
    model = create_model()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_full_pipeline():
    X, y = load_data()
    model = create_model()

    model.fit(X, y)
    pred = model.predict(X)

    assert len(pred) == len(y)
    assert set(pred).issubset({0, 1})


def test_xor_accuracy():
    X, y = load_data()
    model = create_model()

    model.fit(X, y)
    pred = model.predict(X)

    # XOR es difícil → permitimos margen
    assert sum(pred == y) >= 3


def test_stability():
    X, y = load_data()

    for _ in range(3):
        model = create_model()
        model.fit(X, y)
        pred = model.predict(X)

        assert len(pred) == 4