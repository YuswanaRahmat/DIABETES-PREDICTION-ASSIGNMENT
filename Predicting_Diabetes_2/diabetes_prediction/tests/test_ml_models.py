import numpy as np

from diabetes_prediction.data.data_loader import load_data
from diabetes_prediction.preprocessing.preprocessing import preprocess_data
from diabetes_prediction.models.ml_models import get_classifiers, train_models
from diabetes_prediction.config import DATA_FILE


def test_ml_models_fit_and_predict():
    df = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = preprocess_data(df, test_size=0.4)
    models = get_classifiers(random_state=42)
    trained = train_models(models, X_train, y_train)

    for model in trained.values():
        y_pred = model.predict(X_test)
        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == y_test.shape
