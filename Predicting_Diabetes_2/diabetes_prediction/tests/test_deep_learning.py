import numpy as np

from diabetes_prediction.data.data_loader import load_data
from diabetes_prediction.preprocessing.preprocessing import preprocess_data
from diabetes_prediction.models.deep_learning import build_mlp_classifier
from diabetes_prediction.config import DATA_FILE


def test_mlp_classifier_train_and_predict():
    df = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = preprocess_data(df, test_size=0.4)
    model = build_mlp_classifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_test.shape
