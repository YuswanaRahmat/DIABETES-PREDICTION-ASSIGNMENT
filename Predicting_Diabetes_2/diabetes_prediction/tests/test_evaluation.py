from diabetes_prediction.data.data_loader import load_data
from diabetes_prediction.preprocessing.preprocessing import preprocess_data
from diabetes_prediction.models.ml_models import get_classifiers, train_models
from diabetes_prediction.evaluation.evaluation import evaluate_model
from diabetes_prediction.config import DATA_FILE


def test_evaluate_model_returns_metrics():
    df = load_data(DATA_FILE)
    X_train, X_test, y_train, y_test = preprocess_data(df, test_size=0.4)
    classifiers = get_classifiers(random_state=42)
    trained = train_models(classifiers, X_train, y_train)
    metrics = evaluate_model(trained["logistic_regression"], X_test, y_test)

    assert "accuracy" in metrics
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    assert isinstance(metrics["accuracy"], float)
