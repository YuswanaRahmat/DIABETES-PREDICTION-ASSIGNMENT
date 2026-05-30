from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict


def get_classifiers(random_state: int = 42) -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }


def train_models(models: Dict[str, object], X_train, y_train) -> Dict[str, object]:
    trained: Dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained
