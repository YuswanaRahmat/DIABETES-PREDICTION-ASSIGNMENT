from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from diabetes_prediction.data.data_loader import load_data
from diabetes_prediction.preprocessing.preprocessing import (
    clean_data,
    create_target,
    preprocess_data,
    split_train_test,
)
from diabetes_prediction.models.ml_models import get_classifiers, train_models
from diabetes_prediction.models.deep_learning import build_mlp_classifier
from diabetes_prediction.evaluation.evaluation import evaluate_model, print_classification_report
from diabetes_prediction.utils.visualization import (
    plot_correlation_heatmap,
    plot_histograms,
    plot_status_distribution,
)
from diabetes_prediction.config import DATA_FILE, RANDOM_STATE


def main() -> None:
    print("Loading dataset...")
    df = load_data(DATA_FILE)

    print("Cleaning and visualizing data...")
    cleaned_df = clean_data(df)
    target_df = create_target(cleaned_df)
    plot_histograms(cleaned_df)
    plot_correlation_heatmap(cleaned_df)
    plot_status_distribution(target_df)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training classic ML models...")
    classifiers = get_classifiers(random_state=RANDOM_STATE)
    trained_models = train_models(classifiers, X_train, y_train)

    for name, model in trained_models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model, X_test, y_test)
        print_classification_report(name, metrics)

    print("Training MLP model...")
    mlp = build_mlp_classifier(random_state=RANDOM_STATE)
    mlp.fit(X_train, y_train)
    mlp_metrics = evaluate_model(mlp, X_test, y_test)
    print_classification_report("mlp_classifier", mlp_metrics)


if __name__ == "__main__":
    main()
