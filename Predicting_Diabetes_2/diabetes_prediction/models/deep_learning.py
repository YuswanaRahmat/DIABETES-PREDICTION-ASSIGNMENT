from sklearn.neural_network import MLPClassifier


def build_mlp_classifier(random_state: int = 42) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=random_state,
    )
