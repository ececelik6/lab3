import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""

    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)

    n_samples, n_features = X_train.shape

    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        predictions = X_train @ weights + bias
        errors = predictions - y_train

        mse = float(np.mean(errors ** 2))
        loss_history.append(mse)

        dw = (2.0 / n_samples) * (X_train.T @ errors)
        db = 2.0 * float(np.mean(errors))

        weights -= data.learning_rate * dw
        bias -= data.learning_rate * db

    test_predictions = X_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=test_predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""

    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)

    n_samples, n_features = X_train.shape

    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    loss_history = []

    for _ in range(data.epochs):
        linear_output = X_train @ weights + bias
        linear_output = np.clip(linear_output, -500, 500)

        probabilities = 1.0 / (1.0 + np.exp(-linear_output))

        epsilon = 1e-15
        loss = -np.mean(
            y_train * np.log(probabilities + epsilon)
            + (1.0 - y_train) * np.log(1.0 - probabilities + epsilon)
        )

        loss_history.append(float(loss))

        errors = probabilities - y_train

        dw = (1.0 / n_samples) * (X_train.T @ errors)
        db = float(np.mean(errors))

        weights -= data.learning_rate * dw
        bias -= data.learning_rate * db

    test_linear_output = X_test @ weights + bias
    test_linear_output = np.clip(test_linear_output, -500, 500)

    test_probabilities = 1.0 / (
        1.0 + np.exp(-test_linear_output)
    )

    test_predictions = (test_probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=test_predictions,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните собственную Logistic Regression, Decision Tree и KNN на одном split."""

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=int).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=int).reshape(-1)

    def calculate_metrics(probabilities, predictions) -> ClassificationMetrics:
        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_test, probabilities)),
            accuracy=float(accuracy_score(y_test, predictions)),
            precision=float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            recall=float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            f1=float(
                f1_score(y_test, predictions, zero_division=0)
            ),
        )

    # Собственная логистическая регрессия
    own_logistic = train_logistic_classifier(data)

    own_logistic_metrics = calculate_metrics(
        own_logistic.probabilities,
        own_logistic.predictions,
    )

    # Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    tree_probabilities = decision_tree.predict_proba(X_test)[:, 1]
    tree_predictions = decision_tree.predict(X_test)

    decision_tree_metrics = calculate_metrics(
        tree_probabilities,
        tree_predictions,
    )

    # KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    knn_probabilities = knn.predict_proba(X_test)[:, 1]
    knn_predictions = knn.predict(X_test)

    knn_metrics = calculate_metrics(
        knn_probabilities,
        knn_predictions,
    )

    return BaselineComparisonResult(
        logistic_regression=own_logistic_metrics,
        decision_tree=decision_tree_metrics,
        knn=knn_metrics,
    )