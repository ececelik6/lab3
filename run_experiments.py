import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from grader_contracts.linear_models import RegressionInput
from linear_models import train_linear_regression


# Veri setini yükle
df = pd.read_csv("Student_Performance.csv")

# Yes / No değerlerini sayıya çevir
df["Extracurricular Activities"] = (
    df["Extracurricular Activities"]
    .map({"No": 0, "Yes": 1})
)

# Özellikler ve hedef değişken
X = df.drop(columns=["Performance Index"])
y = df["Performance Index"]

# Train / test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Özellikleri ölçeklendir
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kendi yazdığımız linear regression modelini eğit
result = train_linear_regression(
    RegressionInput(
        train_features=X_train_scaled,
        train_target=y_train.to_numpy(),
        test_features=X_test_scaled,
        test_target=y_test.to_numpy(),
        learning_rate=0.05,
        epochs=800,
    )
)

# Test MSE
test_mse = mean_squared_error(
    y_test,
    result.predictions,
)

print("Initial MSE:", result.loss_history[0])
print("Final training MSE:", result.loss_history[-1])
print("Test MSE:", test_mse)
print("Weights:", result.weights)
print("Bias:", result.bias)

# --------------------------------------------------
# Logistic Regression experiment
# --------------------------------------------------

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from grader_contracts.linear_models import ClassificationInput
from linear_models import train_logistic_classifier


# German Credit dataset
german_df = pd.read_csv("german.csv", sep=";")

X_class = german_df.iloc[:, 1:]
y_class = german_df.iloc[:, 0]

# Aynı sınıf oranlarını koruyarak train / test ayır
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class,
)

# Özellikleri ölçeklendir
class_scaler = StandardScaler()

X_train_c_scaled = class_scaler.fit_transform(X_train_c)
X_test_c_scaled = class_scaler.transform(X_test_c)

# Kendi logistic regression modelimizi eğit
classification_result = train_logistic_classifier(
    ClassificationInput(
        train_features=X_train_c_scaled,
        train_target=y_train_c.to_numpy(),
        test_features=X_test_c_scaled,
        test_target=y_test_c.to_numpy(),
        learning_rate=0.1,
        epochs=1000,
    )
)

# Metrikler
roc_auc = roc_auc_score(
    y_test_c,
    classification_result.probabilities,
)

accuracy = accuracy_score(
    y_test_c,
    classification_result.predictions,
)

precision = precision_score(
    y_test_c,
    classification_result.predictions,
    zero_division=0,
)

recall = recall_score(
    y_test_c,
    classification_result.predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test_c,
    classification_result.predictions,
    zero_division=0,
)

print("\n--- Own Logistic Regression ---")
print("Initial BCE:", classification_result.loss_history[0])
print("Final BCE:", classification_result.loss_history[-1])
print("ROC-AUC:", roc_auc)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)


from linear_models import compare_classical_models


comparison = compare_classical_models(
    ClassificationInput(
        train_features=X_train_c_scaled,
        train_target=y_train_c.to_numpy(),
        test_features=X_test_c_scaled,
        test_target=y_test_c.to_numpy(),
    )
)

print("\n--- Classical Models Comparison ---")
print("Logistic Regression:", comparison.logistic_regression)
print("Decision Tree:", comparison.decision_tree)
print("KNN:", comparison.knn)