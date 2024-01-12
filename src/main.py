import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, "../.."))

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
)

from preprocess import load_preprocessed_data
from logistic import BinaryLogisticClassification


def train_naive_bayes(X_train, y_train, log_file):
    tuned_parameters = [{"alpha": [0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=5, scoring="f1")
    clf.fit(X_train, y_train)
    log_file.write(f"Naive Bayes finished training\n")

    return clf


def train_decision_tree(X_train, y_train, log_file):
    tuned_parameters = [{"max_depth": [5, 10, 20, 50, 100]}]
    clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring="f1")
    clf.fit(X_train, y_train)
    log_file.write(f"Decision Tree best hyper param: {clf.best_params_}\n")

    return clf


def train_logistic_regression(X_train, y_train, log_file):
    model = BinaryLogisticClassification()
    model.fit(X_train, y_train)
    log_file.write(f"Logistic Regression finished training\n")

    return model


def train_random_forest(X_train, y_train, log_file):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)
    log_file.write(f"Random Forest finished training\n")

    return model


def train_SVC(X_train, y_train, log_file):
    model = SVC(kernel="rbf", gamma=0.1, C=1)
    model.fit(X_train, y_train)
    log_file.write(f"SVC finished training\n")

    return model


def main():
    log_file = open("./result.log", "w")
    for balance_labels, encode in [(True, "label")]:
        log_file.write("==================================================\n")
        log_file.write(f"Balance labels: {balance_labels}, Encode Strategy: {encode}\n")
        log_file.write("==================================================\n")
        log_file.flush()

        X_train, X_test, y_train, y_test = load_preprocessed_data(
            balance_labels, encode
        )

        for trainer in [
            train_naive_bayes,
            train_random_forest,
            train_logistic_regression,
            train_SVC,
            train_decision_tree,
        ]:
            model = trainer(X_train, y_train, log_file)
            y_true, y_pred = y_test, model.predict(X_test)
            acc = accuracy_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            log_file.write(f"Accuracy: {acc}\nROC AUC: {roc_auc}\nF1: {f1}\n\n")
            log_file.write(classification_report(y_true, y_pred) + "\n")
            log_file.flush()

    log_file.close()


if __name__ == "__main__":
    main()
