import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

YN_DICT = {"yes": 1, "no": 0}


def load_preprocessed_data(balance_labels=True, encode="onehot"):
    from ucimlrepo import fetch_ucirepo

    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    # Concat X, y
    df = pd.concat([X, y], axis=1)

    # Scale all the numeric features
    scaler = MinMaxScaler()
    numeric_features = ["age", "campaign", "pdays", "previous", "balance", "duration"]
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Handle ordinal columns
    df["default"] = df["default"].map(YN_DICT)
    df["housing"] = df["housing"].map(YN_DICT)
    df["loan"] = df["loan"].map(YN_DICT)

    # Handle categorical columns
    categorical_features = [
        "job",
        "marital",
        "education",
        "contact",
        "day_of_week",
        "month",
        "poutcome",
    ]
    if encode == "onehot":  # One-hot encode categorical features
        df = pd.get_dummies(df, columns=categorical_features)

    elif encode == "label":  # Label encode categorical features
        encoder = LabelEncoder()
        df[categorical_features] = df[categorical_features].apply(encoder.fit_transform)

    df["y"] = df["y"].map(YN_DICT)
    X = df.drop(["y"], axis=1)
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    if balance_labels:
        encoder = SMOTE(random_state=42)
        X_train, y_train = encoder.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pass
