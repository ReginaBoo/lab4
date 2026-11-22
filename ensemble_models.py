from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


CATEGORICAL_COLS = [
    "Account_Balance",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Sex_Marital_Status",
    "Guarantors",
    "Most_valuable_available_asset",
    "Concurrent_Credits",
    "Type_of_apartment",
    "Occupation",
    "Telephone",
    "Foreign_Worker",
]

NUMERIC_COLS = [
    "Duration_of_Credit_monthly",
    "Credit_Amount",
    "Instalment_per_cent",
    "Duration_in_Current_address",
    "Age_years",
    "No_of_Credits_at_this_Bank",
    "No_of_dependents",
]


def _add_domain_features(X: Any) -> pd.DataFrame:
    """
    Добавляет признаки, не используя целевую переменную.

    Признаки:
    - сумма кредита на один месяц;
    - отношение возраста к длительности кредита;
    - сумма кредита относительно возраста.
    """
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        arr = np.asarray(X)
        columns = [f"f_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=columns)

    for col in df.columns:
        if col in CATEGORICAL_COLS or col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {
        "Credit_Amount",
        "Duration_of_Credit_monthly",
    }.issubset(df.columns):
        duration = df["Duration_of_Credit_monthly"].replace(0, np.nan)

        df["credit_per_month"] = (
            df["Credit_Amount"] / duration
        ).replace([np.inf, -np.inf], np.nan)

    if {
        "Age_years",
        "Duration_of_Credit_monthly",
    }.issubset(df.columns):
        duration = df["Duration_of_Credit_monthly"].replace(0, np.nan)

        df["age_to_credit_duration"] = (
            df["Age_years"] / duration
        ).replace([np.inf, -np.inf], np.nan)

    if {
        "Credit_Amount",
        "Age_years",
    }.issubset(df.columns):
        age = df["Age_years"].replace(0, np.nan)

        df["credit_to_age"] = (
            df["Credit_Amount"] / age
        ).replace([np.inf, -np.inf], np.nan)

    return df


def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Создаёт предобработку для числовых и категориальных признаков.
    """

    categorical = [
        col for col in CATEGORICAL_COLS
        if col in df.columns
    ]

    numeric = [
        col
        for col in df.columns
        if col not in categorical
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical,
            ),
        ],
        remainder="drop",
    )


def _build_stacking_model(seed: int) -> StackingClassifier:
    """
    Создаёт Stacking-модель, используемую в экспериментах.
    """

    base_models = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "gb",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.04,
                max_depth=3,
                min_samples_leaf=5,
                subsample=0.85,
                random_state=seed,
            ),
        ),
    ]

    return StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(
            C=0.5,
            max_iter=1000,
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """
    Обучает итоговый Stacking только на train-части.

    test_target используется исключительно для финальной оценки.
    """

    seed = int(data.random_seed)

    train_df = _add_domain_features(data.train_features)
    test_df = _add_domain_features(data.test_features)

    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    preprocessor = _build_preprocessor(train_df)
    model = _build_stacking_model(seed)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(train_df, y_train)

    probabilities = clf.predict_proba(test_df)[:, 1]
    predictions = clf.predict(test_df)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    return EnsembleResult(
        model_name=(
            "Stacking: RandomForest + ExtraTrees + "
            "GradientBoosting -> LogisticRegression"
        ),
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )