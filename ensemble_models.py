"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

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
    """Формирует доменные признаки без использования целевой переменной."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        arr = np.asarray(X)
        cols = [f"f_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)

    # Приводим известные столбцы к числовому типу, если они присутствуют
    for col in df.columns:
        if col in CATEGORICAL_COLS or col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Credit_Amount" in df.columns and "Duration_of_Credit_monthly" in df.columns:
        duration = df["Duration_of_Credit_monthly"].replace(0, np.nan)
        df["credit_per_month"] = (df["Credit_Amount"] / duration).fillna(0.0)

    if "Age_years" in df.columns and "Duration_of_Credit_monthly" in df.columns:
        duration = df["Duration_of_Credit_monthly"].replace(0, np.nan)
        df["age_to_duration"] = (df["Age_years"] / duration).fillna(0.0)

    if "Credit_Amount" in df.columns and "Age_years" in df.columns:
        age = df["Age_years"].replace(0, np.nan)
        df["credit_to_age"] = (df["Credit_Amount"] / age).fillna(0.0)

    return df


def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Строит ColumnTransformer под доступные столбцы датафрейма."""
    present_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    present_num = [
        c
        for c in df.columns
        if c not in present_cat and pd.api.types.is_numeric_dtype(df[c])
    ]

    transformers = []
    if present_num:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", num_pipe, present_num))

    if present_cat:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, present_cat))

    if not transformers:
        fallback_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[("fallback", fallback_pipe, list(df.columns))],
            remainder="drop",
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    seed = int(data.random_seed)

    train_df = _add_domain_features(data.train_features)
    test_df = _add_domain_features(data.test_features)

    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    preprocessor = _build_preprocessor(train_df)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(train_df, y_train)

    proba_all = clf.predict_proba(test_df)
    probabilities = proba_all[:, 1]
    predictions = clf.predict(test_df)

    roc_auc = float(roc_auc_score(y_test, probabilities))
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))

    return EnsembleResult(
        model_name="HistGradientBoosting + DomainFeatures",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=roc_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )