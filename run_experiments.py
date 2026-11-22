"""Воспроизводимое сравнение ансамблей на german.csv."""

from __future__ import annotations

import json
from pathlib import Path

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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from ensemble_models import (
    CATEGORICAL_COLS,
    _add_domain_features,
    run_ensemble_experiment,
)
from grader_contracts.ensembles import EnsembleInput


RANDOM_SEED = 42
TEST_SIZE = 0.2


def build_tree_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
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

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_linear_preprocessor(
    df: pd.DataFrame,
) -> ColumnTransformer:

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

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_candidates(
    train_df: pd.DataFrame,
) -> dict[str, Pipeline]:

    tree_preprocessor = build_tree_preprocessor(train_df)
    linear_preprocessor = build_linear_preprocessor(train_df)

    candidates = {}

    candidates[
        "RandomForest"
    ] = Pipeline(
        steps=[
            ("prep", tree_preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    candidates[
        "ExtraTrees"
    ] = Pipeline(
        steps=[
            ("prep", tree_preprocessor),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=400,
                    max_depth=10,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    candidates[
        "GradientBoosting"
    ] = Pipeline(
        steps=[
            ("prep", tree_preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.04,
                    max_depth=3,
                    min_samples_leaf=5,
                    subsample=0.85,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    candidates[
        "Stacking"
    ] = Pipeline(
        steps=[
            ("prep", linear_preprocessor),
            (
                "model",
                StackingClassifier(
                    estimators=[
                        (
                            "rf",
                            RandomForestClassifier(
                                n_estimators=250,
                                max_depth=8,
                                min_samples_leaf=4,
                                max_features="sqrt",
                                class_weight="balanced",
                                random_state=RANDOM_SEED,
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
                                random_state=RANDOM_SEED,
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
                                random_state=RANDOM_SEED,
                            ),
                        ),
                    ],
                    final_estimator=LogisticRegression(
                        C=0.5,
                        max_iter=1000,
                    ),
                    stack_method="predict_proba",
                    cv=5,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return candidates


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> dict:

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    return {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "test_roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
    }


def main() -> None:

    df = pd.read_csv(
        "german.csv",
        sep=";",
    )

    target = "Creditability"

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    X_train = _add_domain_features(X_train)
    X_test = _add_domain_features(X_test)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    candidates = build_candidates(X_train)

    results = {}

    print("СРАВНЕНИЕ МОДЕЛЕЙ")

    for name, model in candidates.items():

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            cv,
        )

        results[name] = metrics

        print(f"\n{name}")
        print(
            f"CV ROC-AUC: "
            f"{metrics['cv_roc_auc_mean']:.4f} "
            f"+/- {metrics['cv_roc_auc_std']:.4f}"
        )

        print(
            f"Test ROC-AUC: "
            f"{metrics['test_roc_auc']:.4f}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1: "
            f"{metrics['f1']:.4f}"
        )

    # Критерий выбора использует ТОЛЬКО train/CV.
    selected_name = max(
        results,
        key=lambda name: results[name][
            "cv_roc_auc_mean"
        ],
    )

    print(
        f"Лучшая модель по CV ROC-AUC: "
        f"{selected_name}"
    )

    selected_model = candidates[selected_name]

    selected_model.fit(
        X_train,
        y_train,
    )

    test_probabilities = (
        selected_model.predict_proba(X_test)[:, 1]
    )

    test_predictions = (
        selected_model.predict(X_test)
    )

    final_metrics = {
        "roc_auc": float(
            roc_auc_score(
                y_test,
                test_probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_test,
                test_predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
    }

    # Проверяем именно контрактную функцию.
    contract_input = EnsembleInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
        random_seed=RANDOM_SEED,
    )

    contract_result = run_ensemble_experiment(
        contract_input
    )

    payload = {
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "stratified": True,
        "selected_model": (
            contract_result.model_name
        ),
        "selection_criterion": (
            "maximum mean ROC-AUC on 5-fold "
            "StratifiedKFold using train only"
        ),
        "cv_results": results,
        "metrics": {
            key: round(value, 6)
            for key, value in final_metrics.items()
        },
    }

    Path(
        "experiment_results.json"
    ).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nРезультаты сохранены:")
    print("experiment_results.json")


if __name__ == "__main__":
    main()