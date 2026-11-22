"""Скрипт воспроизводимого сравнения моделей на german.csv."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ensemble_models import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    _add_domain_features,
    run_ensemble_experiment,
)
from grader_contracts.ensembles import EnsembleInput

RANDOM_SEED = 42


def main() -> None:
    df = pd.read_csv("german.csv", sep=";")
    target_col = "Creditability"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    train_fe = _add_domain_features(X_train)
    test_fe = _add_domain_features(X_test)

    cat_cols = [c for c in CATEGORICAL_COLS if c in train_fe.columns]
    num_cols = [c for c in train_fe.columns if c not in cat_cols]

    tree_prep = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "enc",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    linear_mlp_prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    candidates = {
        "RandomForest (n_estimators=300, max_depth=8)": Pipeline(
            [
                ("prep", tree_prep),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "ExtraTrees (n_estimators=400, max_depth=10)": Pipeline(
            [
                ("prep", tree_prep),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=4,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting (lr=0.05, n_estimators=150, depth=3)": Pipeline(
            [
                ("prep", tree_prep),
                (
                    "clf",
                    GradientBoostingClassifier(
                        learning_rate=0.05,
                        n_estimators=150,
                        max_depth=3,
                        subsample=0.8,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting (lr=0.05, iter=200, leaf=15)": Pipeline(
            [
                ("prep", tree_prep),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "MLPClassifier (64, 32, alpha=1.0, early_stopping=True)": Pipeline(
            [
                ("prep", linear_mlp_prep),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        alpha=1.0,
                        max_iter=500,
                        early_stopping=True,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Stacking (HGB + RF + MLP -> LogReg)": Pipeline(
            [
                ("prep", linear_mlp_prep),
                (
                    "clf",
                    StackingClassifier(
                        estimators=[
                            (
                                "hgb",
                                HistGradientBoostingClassifier(
                                    learning_rate=0.05,
                                    max_iter=150,
                                    max_leaf_nodes=15,
                                    min_samples_leaf=20,
                                    l2_regularization=1.0,
                                    random_state=RANDOM_SEED,
                                ),
                            ),
                            (
                                "rf",
                                RandomForestClassifier(
                                    n_estimators=250,
                                    max_depth=7,
                                    min_samples_leaf=5,
                                    random_state=RANDOM_SEED,
                                ),
                            ),
                            (
                                "mlp",
                                MLPClassifier(
                                    hidden_layer_sizes=(64, 32),
                                    alpha=1.0,
                                    max_iter=400,
                                    early_stopping=True,
                                    random_state=RANDOM_SEED,
                                ),
                            ),
                        ],
                        final_estimator=LogisticRegression(max_iter=1000),
                        cv=5,
                    ),
                ),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    print("=== CV ROC-AUC on TRAIN & Test Metrics ===")
    for name, pipe in candidates.items():
        cv_scores = cross_val_score(
            pipe, train_fe, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        pipe.fit(train_fe, y_train)
        proba = pipe.predict_proba(test_fe)[:, 1]
        pred = pipe.predict(test_fe)
        print(
            f"{name}\n"
            f"  CV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n"
            f"  Test ROC-AUC: {roc_auc_score(y_test, proba):.4f} | "
            f"Acc: {accuracy_score(y_test, pred):.4f} | "
            f"Prec: {precision_score(y_test, pred, zero_division=0):.4f} | "
            f"Rec: {recall_score(y_test, pred, zero_division=0):.4f} | "
            f"F1: {f1_score(y_test, pred, zero_division=0):.4f}"
        )

    # Проверка контрактной функции
    contract_input = EnsembleInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
        random_seed=RANDOM_SEED,
    )
    res = run_ensemble_experiment(contract_input)

    results_payload = {
        "random_seed": RANDOM_SEED,
        "test_size": 0.2,
        "stratified": True,
        "selected_model": res.model_name,
        "selection_criterion": "roc_auc (на валидации/тесте при контроле переобучения и баланса F1)",
        "metrics": {
            "roc_auc": round(res.roc_auc, 6),
            "accuracy": round(res.accuracy, 6),
            "precision": round(res.precision, 6),
            "recall": round(res.recall, 6),
            "f1": round(res.f1, 6),
        },
    }

    Path("experiment_results.json").write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSaved experiment_results.json")


if __name__ == "__main__":
    main()