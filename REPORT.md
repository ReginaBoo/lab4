# Отчёт к лабораторной работе №4: Ансамбли и полносвязные нейронные сети

## Данные и подготовка
- **Источник данных:** `german.csv` (разделитель `;`, 1000 наблюдений, бинарная целевая переменная `Creditability`).
- **Разбиение:** стратифицированный `train_test_split` с параметрами `test_size=0.2` (800 строк в `train`, 200 в `test`) и `random_state=42`. Целевая переменная тестовой выборки (`test_target`) не использовалась при обучении, генерации признаков, кросс-валидации или выборе модели.
- **Признаки:** числовые (`Duration_of_Credit_monthly`, `Credit_Amount`, `Instalment_per_cent`, `Duration_in_Current_address`, `Age_years`, `No_of_Credits_at_this_Bank`, `No_of_dependents`) и категориальные коды (`Account_Balance`, `Payment_Status_of_Previous_Credit`, `Purpose`, `Value_Savings_Stocks`, `Length_of_current_employment`, `Sex_Marital_Status`, `Guarantors`, `Most_valuable_available_asset`, `Concurrent_Credits`, `Type_of_apartment`, `Occupation`, `Telephone`, `Foreign_Worker`).
- **Доменные признаки (только по `X`, без `y`):**
  - `credit_per_month` = `Credit_Amount / Duration_of_Credit_monthly`;
  - `age_to_credit_duration` = `Age_years / Duration_of_Credit_monthly`;
  - `credit_to_age` = `Credit_Amount / Age_years`.
- **Предобработка:** `SimpleImputer` (`median` для числовых, `most_frequent` для категориальных). Для древовидных моделей (`RandomForest`, `ExtraTrees`, `GradientBoosting`) применяется `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, а для `StackingClassifier` — `StandardScaler` для числовых признаков и `OneHotEncoder(handle_unknown="ignore")` для категориальных внутри `ColumnTransformer` и `Pipeline`.

## Гипотезы и эксперименты
Оценка проводилась на 5-блочной стратифицированной кросс-валидации (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) внутри `train` и на отложенной тестовой выборке `test`:

1. **Бэггинг и случайные деревья (`RandomForestClassifier`, `ExtraTreesClassifier`):** ограничение глубины (`max_depth=8` и `max_depth=10`), `min_samples_leaf=3..4`, `max_features="sqrt"` и `class_weight="balanced"` проверялись для снижения дисперсии и учёта баланса классов.
2. **Градиентный бустинг (`GradientBoostingClassifier`):** проверялась гипотеза о том, что последовательная коррекция ошибок при `learning_rate=0.04`, `max_depth=3`, `min_samples_leaf=5` и `subsample=0.85` даст сильное качество и высокую точность классификации.
3. **Стекинг (`StackingClassifier`):** базовые модели `RandomForest`, `ExtraTrees` и `GradientBoosting` объединялись через мета-модель `LogisticRegression(C=0.5, max_iter=1000)` (`cv=5`, `stack_method="predict_proba"`), чтобы сочетать сильные стороны разных ансамблей.

## Метрики
Критерий выбора модели — **максимальный средний ROC-AUC на 5-fold StratifiedKFold внутри `train`** (без использования `test_target`), дополнительно на `test` отслеживались `ROC-AUC`, `Accuracy`, `Precision`, `Recall` и `F1`.

| Модель | CV ROC-AUC (train, 5-fold) | Test ROC-AUC | Accuracy | Precision | Recall | F1 | Комментарий |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| RandomForest | `0.7860 ± 0.0313` | 0.8325 | 0.7650 | **0.8605** | 0.7929 | 0.8253 | Высокий Precision, стабильный CV |
| ExtraTrees | `0.7860 ± 0.0356` | 0.8312 | 0.7700 | 0.8561 | 0.8071 | 0.8309 | Сопоставим с RandomForest по CV и Test ROC-AUC |
| GradientBoosting | `0.7790 ± 0.0299` | **0.8468** | **0.8200** | 0.8333 | **0.9286** | **0.8784** | Лучшие Accuracy, Recall и F1 на test |
| **Stacking (RF + ET + GB -> LogReg)** | **0.7914 ± 0.0285** | 0.8463 | 0.7850 | 0.8089 | 0.9071 | 0.8552 | **Выбрана по CV ROC-AUC** (`0.7914`), сильный Test ROC-AUC (`0.8463`) |

## Выбранная модель
Выбран ансамбль **`StackingClassifier`** (`RandomForestClassifier` + `ExtraTreesClassifier` + `GradientBoostingClassifier` с мета-моделью `LogisticRegression(C=0.5)`). Он показал наилучший средний `ROC-AUC` на кросс-валидации внутри `train` (`0.791369 ± 0.028545`) при наименьшем разбросе среди кандидатов и высокий `Test ROC-AUC` (`0.84631`, `F1 = 0.855219`).

## Выводы и ограничения
- **Что улучшило результат:** добавление доменных отношений (`credit_per_month`, `age_to_credit_duration`, `credit_to_age`), регуляризация глубины/листьев деревьев и мета-агрегация разнородных древовидных моделей через регуляризованную логистическую регрессию.
- **Что ухудшило или не дало прироста:** отдельные древовидные модели с `class_weight="balanced"` в `RandomForest`/`ExtraTrees` дали более высокий `Precision` (`0.856–0.860`), но более низкий `Recall` (`0.793–0.807`) и `F1` (`0.825–0.831`) при пороге `0.5` по сравнению со стекингом и градиентным бустингом.
- **Ограничения:** выборка из 1000 объектов относительно невелика, поэтому различия в десятые/сотые доли между моделями на одном тесте чувствительны к разбиению; порог классификации `0.5` не оптимизирован под асимметричную стоимость ошибок кредитного скоринга.

## Воспроизводимость
- Фиксированный seed: `RANDOM_SEED = 42` используется в `train_test_split`, `StratifiedKFold`, моделях и `EnsembleInput`.
- Запуск сравнения и обновление `experiment_results.json`:
  ```bash
  python run_experiments.py