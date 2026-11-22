# Отчёт к лабораторной работе №4: Ансамбли и полносвязные нейронные сети

## Данные и подготовка
- **Источник данных:** датасет `german.csv` (разделитель `;`, 1000 записей, 20 исходных признаков, бинарная целевая переменная `Creditability`).
- **Разбиение выборки:** выполнено стратифицированное разбиение `train_test_split` с параметрами `test_size=0.2` (800 объектов в `train`, 200 в `test`) и `random_state=42`, чтобы сохранить исходное распределение классов в обеих выборках. Целевая переменная тестовой выборки (`test_target`) нигде не использовалась при обучении, генерации признаков, кросс-валидации или выборе модели.
- **Типы признаков:**
  - числовые: `Duration_of_Credit_monthly`, `Credit_Amount`, `Instalment_per_cent`, `Duration_in_Current_address`, `Age_years`, `No_of_Credits_at_this_Bank`, `No_of_dependents`;
  - категориальные коды: `Account_Balance`, `Payment_Status_of_Previous_Credit`, `Purpose`, `Value_Savings_Stocks`, `Length_of_current_employment`, `Sex_Marital_Status`, `Guarantors`, `Most_valuable_available_asset`, `Concurrent_Credits`, `Type_of_apartment`, `Occupation`, `Telephone`, `Foreign_Worker`.
- **Доменные признаки (вычисляются только по признакам `X`, без обращения к `y`):**
  - `credit_per_month` = `Credit_Amount / Duration_of_Credit_monthly` (ежемесячная кредитная нагрузка);
  - `age_to_credit_duration` = `Age_years / Duration_of_Credit_monthly` (отношение возраста заёмщика к длительности кредита);
  - `credit_to_age` = `Credit_Amount / Age_years` (отношение суммы кредита к возрасту заёмщика).
- **Предобработка и масштабирование:**
  - пропуски и возможные бесконечные значения при делении заменяются на `NaN` и заполняются с помощью `SimpleImputer` (`strategy="median"` для числовых признаков, `strategy="most_frequent"` для категориальных);
  - для древовидных ансамблей (`RandomForest`, `ExtraTrees`, `GradientBoosting`) используется `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`;
  - для стекинга (`StackingClassifier`) применяется `StandardScaler` для числовых признаков и `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` для категориальных внутри `ColumnTransformer` и `Pipeline`.

## Гипотезы и эксперименты
Все модели оценивались на 5-блочной стратифицированной кросс-валидации (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) внутри `train`, после чего обученные на полном `train` модели проверялись на отложенном `test`:

1. **Бэггинг и случайные подпространства (`RandomForestClassifier`, `ExtraTreesClassifier`):** проверялась гипотеза о том, что ансамблирование большого числа деревьев (`n_estimators=400`) с ограничением глубины (`max_depth=8` и `max_depth=10`), `min_samples_leaf=3..4`, `max_features="sqrt"` и балансировкой весов (`class_weight="balanced"`) снизит дисперсию предсказаний на табличных признаках скоринга.
2. **Градиентный бустинг (`GradientBoostingClassifier`):** проверялась гипотеза о том, что последовательная коррекция ошибок при малом шаге обучения (`learning_rate=0.04`), неглубоких деревьях (`max_depth=3`, `min_samples_leaf=5`) и стохастическом подвыборе (`subsample=0.85`) позволит лучше уловить нелинейные взаимодействия признаков.
3. **Стекинг разнородных ансамблей (`StackingClassifier`):** проверялась гипотеза о том, что объединение вероятностных прогнозов (`stack_method="predict_proba"`, `cv=5`) трёх различных моделей (`RandomForest`, `ExtraTrees` и `GradientBoosting`) с помощью мета-классификатора `LogisticRegression(C=0.5, max_iter=1000)` повысит качество ранжирования и устойчивость обобщения.

## Метрики
Основной критерий отбора модели — **максимальный средний ROC-AUC на 5-fold StratifiedKFold внутри `train`** (без утечки из `test_target`), дополнительно на `test` оценивались `ROC-AUC`, `Accuracy`, `Precision`, `Recall` и `F1` при стандартном пороге `0.5`.

| Модель | CV ROC-AUC (train, 5-fold) | Test ROC-AUC | Accuracy | Precision | Recall | F1 | Комментарий / критерий выбора |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| RandomForest (`n_estimators=400`, `max_depth=8`) | `0.7860 ± 0.0313` | 0.8325 | 0.7650 | **0.8605** | 0.7929 | 0.8253 | Наивысший Precision на test, стабильный CV |
| ExtraTrees (`n_estimators=400`, `max_depth=10`) | `0.7860 ± 0.0356` | 0.8312 | 0.7700 | 0.8561 | 0.8071 | 0.8309 | Сопоставимое качество с RandomForest |
| GradientBoosting (`lr=0.04`, `n_estimators=150`, `depth=3`) | `0.7790 ± 0.0299` | **0.8468** | **0.8200** | 0.8333 | **0.9286** | **0.8784** | Лучшие Accuracy, Recall и F1 на test |
| **Stacking (`RF + ExtraTrees + GB -> LogReg`)** | **0.7914 ± 0.0285** | 0.8463 | 0.7850 | 0.8089 | 0.9071 | 0.8552 | **Выбрана по критерию максимума CV ROC-AUC** (`0.7914`) и минимальному разбросу (`±0.0285`) |

## Выбранная модель
В контрактной функции `run_ensemble_experiment` (`ensemble_models.py`) выбрана модель **`Stacking: RandomForest + ExtraTrees + GradientBoosting -> LogisticRegression`**.

**Обоснование выбора:**
- заняла первое место по целевому критерию отбора на кросс-валидации внутри `train` (`CV ROC-AUC = 0.791369`) и показала наименьший стандартный разброс между фолдами (`±0.028545`);
- продемонстрировала высокое итоговое качество на отложенной тестовой выборке: `ROC-AUC = 0.84631`, `Accuracy = 0.785`, `Precision = 0.808917`, `Recall = 0.907143`, `F1 = 0.855219` (значительно выше порога `0.80`);
- объединяет разные механизмы снижения ошибки: снижение дисперсии через бэггинг/рандомизированные деревья (`RandomForest`, `ExtraTrees`) и снижение смещения через последовательный градиентный бустинг (`GradientBoosting`), взвешенные регуляризованной логистической регрессией.

## Выводы и ограничения
- **Что улучшило результат:** добавление относительных доменных признаков (`credit_per_month`, `age_to_credit_duration`, `credit_to_age`), ограничение глубины деревьев и числа объектов в листе против переобучения, а также мета-объединение нескольких ансамблей через `StackingClassifier`.
- **Что ухудшило или не дало прироста:** использование `class_weight="balanced"` в одиночных `RandomForest` и `ExtraTrees` сместило предсказания в сторону более строгого отбора положительного класса (`Precision = 0.856–0.860`, но `Recall = 0.793–0.807` и более низкий `F1 = 0.825–0.831` при пороге `0.5`).
- **Ограничения:**
  - объём датасета (`1000` строк всего, `200` в тестовой части) ограничивает статистическую мощность сравнения моделей на одном сплите — разница в `Test ROC-AUC` между `GradientBoosting` (`0.8468`) и `Stacking` (`0.8463`) находится в пределах доли процента;
  - порог бинаризации `0.5` выбран по умолчанию; в прикладном кредитном скоринге порог следует калибровать с учётом разной стоимости выдачи дефолтного кредита и отказа надёжному заёмщику.

## Воспроизводимость
- **Фиксированный seed:** `RANDOM_SEED = 42` передаётся в `train_test_split`, `StratifiedKFold`, все базовые модели ансамблей и структуру `EnsembleInput`.
- **Запуск полного эксперимента и сохранение `experiment_results.json`:**
  ```bash
  python run_experiments.py