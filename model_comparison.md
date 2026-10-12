# Model comparison

| Model | Explained variance | MSE | MAE | R2 | Training CV MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear regression | 0.4555 | 2900.1936 | 42.7941 | 0.4526 | 3074.1421 |
| Random forest | 0.4676 | 2820.8966 | 43.0344 | 0.4676 | 3323.4412 |
| Support vector regression | 0.4797 | 2812.7179 | 41.9565 | 0.4691 | 3089.9563 |

## Brief explanation

On this held-out test set, support vector regression had the lowest mean squared error (2812.7179) and the highest R2 score (0.4691). Its explained variance score was 0.4797, and its mean absolute error was 41.9565. Lower MSE and MAE indicate smaller prediction errors, while higher R2 and explained variance indicate better fit. Its R2 corresponds to about 46.9% of the test targets' variation being explained. The model selected by the lowest training cross-validation MSE was linear regression. Test results describe performance on these 89 held-out samples and were not used to choose settings or change the split. Linear regression is simpler to interpret; random forests can capture nonlinear relationships but are less transparent; SVR depends on scaling and parameter choices. One split cannot prove that a model is best on every dataset or ready for medical use.


SVR has a test MSE of 2812.7179, compared with 2820.8966 for the random forest and 2900.1936 for linear regression. Its advantage over the forest is about 0.3%, so this test split does not establish a decisive overall winner. SVR also has the lowest MAE (41.9565) and highest explained variance (0.4797). Linear regression has the strongest training validation result and offers the simplest interpretation; I would treat it as a strong baseline rather than dismiss it based on this one test split.

## Selected settings

- Linear regression: `{}`
- Random forest: `{'max_depth': None, 'max_features': 0.7, 'min_samples_leaf': 5}`
- Support vector regression: `{'regressor__model__C': 1, 'regressor__model__epsilon': 0.05, 'regressor__model__gamma': 0.01}`

All models used 353 training samples and the same 89 test samples. Settings were chosen by five-fold training-only cross-validation. Preprocessing was fitted inside each training fold. Results can vary by software version.
