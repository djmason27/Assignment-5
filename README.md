# Assignment 5 Scikit Learn Regression

**Author:** Donovan Mason

## Purpose

This project uses scikit-learn's built-in diabetes dataset to compare three regression models: linear regression, random forest regression, and support vector regression (SVR). The target is a quantitative measure of disease progression one year after baseline, not a yes/no diabetes diagnosis. The dataset has 442 samples and 10 features: age, sex, body mass index, average blood pressure, and six blood serum measurements.

## Running the project

Use Python 3 with scikit-learn installed, such as a course Anaconda environment. If needed, install the dependency with:

```bash
python -m pip install -r requirements.txt
python regression_project.py
```

Run the second command from the project folder. No separate dataset download is necessary. The program creates or overwrites `results.json` and `model_comparison.md` in that folder. The included versions contain the actual results of the tested run. The parameter searches may take a minute depending on the computer.

## Data division and evaluation

1. Load the raw diabetes features with `load_diabetes(scaled=False)`.
2. Split once into 80% training and 20% testing using `random_state=42`: 353 training samples and 89 test samples.
3. Use five-fold cross-validation **within the training set** to choose model settings by the lowest validation mean squared error.
4. Refit each selected model on all training samples.
5. Evaluate each model on the same untouched test set using explained variance, mean squared error (MSE), mean absolute error (MAE), and R2.

The 80/20 ratio is based on the regression lab's specified split. The Machine Learning lecture itself was not supplied; if it specifies a different division, update `test_size` to match it before running. Cross-validation supplies a validation stage without using the held-out test set for tuning. No split seed or settings were changed after inspecting test performance.

Higher explained variance and R2 are better; lower MSE and MAE are better. MSE and MAE are measured in squared target units and target units, respectively. R2 is not classification accuracy. Explained variance can differ from R2 because it does not penalize a systematic prediction offset in the same way.

## Model implementation

- **Linear regression:** a `Pipeline` fits `StandardScaler` on training features followed by `LinearRegression`. This is an interpretable baseline with no searched parameters.
- **Random forest:** `RandomForestRegressor` uses 200 trees and a fixed random seed. The training search compares maximum depths of 3, 5, or unlimited; minimum leaf sizes of 2 or 5; and feature fractions of 0.7 or 1.0. Scaling is unnecessary for this tree model.
- **SVR:** a pipeline standardizes features and fits an RBF-kernel `SVR`. `TransformedTargetRegressor` standardizes training targets and converts predictions back to their original units. The search compares C values of 0.1, 1, or 10; gamma values of 0.01, 0.1, or `scale`; and epsilon values of 0.05 or 0.1. Epsilon is in standardized target units.

Scalers are inside the estimators, so they are fitted separately in each cross-validation training fold. Test data are transformed using training statistics. The parameter search uses only scikit-learn; no additional external libraries are imported.

## Class design

`DiabetesRegression` keeps the dataset, split, estimators, and results together in one class. Separating the workflow into methods makes it easier to follow and prevents evaluation from being mixed into model training.

### Every class attribute

| Attribute | Purpose |
| --- | --- |
| `test_size` | Fraction of samples held out for testing; defaults to 0.2. |
| `random_state` | Reproducible seed for the split, shuffled validation folds, and forest. |
| `dataset` | Dataset object containing data, target, and feature names. |
| `X`, `y` | Full feature matrix and target vector. |
| `X_train`, `X_test` | Feature matrices for training and testing. |
| `y_train`, `y_test` | Corresponding training and testing targets. |
| `models` | Dictionary of the three estimators; replaced with fitted best estimators after searching. |
| `searches` | Dictionary of fitted `GridSearchCV` objects, with selected parameters and validation scores. |
| `results` | Dictionary of test metrics, validation MSE, and selected parameters for each model. |

### Every class method

| Method | Purpose |
| --- | --- |
| `__init__(test_size=0.2, random_state=42)` | Validate the test fraction and initialize all attributes. |
| `load_data()` | Load the built-in dataset and set `dataset`, `X`, and `y`. |
| `split_data()` | Split features and targets together; require loaded data. |
| `create_models()` | Define all three model estimators. |
| `train_models()` | Search settings on training folds and refit chosen estimators. |
| `evaluate_models()` | Predict test targets and calculate metrics; return the results dictionary. |
| `save_results(output_dir)` | Save JSON results and a Markdown comparison; require evaluation results. |
| `run(output_dir)` | Call all workflow methods in order. |

`run()` is the normal entry point. The `if __name__ == '__main__'` block runs it only when the file is executed directly. Methods report clear errors if loading, splitting, training, or evaluation prerequisites are missing. Call `create_models()` again before repeating a training search manually.

## Performance explanation

See **[model_comparison.md](model_comparison.md)** for the required brief paragraph, the measured metrics for all three models, and the selected settings. `results.json` also records the scikit-learn version used. The report distinguishes the best model by training validation from the best model observed on the test set.

## Limitations

- This is a small educational dataset and a regression exercise, not a medical diagnostic system.
- A single held-out split may not represent future patients. There is no external validation or fairness audit.
- The searches are deliberately limited and do not establish globally optimal settings or guarantee extra credit.
- Linear regression assumes an additive linear relationship. Forest and SVR models are more difficult to interpret.
- The default split must leave enough training samples for five-fold validation; extreme custom test fractions may fail.
- Results may vary slightly with scikit-learn versions. Run the project in the course environment to reproduce them.
- Existing report files in the output directory are overwritten.

## Files and dependencies

- `regression_project.py`: complete implementation.
- `README.md`: purpose, usage, implementation, class documentation, and limitations.
- `model_comparison.md`: measured comparison and explanation.
- `results.json`: measured metrics and reproducibility information.
- `requirements.txt`: scikit-learn dependency.
- `AI_CHAT_LOG.md`: assignment prompt, relevant assistant messages, and disclosure of AI assistance.

The code imports **only scikit-learn and Python's standard library** (`json` and `pathlib`). Scikit-learn's own dependencies are installed automatically. The class and support utilities are documented above; verify that your instructor's lecture covers the evaluation approach you submit.

## Sources

- [Diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [Pipelines and target transformations](https://scikit-learn.org/stable/modules/compose.html)

## Submission

Create a **public** GitHub repository, upload all the files from this folder, and submit the repository URL. Upload the extracted files, not the ZIP archive. Include the AI chat log as required by the assignment. If you use AI for further changes, append those exchanges to the log.
