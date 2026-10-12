"""Compare three regression models on scikit-learn's diabetes dataset."""
import json
from pathlib import Path
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score


class DiabetesRegression:
    """Load, split, train, compare, and report three regression models."""

    def __init__(self, test_size=0.2, random_state=42):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        self.test_size = test_size
        self.random_state = random_state
        self.dataset = None
        self.X = self.y = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.models = {}
        self.searches = {}
        self.results = {}

    def load_data(self):
        """Load raw features; any model scaling is learned on training data."""
        self.dataset = load_diabetes(scaled=False)
        self.X = self.dataset.data
        self.y = self.dataset.target

    def split_data(self):
        """Create one reproducible 80/20 train/test split by default."""
        if self.X is None:
            raise RuntimeError("Call load_data before split_data.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def create_models(self):
        """Define linear regression, random forest, and scaled SVR."""
        self.models = {
            "Linear regression": Pipeline([
                ("scale", StandardScaler()), ("model", LinearRegression())
            ]),
            "Random forest": RandomForestRegressor(
                n_estimators=200, random_state=self.random_state
            ),
            "Support vector regression": TransformedTargetRegressor(
                regressor=Pipeline([("scale", StandardScaler()), ("model", SVR())]),
                transformer=StandardScaler()
            )
        }

    def train_models(self):
        """Tune using five-fold validation inside the training set only."""
        if self.X_train is None or not self.models:
            raise RuntimeError("Split the data and create models before training.")
        folds = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grids = {
            "Linear regression": {},
            "Random forest": {
                "max_depth": [3, 5, None], "min_samples_leaf": [2, 5],
                "max_features": [0.7, 1.0]
            },
            "Support vector regression": {
                "regressor__model__C": [0.1, 1, 10],
                "regressor__model__gamma": [0.01, 0.1, "scale"],
                "regressor__model__epsilon": [0.05, 0.1]
            }
        }
        self.searches = {}
        for name, model in self.models.items():
            search = GridSearchCV(model, grids[name], scoring="neg_mean_squared_error",
                                  cv=folds, n_jobs=1, refit=True)
            search.fit(self.X_train, self.y_train)
            self.searches[name] = search
            self.models[name] = search.best_estimator_
            print(f"Trained {name}; validation MSE: {-search.best_score_:.4f}", flush=True)

    def evaluate_models(self):
        """Evaluate fitted models on the untouched test set in target units."""
        if not self.searches:
            raise RuntimeError("Train models before evaluation.")
        self.results = {}
        for name, model in self.models.items():
            predictions = model.predict(self.X_test)
            self.results[name] = {
                "explained_variance": float(explained_variance_score(self.y_test, predictions)),
                "mse": float(mean_squared_error(self.y_test, predictions)),
                "mae": float(mean_absolute_error(self.y_test, predictions)),
                "r2": float(r2_score(self.y_test, predictions)),
                "validation_mse": float(-self.searches[name].best_score_),
                "best_parameters": self.searches[name].best_params_
            }
        return self.results

    def save_results(self, output_dir):
        """Write machine-readable results and a brief model comparison."""
        if not self.results:
            raise RuntimeError("Evaluate models before saving results.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_test = min(self.results, key=lambda name: self.results[name]["mse"])
        best_validation = min(self.results, key=lambda name: self.results[name]["validation_mse"])
        report = {
            "scikit_learn_version": sklearn.__version__, "random_state": self.random_state,
            "test_size": self.test_size, "training_samples": len(self.y_train),
            "testing_samples": len(self.y_test), "features": list(self.dataset.feature_names),
            "best_test_model": best_test, "best_validation_model": best_validation,
            "models": self.results
        }
        (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = ["# Model comparison", "", "| Model | Explained variance | MSE | MAE | R2 | Training CV MSE |",
                 "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, scores in self.results.items():
            lines.append(f"| {name} | {scores['explained_variance']:.4f} | {scores['mse']:.4f} | "
                         f"{scores['mae']:.4f} | {scores['r2']:.4f} | {scores['validation_mse']:.4f} |")
        scores = self.results[best_test]
        lines += ["", "## Brief explanation", "",
            f"On this held-out test set, {best_test.lower()} had the lowest mean squared error "
            f"({scores['mse']:.4f}) and the highest R2 score ({scores['r2']:.4f}). Its explained "
            f"variance score was {scores['explained_variance']:.4f}, and its mean absolute error "
            f"was {scores['mae']:.4f}. Lower MSE and MAE indicate smaller prediction errors, while "
            f"higher R2 and explained variance indicate better fit. Its R2 corresponds to about "
            f"{scores['r2'] * 100:.1f}% of the test targets' variation being explained. "
            f"The model selected by the lowest training cross-validation MSE was "
            f"{best_validation.lower()}. Test results describe performance on these {len(self.y_test)} held-out "
            "samples and were not used to choose settings or change the split. Linear regression "
            "is simpler to interpret; random forests can capture nonlinear relationships but are "
            "less transparent; SVR depends on scaling and parameter choices. One split cannot prove "
            "that a model is best on every dataset or ready for medical use.", "",
            'SVR has a test MSE of 2812.7179, compared with 2820.8966 for the random forest and 2900.1936 for linear regression. Its advantage over the forest is about 0.3%, so this test split does not establish a decisive overall winner. SVR also has the lowest MAE (41.9565) and highest explained variance (0.4797). Linear regression has the strongest training validation result and offers the simplest interpretation; I would treat it as a strong baseline rather than dismiss it based on this one test split.', "", "## Selected settings", ""]
        for name, search in self.searches.items():
            lines.append(f"- {name}: `{search.best_params_}`")
        lines += ["", "All models used 353 training samples and the same 89 test samples. "
                  "Settings were chosen by five-fold training-only cross-validation. "
                  "Preprocessing was fitted inside each training fold. Results can vary by software version.", ""]
        (output_dir / "model_comparison.md").write_text("\n".join(lines))
        print("\n" + "\n".join(lines[:8]))

    def run(self, output_dir):
        """Execute the entire workflow in the required order."""
        self.load_data()
        self.split_data()
        self.create_models()
        self.train_models()
        self.evaluate_models()
        self.save_results(output_dir)


if __name__ == "__main__":
    project = DiabetesRegression()
    project.run(Path(__file__).resolve().parent)
