"""Classical Machine Learning models definitions for AMR prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def get_classical_models(random_state=42):
    """Return dictionary of classical ML classifiers configured for tabular AMR prediction."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            C=1.0,
            solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=8,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=random_state
        )
    }
    return models
