import numpy as np

from optuna.trial import Trial

try:
    from sklearn.base import BaseEstimator
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.model_selection import train_test_split
except:
    raise ModuleNotFoundError(
        "You need to install scikit-learn to run this example: "
        "pip install scikit-learn"
    )

from statespace.base import BaseStudy, Listed
from statespace.decorators import run_study


class MyCustomObjective(BaseStudy):
    def __init__(self, config, strategy, *data, **create_study_kwargs):
        super().__init__(config, strategy, *data, **create_study_kwargs)

    @run_study
    def objective(self, trial: Trial) -> float:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            *self.data,
            test_size=0.33,
            shuffle=False,
            random_state=42
        )
        # Model
        pred = self.model.fit(y_train, X_train).predict(y_test)
        # Evaluate performance
        return mean_squared_error(y_test, pred)


def strategy(estimator: BaseEstimator, preprocessor: BaseEstimator) -> Pipeline:
    return make_pipeline(preprocessor, estimator)


if __name__ == "__main__":

    # Seed

    np.random.seed(123)

    # Params

    length, paths = 100, 10

    # Data

    X = np.random.normal(size=(length, paths))
    y = np.random.normal(size=(length, paths))

    # Configuration file

    config = {
        'preprocessor': Listed([MinMaxScaler(), RobustScaler(), StandardScaler()]),
        'estimator':    Listed([LinearRegression(), Ridge(), Lasso(), ElasticNet()]),
    }

    # Create study

    custom_model = MyCustomObjective(
        config, strategy, X, y, study_name='statespace', direction="minimize")

    # Run study

    model = custom_model.execute(n_trials=5)
    # A new study created in memory with name: statespace


    # Get best parameters

    print(model.best_trial.params)
    # {'preprocessor': StandardScaler(), 'estimator': LinearRegression()}


    # Get best value

    print(model.best_value)
    # 1.3722278355576882


    # Visulatize the output

    from optuna import visualization
    fig = visualization.plot_contour(
        model, params=["estimator", "preprocessor"])
    fig.show(renderer='svg')
    fig.write_image("../docs/static/example_contour_plot.png")
