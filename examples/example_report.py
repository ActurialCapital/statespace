import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn import set_config
    set_config(transform_output='pandas')
except:
    raise ModuleNotFoundError(
        "You need to install scikit-learn to run this example: "
        "pip install scikit-learn"
    )

from statespace.reporting import Tabular


warnings.filterwarnings('ignore')


@dataclass
class ParamTransformer:
    inverse_func: object = None
    validate: bool = False
    accept_sparse: bool = False
    check_inverse: bool = True
    feature_names_out = None
    kw_args: dict = None
    inv_kw_args: dict = None


class BaseTransformer(ABC, FunctionTransformer):
    """Abstract base class for data transformation."""

    def __init__(self, params: dict | ParamTransformer = None):
        params = params or asdict(ParamTransformer())
        super().__init__(func=self, **params)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "BaseTransformer":
        """Fit the underlying estimator on training data `X` and `y`"""
        return self

    @abstractmethod
    def __call__(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        pass


class QuantileRanks(BaseTransformer):
    """Transformer class to compute quantile-based signals."""

    def __init__(self, q: int = 4, group_by: str | list = None):
        self.q = q
        self.group_by = group_by
        super().__init__()

    def __call__(cls, X: pd.DataFrame, y=None) -> pd.DataFrame:
        transposed_data = X.T
        clean_data = transposed_data.dropna(how='all', axis=1)
        if isinstance(cls.group_by, (list, str)):
            clean_data = clean_data.groupby(level=cls.group_by)
        ranks = clean_data.transform(
            lambda df: pd.qcut(df, cls.q, labels=False, duplicates='drop')
        )
        return ranks.T


class Signal(BaseTransformer):
    """Transformer class to convert ranks into investment signals."""

    def __init__(self, higher_is_better: bool = True):
        self.higher_is_better = higher_is_better
        super().__init__()

    def __call__(cls, X: pd.DataFrame, y=None) -> pd.DataFrame:
        columns = X.columns
        if isinstance(X.columns, pd.MultiIndex):
            X.columns = X.columns.get_level_values('symbol')
        # Get Unique Keys
        keys = sorted(set(X.stack()))
        lower, upper = (-1, 1) if cls.higher_is_better else (1, -1)
        scores = {
            key: lower if key == min(keys)
            else upper if key == max(keys)
            else np.nan
            for key in keys
        }
        results = X.apply(lambda x: x.map(scores))
        results.columns = columns
        return results


class EqualWeighted(BaseTransformer):
    """Create an equal-weighted portfolio based on given signals."""

    def __init__(self):
        super().__init__()

    def __call__(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        pos = (
            X[X == 1]
            .multiply(1 / X[X == 1].count(axis=1), axis=0)
        )
        neg = (
            X[X == -1]
            .multiply(1 / X[X == -1].count(axis=1), axis=0)
        )
        return pos.add(neg, fill_value=0)


if __name__ == "__main__":

    # Seed

    np.random.seed(123)

    # Params

    length, paths = 5000, 5

    # Metadata

    categories = [
        ['ABC', 'ABC', 'ABC', 'JKL', 'JKL'],
        ['CBA', 'CBA', 'CBA', 'LKJ', 'ONM'],
        ['QWE', 'QWE', 'ERT', 'ERT', 'DFG'],
        ['asset1', 'asset2', 'asset3', 'asset4', 'asset5']
    ]
    columns = pd.MultiIndex.from_tuples(
        list(zip(*categories)),
        names=["Sector", "Industry", "Country", "symbol"]
    )
    index = pd.date_range(
        start=datetime.now().date() - timedelta(days=length - 1),
        end=datetime.now().date(),
        freq="D",
        name='Date'
    )


    # Data


    X = pd.DataFrame(
        np.random.normal(size=(length, paths)),
        columns=columns,
        index=index
    )

    y = pd.DataFrame(
        np.random.normal(size=(length, paths)),
        columns=columns,
        index=index
    )


    # Pipeline and Portfolio Construction


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, shuffle=False
    )
    pipe = make_pipeline(
        QuantileRanks(q=4, group_by=None),
        Signal(higher_is_better=True)
    )
    signals = pipe.fit(y_train, X_train).transform(y_test)
    weights = EqualWeighted().transform(signals)


    # Summary Report


    df = Tabular(accessor=False).apply(
        signals,
        market_returns=y_test,
        portfolio=weights
    )
    print(df)


    # Summary Accessor


    rp = Tabular(accessor=True).apply(
        signals,
        market_returns=y_test,
        portfolio=weights
    )
    print(rp)

    # Accessor Report

    rp.key
    rp.market_returns
    rp.weights
    rp.signals
    rp.predictions
    rp.direction
    rp.records
    rp.frequency
    rp.start_index
    rp.end_index
    rp.n_periods
    rp.symbols
    rp.n_orders
    rp.benchmark

    # Accessor Analyzer

    rp.backtest()
    rp.performance(select="portfolio")
    rp.performance(select="portfolio", group_by='Sector')
    rp.performance(select="market")
    rp.performance(select="market", group_by='Sector')
    rp.turnover()
    rp.exposure()
    rp.contribution()
    rp.size_by_group(group_by='symbol')
    rp.size_by_group(group_by='symbol', resampled=True, freq='MS')
    rp.binarize(rank=True, q=4)
    rp.classification_metrics(rank=False)
    rp.classification_metrics(rank=True, q=4)

    # ...
