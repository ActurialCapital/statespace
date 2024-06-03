import pandas as pd
import numpy as np
from typing import Any

from statespace.reporting.base import BaseReport


class Backtest(BaseReport):
    """
    Class for performing a backtest using signals and market benchmarks.

    Methods
    -------
    apply
        Compute the backtest results from signals and market benchmarks.
    """

    def __init__(self, key: str):
        self.key = key

    def apply(self, records: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        """
        Perform backtest calculations on portfolio records against a market 
        benchmark.

        Parameters
        ----------
        records : pd.DataFrame
            DataFrame containing portfolio records.
        benchmark : pd.DataFrame
            DataFrame containing market benchmark data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the backte
        """
        # Case pandas MultiIndex beyound 'Date' and 'symbol'
        df = records.droplevel(
            list(set(records.index.names) - set(['Date', 'symbol']))
        )
        bench = benchmark.droplevel(
            list(set(benchmark.columns.names) - set(['symbol'])),
            axis=1
        )
        # By convention, we shift forward predictors and not returns
        pred = df['Signal'].unstack().shift()
        test = bench.loc[pred.index, pred.columns]

        # Apply simple backtest
        bt = []
        for book in [self.key, 'long', 'short']:
            # Shift weights (as predictors)
            weight = df[book].unstack().shift().fillna(0).abs()
            # Lazy backtest
            returns = (pred * test * weight).sum(axis=1).rename(book)
            bt.append(returns)

        return pd.concat(bt, axis=1)


class BenchmarkReturns(BaseReport):
    """
    Class for calculating the benchmark returns using a specified weithing 
    scheme.

    Methods
    -------
    apply
        Compute the weighted benchmark returns.
    """

    def __init__(
        self,
        model: Any = None,
        group_by: None | list | str = None,
        resampled: bool = False,
        freq: str = 'Q'
    ):
        self.model = model
        self.group_by = group_by
        self.resampled = resampled
        self.freq = freq

    def apply(self, benchmark: pd.DataFrame) -> pd.DataFrame:
        """
        Perform backtest calculations on portfolio records against a market 
        benchmark.

        Parameters
        ----------
        benchmark : pd.DataFrame
            DataFrame containing market benchmark data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the weighted benchmark returns.
        """
        signals = pd.DataFrame(
            np.ones(benchmark.shape),
            index=benchmark.index,
            columns=benchmark.columns
        )
        weights = (
            signals
            if self.model is None
            else self.model.transform(signals, benchmark)
        )
        data = benchmark * weights

        # Resample
        if self.resampled:
            data = data.resample(self.freq).sum()

        # GroupBy
        if self.group_by is not None:
            data = data.groupby(level=self.group_by, axis=1).sum()

        else:  # Sum weighted returns
            name = 'EqualWeighted' if self.model is None else self.model
            data = data.sum(axis=1).rename(f'Benchmark {name}')

        return data


class Performance:
    """
    Class for calculating overall portfolio performance.

    Methods
    -------
    apply(records, benchmark)
        Compute the cumulative portfolio performance.
    """

    def apply(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the overall portfolio performance.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame containing portfolio or benchmark returns.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the cumulative portfolio or benchamrk 
            performance.
        """
        return np.cumprod(1 + returns).ffill()


class Turnover(BaseReport):
    """
    Class for computing portfolio turnover.

    Parameters
    ----------
    group_by : list or str
        Grouping criteria for portfolio weights, default is 'symbol'.
    cumsum : bool
        Flag to apply cumulative sum of turnover, default is False.
    average : bool
        Flag to apply average turnover, default is False.

    Methods
    -------
    apply(weights)
        Compute the turnover of the portfolio based on given weights.
    """

    def __init__(
        self,
        group_by: None | list | str = None,
        cumsum: bool = False,
        average: bool = False
    ):
        self.group_by = group_by
        self.cumsum = cumsum
        self.average = average

    def apply(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the turnover of the portfolio.

        Parameters
        ----------
        weights : pd.DataFrame
            DataFrame containing portfolio weights.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio turnover.
        """
        if self.average and self.cumsum:
            raise ValueError(
                "Parameters `cumsum` and `average` can not be both set to True."
            )

        # Fill NaN with zeros: NaN are considered as non-invested
        weights = weights.fillna(0)

        # GroupBy
        if self.group_by is not None:
            weights = weights.groupby(level=self.group_by, axis=1).sum()

        # Calculate Turnover: The sum of absolute changes
        turnover = weights.diff().abs().sum(axis=1)

        # Cumsum
        if self.cumsum:
            turnover = turnover.cumsum()

        # Average
        if self.average:
            turnover = turnover.expanding().mean()

        name = (
            f"{'Average' if self.average else 'Cumulative' if self.cumsum else ''} "
            "Turnover"
        )

        return turnover.rename(name).to_frame()


class Exposure(BaseReport):
    """
    Class for computing portfolio exposure.

    Parameters
    ----------
    group_by : list or str
        Grouping criteria for exposure calculation, default is 'symbol'.
    gross_exposure : bool
        Flag to calculate gross exposure, default is True.
    percent : bool
        Flag to express exposure in percent, default is False.
    resampled : bool
        Flag to resample the exposure data, default is True.
    freq : str
        Frequency string for resampling, default is 'Q'.

    Methods
    -------
    apply(records, market_returns)
        Compute the exposure of the portfolio.
    """

    def __init__(
        self,
        key: str,
        group_by: None | list | str = None,
        gross_exposure: bool = True,
        percent: bool = False,
        resampled: bool = True,
        freq: str = 'MS',
    ):
        self.key = key
        self.group_by = group_by
        self.gross_exposure = gross_exposure
        self.percent = percent
        self.resampled = resampled
        self.freq = freq

    def apply(self, records, market_returns) -> pd.DataFrame:
        """
        Calculate the exposure of the portfolio.

        Parameters
        ----------
        records : pd.DataFrame
            DataFrame containing portfolio records.
        market_returns : pd.DataFrame
            DataFrame containing market return data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio exposure.
        """
        # Get axis
        axis = dict(
            index='Date',
            columns=[name for name in records.index.names if name != 'Date']
        )

        container = {}
        for book in [self.key, 'long', 'short']:

            # Pivot weights and shift forward
            exposure = pd.pivot_table(records, values=book, **axis).shift()

            # Gross exposure
            if self.gross_exposure:
                exposure = exposure.abs()

            # Resample
            if self.resampled:
                exposure = exposure.resample(self.freq).mean()

            # Percent
            if self.percent:
                exposure = exposure.div(exposure.abs().sum(axis=1), axis=0)

            # GroupBy
            if self.group_by is not None:
                grouping = exposure.groupby(level=self.group_by, axis=1)
                exposure = grouping.sum().replace(0, np.nan)

            # Append to container
            container[book] = exposure

        if self.group_by is not None:
            if isinstance(self.group_by, list):
                names = ['Weight'] + self.group_by
            else:
                names = ['Weight', self.group_by]
        else:
            names = ['Weight'] + [col for col in axis.get('columns')]

        return pd.concat(container, axis=1, names=names)


class Contribution(BaseReport):
    """
    Class for computing portfolio contribution.

    Parameters
    ----------
    group_by : list or str
        Grouping criteria for contribution calculation, default is 'symbol'.
    resampled : bool
        Flag to resample the contribution data, default is True.
    freq : str
        Frequency string for resampling, default is 'Q'.

    Methods
    -------
    apply(records, market_returns)
        Compute the contribution of the portfolio.
    """

    def __init__(
        self,
        key: str,
        group_by: None | list | str = None,
        resampled: bool = False,
        freq: str = 'MS'
    ):
        self.key = key
        self.group_by = group_by
        self.resampled = resampled
        self.freq = freq

    def apply(self, records, market_returns) -> pd.DataFrame:
        """
        Calculate the contribution of the portfolio.

        Parameters
        ----------
        records : pd.DataFrame
            DataFrame containing portfolio records.
        market_returns : pd.DataFrame
            DataFrame containing market return data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio contribution.
        """
        # Get axis
        axis = dict(
            index='Date',
            columns=[name for name in records.index.names if name != 'Date']
        )
        # Pivot signals and shift forward
        signals = pd.pivot_table(records, values='Signal', **axis).shift()

        # Bechmark aligned
        rets = market_returns.loc[signals.index, signals.columns]

        container = {}
        for book in [self.key, 'long', 'short']:

            # Pivot absolute weights and shift forward
            weight = pd.pivot_table(records, values=book, **axis).shift().abs()

            # Calculate contribution
            contribution = signals * rets * weight

            # Resample
            if self.resampled:
                contribution = contribution.resample(self.freq).sum()

            # GroupBy
            if self.group_by is not None:
                grouping = contribution.groupby(level=self.group_by, axis=1)
                contribution = grouping.sum().replace(0, np.nan)

            # Append to container
            container[book] = contribution

        if self.group_by is not None:
            if isinstance(self.group_by, list):
                names = ['Weight'] + self.group_by
            else:
                names = ['Weight', self.group_by]
        else:
            names = ['Weight'] + [col for col in axis.get('columns')]

        return pd.concat(container, axis=1, names=names)


class Predictors(BaseReport):
    """
    Class for processing portfolio predictors.

    Parameters
    ----------
    group_by : list or str
        Grouping criteria for predictors, default is 'symbol'.
    transform_to_ranks : bool
        Flag to transform predictions to ranks, default is True.
    resampled : bool
        Flag to resample the predictors data, default is True.
    freq : str
        Frequency string for resampling, default is 'Q'.

    Methods
    -------
    apply(pred)
        Process and apply predictors for the portfolio.
    """

    def __init__(
        self,
        group_by: None | list | str = None,
        transform_to_ranks: bool = True,
        resampled: bool = True,
        freq: str = 'MS'
    ):
        self.group_by = group_by
        self.transform_to_ranks = transform_to_ranks
        self.resampled = resampled
        self.freq = freq

    def apply(self, pred) -> pd.DataFrame:
        """
        Process and apply predictors for the portfolio.

        Parameters
        ----------
        pred : pd.DataFrame
            DataFrame containing predictors.

        Returns
        -------
        pd.DataFrame
            DataFrame containing processed predictors.
        """
        # Resampled
        if self.resampled:
            pred = pred.resample(self.freq).sum()

        # GroupBy
        if self.group_by is not None:
            pred = pred.groupby(level=self.group_by, axis=1).sum()

        # Focus on symbol or Identifier
        if isinstance(pred.columns, pd.MultiIndex) and self.group_by in ['symbol']:
            pred = pred.loc[pred.columns.get_level_values(self.group_by)]

        # Top 10
        if self.group_by in ['symbol', 'Identifier']:
            pred = pred.apply(lambda series, n: pd.Series(
                series.nlargest(n)), axis=1, n=10)

        # Ranks
        if self.transform_to_ranks:
            pred = pred.rank(axis=1)

        return pred


class SizeByGroup(BaseReport):
    """
    Class for computing portfolio size by group.

    Parameters
    ----------
    group_by : list or str
        Grouping criteria for size calculation, default is 'symbol'.

    Methods
    -------
    apply(signals)
        Compute the size of the portfolio by group.
    """

    def __init__(
        self,
        key: str,
        group_by: None | list | str = None,
        resampled: bool = False,
        freq: str = 'MS'
    ):
        self.key = key
        self.group_by = group_by
        self.resampled = resampled
        self.freq = freq

    def apply(self, signals) -> pd.DataFrame:
        """
        Calculate the size of the portfolio by group.

        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame containing portfolio signals.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio size by group.
        """
        df = signals.stack([name for name in signals.columns.names])
        level = self.group_by if isinstance(
            self.group_by, list) else [self.group_by]
        level = ['Date'] + level
        # Size
        size = pd.concat([
            df[df != 0].dropna().groupby(level=level, axis=0).count(),
            df[df > 0].dropna().groupby(level=level, axis=0).count(),
            df[df < 0].dropna().groupby(level=level, axis=0).count(),
        ],
            axis=1,
            keys=[self.key, 'long', 'short']
        )
        if self.resampled:
            size = size.resample(self.freq, level='Date').sum()

        return size.fillna(0).astype(int)


class Binarizer(BaseReport):
    """
    Class for creating binary labels from signals and benchmarks.

    Parameters
    ----------
    rank : bool
        If True, generate labels based on quantile ranks, default is False.
    q : int
        Number of quantiles to use for ranking if rank is True, default 
        is 4.

    Methods
    -------
    apply(signals, benchmark)
        Compute binary labels based on signals and benchmark data.
    """

    def __init__(self, rank: bool = False, q: int = 4):
        self.rank = rank
        self.q = q

    def apply(
        self,
        signals: pd.DataFrame,
        benchmark: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute binary labels based on the provided signals and benchmark data.

        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame containing the signals to be used for generating labels.
        benchmark : pd.DataFrame
            DataFrame containing benchmark data for comparison.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing binary labels generated from signals and 
            benchmark.
        """
        # Signals shifted forward
        signals = signals.shift()

        # Ranks
        if self.rank:
            # Create quantile ranks
            target = pd.concat([
                pd.qcut(
                    series, q=self.q, duplicates='drop', labels=False
                )
                for _, series in benchmark.iterrows()
                if not series.isnull().all()
            ],
                axis=1
            ).T

            keys = sorted(
                set(target.stack([name for name in target.columns.names])))
            mapping = {
                key: -1 if key == min(keys) else 1 if key == max(keys) else 0
                for key in keys
            }
            target = target.replace(mapping)

        else:
            # Create binary target from positive and negative returns
            target = pd.DataFrame(
                np.select(
                    [benchmark < 0, benchmark > 0], [-1, 1], default=np.nan
                ),
                index=benchmark.index,
                columns=benchmark.columns
            )

        df = pd.concat([
            signals.stack([name for name in signals.columns.names]),
            target.stack([name for name in target.columns.names])
        ],
            axis=1,
            keys=['Signal', 'Target']
        )

        return df.dropna()


class ClassificationMetrics(BaseReport):
    """
    Class for computing classification metrics based on binary labels model.

    Parameters
    ----------
    rank : bool
        Determines if labels should be created based on quantile ranks.
    q : int
        Number of quantiles to use for ranking if `rank` is True.

    Methods
    -------
    apply(signals, benchmark)
        Computes a variety of classification metrics based on signals and 
        benchmark data.
    """

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and return various classification metrics.

        This method first applys binary labels using signals and benchmark 
        data, and then calculates several classification metrics such as 
        accuracy, AUC, Cohen's kappa, Matthews correlation coefficient, Hamming 
        loss, and Jaccard score.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the binary labels.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing various classification metrics, each metric 
            as a row.

        Raises
        ------
        ValueError
            If an error occurs in the computation of binary labels or 
            classification metrics.
        """
        # Import sklearn metrics locally
        from sklearn.metrics import classification_report as cr
        # Data
        y_true, y_pred = data['Target'], data['Signal']
        # Initial classification report
        return pd.DataFrame(cr(y_true, y_pred, output_dict=True)).T
