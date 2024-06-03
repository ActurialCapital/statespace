from functools import cache, cached_property
from typing import Any
import pandas as pd
import numpy as np

import statespace.reporting as rp


class WrapAccessor:
    """
    Wrapper class for delegating method calls to an instance of `Reporter`, 
    `Analyzer` and `Plotter` classes, allowing for simplified method calls.

    Parameters
    ----------
    summary : pd.DataFrame
        Summary DataFrame to be used in plotting.
    key : str | int, optional
        Weight column name to be analysed. Defaults to -1.
    """

    def __init__(self, summary: pd.DataFrame, key: str | int = -1):
        self.analyzer = Analyzer(summary, key)

    def __getattr__(self, name):
        """
        Delegate method calls or attribute access to the plotter instance.

        Parameters
        ----------
        name : str
            The name of the method or attribute to access.

        Returns
        -------
        Any
            The attribute or method from the accessor instance.

        Raises
        ------
        AttributeError
            If the attribute or method is not found in the accessor instance.
        """
        # Delegate method calls to the Analyzer instance
        if hasattr(self.analyzer, name):
            return getattr(self.analyzer, name)
        else:
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name}"
            )


@pd.api.extensions.register_dataframe_accessor("statespace")
class DataFrameAccessor:
    """
    Custom DataFrame accessor for creating an instance of WrapAccessor.

    This accessor extends a DataFrame allowing it to initialize an instance of
    WrapAccessor with additional parameters for market returns, weight, and
    column names.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame to which the accessor is attached.

    Methods
    -------
    __call__
        Initializes and returns an instance of WrapAccessor.

    Parameters
    ----------
    summary : pd.DataFrame
        The DataFrame to which this accessor is attached.
    """

    def __init__(self, summary: pd.DataFrame):
        self.df = summary

    def __call__(self, key: str | int = -1):
        """
        Initialize and return an instance of WrapAccessor.

        Parameters
        ----------
        key : str | int, optional
            Weight column name to be used in WrapAccessor for analytics. 
            Defaults to -1.

        Returns
        -------
        WrapAccessor
            An instance of WrapAccessor initialized with the provided 
            parameters.
        """
        return WrapAccessor(self.df, key)


class Reporter:
    """
    Class for conducting various report on a financial portfolio.

    Parameters
    ----------
    summary : pd.DataFrame
        DataFrame containing portfolio summary data.
    key : str | int, optional
        Weight column name in the summary. Defaults to -1.

    Attributes
    ----------
    weights
        Unstacks and returns the portfolio weights.
    signals
        Unstacks and returns the portfolio signals.
    predictions
        Unstacks and returns the portfolio predictions.
    direction
        Unstacks and returns the portfolio direction.
    levels
        Get the levels (column names) of the portfolio weights DataFrame.
    records
        Compute and return portfolio records including metadata, signals, and
        weights.
    frequency
        Get the frequency of the index in the portfolio weights DataFrame.
    start_index
        Get the start date of the data in the portfolio weights DataFrame.
    end_index
        Get the end date of the data in the portfolio weights DataFrame.
    n_periods 
        Get the number of periods in the data.
    symbols
        Get a list of unique symbols (assets) from the portfolio weights
        DataFrame.
    n_orders
        Get the total number of orders in different categories.
    benchmark
        Get benchmark, the aligned market returns to portfolio weights.

    Methods
    -------
    create_report
        Generate a report using a specified analytics computation.
    create_plot
        Generate a plot using a specified plotter computation.
    """

    def __init__(self, summary: pd.DataFrame, key: str | int = -1):
        self.summary = summary
        self.key = key if isinstance(key, str) else self.summary.columns[-1]

    @property
    def levels(self) -> list:
        """
        Get the levels (column names) of the portfolio weights DataFrame.

        Returns
        -------
        list
            List of column names (levels).

        """
        index_levels = [
            x for x in self.summary.index.names if x != 'Date'
        ]
        if not index_levels:
            raise ValueError(
                'Unable to identify name(s) from `weights`.'
            )

        return index_levels

    def __str__(self):
        """
        Print the weight attribute.

        Returns
        -------
        str
            weight attribute

        """
        return str(self.key)

    @property
    def market_returns(self) -> pd.DataFrame:
        """
        Unstacks and returns the market returns.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the unstacked market returns.

        """
        return self.summary['Market'].unstack(self.levels)

    @property
    def weights(self) -> pd.DataFrame:
        """
        Unstacks and returns the portfolio weights.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the unstacked portfolio weights.

        """
        return self.summary[self.key].unstack(self.levels)

    @property
    def signals(self) -> pd.DataFrame:
        """
        Unstacks and returns the portfolio signals.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the unstacked portfolio signals.

        """
        return self.summary['Signal'].unstack(self.levels)

    @property
    def predictions(self) -> pd.DataFrame:
        """
        Unstacks and returns the portfolio predictions.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the unstacked portfolio predictions.

        """
        return self.summary['Prediction'].unstack(self.levels)

    @property
    def direction(self) -> pd.DataFrame:
        """
        Unstacks and returns the portfolio direction.

        Returns
        -------
        pd.DataFrame
            A DataFrame representing the unstacked portfolio direction.

        """
        return self.summary['Direction'].unstack(self.levels)

    @cached_property
    def records(self) -> pd.DataFrame:
        """
        Compute and return portfolio records including metadata, signals, and
        weights.

        Returns
        -------
        pd.DataFrame
            Portfolio records time series.

        """
        port = self.summary.loc[:, ['Signal', self.key]]
        port['long'] = port[self.key].where(port['Signal'] > 0)
        port['short'] = port[self.key].where(port['Signal'] < 0)

        return port

    @property
    def frequency(self) -> str:
        """
        Get the frequency of the index in the portfolio weights DataFrame.

        Returns
        -------
        str
            String frequency of the index.

        """
        return pd.infer_freq(self.weights.index)

    @property
    def start_index(self) -> pd.Timestamp:
        """
        Get the start date of the data in the portfolio weights DataFrame.

        Returns
        -------
        pd.Timestamp
            Start date of the data.

        """
        return self.weights.index[0]

    @property
    def end_index(self) -> pd.Timestamp:
        """
        Get the end date of the data in the portfolio weights DataFrame.

        Returns
        -------
        pd.Timestamp
            End date of the data.

        """
        return self.weights.index[-1]

    @property
    def n_periods(self) -> int:
        """
        Get the number of periods in the data.

        Returns
        -------
        int
            Number of periods.

        """
        # minus first record
        return int(len(self.weights.index) - 1)

    @property
    def n_trades(self):
        pass

    @property
    def symbols(self) -> list:
        """
        Get a list of unique symbols (assets) from the portfolio weights
        DataFrame.

        Returns
        -------
        list
            List of unique symbols (assets).

        """
        return list(
            self.weights.columns.get_level_values('symbol').drop_duplicates()
        )

    @property
    def n_orders(self) -> dict:
        """
        Get the total number of orders in different categories.

        Returns:
            dict: Dictionary with the count of orders in
            overall, long, and short categories.
        """
        return {b: len(self.records[b].dropna()) for b in [self.key, 'long', 'short']}

    @property
    def benchmark(self) -> pd.DataFrame:
        """
        Get benchmark, the aligned market returns to portfolio weights.

        Returns
        -------
        pd.DataFrame
            Benchmark returns time series.

        """
        testset = self.market_returns.copy()
        if isinstance(testset.columns, pd.MultiIndex):
            testset.columns = testset.columns.get_level_values('symbol')
        # Align returns
        testset, _ = testset.align(self.weights, join='right', level='symbol')
        return testset

    def create_report(
        self,
        analytics: Any,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate a report using a specified analytics computation.

        This method allows for flexible analytics computations by passing an 
        analytics object and additional arguments required for its computation. 
        The method calls the `compute` method of the provided analytics object.

        Parameters
        ----------
        analytics : Any
            An instance of a BaseAnalytics subclass that will 
            perform the computation.
        *args
            Positional arguments to be passed to the `compute` method of the 
            analytics object.
        **kwargs
            Keyword arguments to be passed to the `compute` method of the 
            analytics object.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results of the analytics computation.

        """
        return analytics.apply(*args, **kwargs)

    def create_plot(
        self,
        plotter: Any,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate a plot using a specified plotter computation.

        This method allows for flexible plotting by passing an 
        plotter object and additional arguments required for its computation. 
        The method calls the `plot` method of the provided analytics object.

        Parameters
        ----------
        plotter : Any
            An instance of a plotter subclass that will perform 
            the visulalization.
        *args
            Positional arguments to be passed to the `plot` method of the 
            plotter object.
        **kwargs
            Keyword arguments to be passed to the `plot` method of the 
            plotter object.

        Returns
        -------
        BaseFigureDisplayer
            A DataFrame containing the results of the analytics computation.

        """
        return NotImplementedError

    def select_data(
        self,
        select: str,
        market: pd.DataFrame,
        portfolio: pd.DataFrame,
        records: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Mapping of selected values to corresponding attributes or methods"""
        select_mapping = {
            'market': market,
            'portfolio': portfolio,
            'records': records
        }
        data = select_mapping.get(select)
        if data is None:
            raise ValueError('Please pass a valid selector.')
        return data


class Analyzer(Reporter):
    """
    Analyzer class for performing various analysis tasks on portfolio data.

    Parameters
    ----------
    summary : pd.DataFrame
        Summary DataFrame for analysis.
    key : str | int, optional
        Weight column name to be analysed. Defaults to -1.
    """

    def __init__(
        self,
        summary: pd.DataFrame,
        key: str | int = -1,
    ):
        super().__init__(summary, key)

    def llm(self, openai_api_token: str) -> Any:
        """
        Analyze strategy results using LLM (Large Language Model).

        This method provides a default implementation for summarizing 
        investment strategy results using LLM. It leverages `pandasai`:

        > PandasAI is a Python library that makes it easy to ask questions to 
        your data (CSV, XLSX, PostgreSQL, MySQL, BigQuery, Databrick, 
        Snowflake, etc.) in natural language. xIt helps you to explore, clean, 
        and analyze your data using generative AI.

        More information in the [documentation](https://docs.pandas-ai.com/en/latest/).

        !!! note

            Currently, this wrapper leverages OpenAI GPT3.5 or GPT4 depending 
            on the provided `openai_api_token`.

        Parameters
        ----------
        openai_api_token : dict
            `pandasai` OpenAI API token.

        Returns
        -------
        SmartDataframe
            The SmartDataframe class is the main class of pandasai. It is used 
            to interact with a single dataframe.

        """
        try:
            from pandasai import SmartDataframe
            from pandasai.llm import OpenAI
        except:
            raise ModuleNotFoundError(
                "You need to install pandasai to run this example: "
                "pip install pandasai"
            )
        model = OpenAI(api_token=openai_api_token)
        return SmartDataframe(self.summary, config={"llm": model})

    @cache
    def backtest(self) -> pd.DataFrame:
        """
        Perform a backtest on the portfolio.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the dummy backte
        """
        model = rp.Backtest(self.key)
        return self.create_report(model, self.records, self.benchmark)

    def benchmark_returns(
        self,
        model: Any = None,
        group_by: None | list | str = None,
        resampled: bool = False,
        freq: str = 'Q',
    ) -> pd.DataFrame:
        """
        Calculate the weighted benchmark returns

        Parameters
        ----------
        benchmark : pd.DataFrame
            DataFrame containing benchmark return data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing benchmark performance.
        """
        model = rp.BenchmarkReturns(model, group_by, resampled, freq)
        return self.create_report(model, self.benchmark)

    def performance(
        self,
        select: str = "portfolio",
        group_by: None | list | str = None,
        resampled: bool = False,
        freq: str = 'Q',
        model: Any = None
    ) -> pd.DataFrame:
        """
        Compute the performance.

        Parameters
        ----------
        select : str, optional
            Data selection. Could be 'market', 'portfolio' or 'records'.
        group_by : None | list | str, optional
            Criteria to group the performance calculation. Only effective if 
            `select` is set to `portfolio`. default is 'symbol'.
        resampled : bool, optional
            Flag to resample performance data. Only effective if `group_by` is
            set to True. Default is True.
        freq : str, optional
            Frequency string for resampling. Only effective if `group_by` is
            set to True. Default is 'Q'.
        model : Any, optional
            Market weights. If None, EqualWeighted is applied. Defaults to 
            None.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio performance.

        Raises
        ------
        ValueError
            If an invalid selector is passed.
        """
        data = self.select_data(
            select=select,
            market=self.benchmark_returns(model, group_by, resampled, freq),
            portfolio=self.contribution(
                group_by, resampled, freq) if group_by else self.backtest()
        )
        # Generate the report with the specified data.
        return self.create_report(rp.Performance(), data)

    def turnover(
        self,
        group_by: None | list | str = None,
        cumsum: bool = False,
        average: bool = False
    ) -> pd.DataFrame:
        """
        Compute the turnover of the portfolio.

        Parameters
        ----------
        group_by : None | list | str, optional
            Criteria to group the turnover calculation, default is 'symbol'.
        cumsum : bool, optional
            Flag to compute cumulative sum, default is False.
        average : bool, optional
            Flag to compute average, default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio turnover.
        """
        model = rp.Turnover(group_by, cumsum, average)
        return self.create_report(model, self.weights)

    def exposure(
        self,
        group_by: None | list | str = None,
        gross_exposure: bool = True,
        percent: bool = False,
        resampled: bool = False,
        freq: str = 'MS'
    ) -> pd.DataFrame:
        """
        Compute the exposure of the portfolio.

        Parameters
        ----------
        group_by : None | list | str, optional
            Criteria to group the exposure calculation, default is 'symbol'.
        gross_exposure : bool, optional
            Flag to compute gross exposure, default is True.
        percent : bool, optional
            Flag to compute exposure in percent, default is False.
        resampled : bool, optional
            Flag to resample exposure data, default is True.
        freq : str, optional
            Frequency string for resampling, default is 'Q'.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio exposure.
        """
        model = rp.Exposure(
            self.key,
            group_by,
            gross_exposure,
            percent,
            resampled,
            freq
        )
        return self.create_report(model, self.records, self.market_returns)

    def contribution(
        self,
        group_by: None | list | str = None,
        resampled: bool = False,
        freq: str = 'Q'
    ) -> pd.DataFrame:
        """
        Compute the contribution of the portfolio.

        Parameters
        ----------
        group_by : None | list | str, optional
            Criteria to group the contribution calculation, default is 'symbol'.
        resampled : bool, optional
            Flag to resample contribution data, default is True.
        freq : str, optional
            Frequency string for resampling, default is 'Q'.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio contribution.
        """
        model = rp.Contribution(
            self.key,
            group_by,
            resampled,
            freq
        )
        return self.create_report(model, self.records, self.market_returns)

    def predictors(
        self,
        group_by: None | list | str = None,
        transform_to_ranks: bool = True,
        resampled: bool = False,
        freq: str = 'Q'
    ) -> pd.DataFrame:
        """
        Compute predictors for the portfolio.

        Parameters
        ----------
        group_by : None | list | str, optional
            Criteria to group the predictor calculation, default is 'symbol'.
        transform_to_ranks : bool, optional
            Flag to transform predictions to ranks, default is True.
        resampled : bool, optional
            Flag to resample predictor data, default is True.
        freq : str, optional
            Frequency string for resampling, default is 'Q'.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio predictors.
        """
        model = rp.Predictors(
            group_by,
            transform_to_ranks,
            resampled,
            freq
        )
        return self.create_report(model, self.predictions)

    def size_by_group(
        self,
        group_by:  list | str,
        resampled: bool = False,
        freq: str = 'Q'
    ) -> pd.DataFrame:
        """
        Compute the size of the portfolio by group.

        Parameters
        ----------
        group_by : list | str
            Criteria to group the size calculation.
        resampled : bool, optional
            Flag to resample predictor data, default is True.
        freq : str, optional
            Frequency string for resampling, default is 'Q'.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the portfolio size by group.
        """
        model = rp.SizeByGroup(self.key, group_by, resampled, freq)
        return self.create_report(model, self.signals)

    def binarize(self, rank: bool = False, q: int = 4) -> pd.DataFrame:
        """
        Create binary labels for the portfolio.

        Parameters
        ----------
        rank : bool, optional
            Flag to create labels based on quantile ranks, default is False.
        q : int, optional
            Number of quantiles to use for ranking, default is 4.

        Returns
        -------
        pd.DataFrame
            DataFrame containing binary labels for the portfolio.
        """
        model = rp.Binarizer(rank, q)
        return self.create_report(model, self.signals, self.benchmark)

    def classification_metrics(self, rank: bool = False, q: int = 4) -> pd.DataFrame:
        """
        Compute classification metrics for the portfolio.

        Parameters
        ----------
        rank : bool, optional
            Flag to compute metrics based on quantile ranks, default is False.
        q : int, optional
            Number of quantiles to use for ranking, default is 4.

        Returns
        -------
        pd.DataFrame
            DataFrame containing classification metrics for the portfolio.
        """
        model = rp.ClassificationMetrics()
        return self.create_report(model, self.binarize(rank, q))

    def mean_returns(self, select: str = "portfolio"):
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        return data.mean()

    def std_returns(self, select: str = "portfolio"):
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        return data.std()

    def min_returns(self, select: str = "portfolio"):
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        return data.min()

    def median_returns(self, select: str = "portfolio"):
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        return data.median()

    def max_returns(self, select: str = "portfolio"):
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        return data.max()

    def min_returns_index(self, select: str = "portfolio"):
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        return data.idxmin()

    def max_returns_index(self, select: str = "portfolio"):
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        return data.idxmax()

    def total_fees_paid(self):
        return NotImplementedError

    def win_rate(self):
        return NotImplementedError

    def best_trade(self):
        return NotImplementedError

    def average_winning_trade(self):
        return NotImplementedError

    def average_losing_trade(self):
        return NotImplementedError

    def average_winning_trade_duration(self):
        return NotImplementedError

    def average_losing_trade_duration(self):
        return NotImplementedError

    def profit_factor(self):
        return NotImplementedError

    def expectancy(self):
        return NotImplementedError

    def coverage(self):
        return NotImplementedError

    def total_records(self):
        return NotImplementedError

    def total_recovered_drawdowns(self):
        return NotImplementedError

    def total_active_drawdowns(self):
        return NotImplementedError

    def active_drawdowns(self):
        return NotImplementedError

    def active_duration(self):
        return NotImplementedError

    def active_recovery(self):
        return NotImplementedError

    def active_recovery_returns(self):
        return NotImplementedError

    def active_recovery_duration(self):
        return NotImplementedError

    def max_drawdown(self):
        return NotImplementedError

    def average_drawdown(self):
        return NotImplementedError

    def max_drawdown_duration(self):
        return NotImplementedError

    def average_drawdown_duration(self):
        return NotImplementedError

    def max_recovery_duration(self):
        return NotImplementedError

    def average_recovery_duration(self):
        return NotImplementedError

    def average_recovery_duration_ratio(self):
        return NotImplementedError

    def annualized_returns(
        self,
        select: str = "portfolio",
        annualize: int = 252
    ):
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        return np.prod(1 + data) ** (annualize / len(data)) - 1

    def annualized_volatility(
        self,
        select: str = "portfolio",
        annualize: int = 252
    ):
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        return data.std() * (annualize ** 0.5)

    def sharpe_ratio(
        self,
        select: str = "portfolio",
        annualize: int = 252,
        risk_free_rate: float = 0
    ) -> float:
        """
        Compute the Sharpe ratio of the portfolio.

        Returns
        -------
        float
            The Sharpe ratio of the portfolio.
        """
        returns = self.annualized_returns(select, annualize)
        volatility = self.annualized_volatility(select, annualize)
        return (returns - risk_free_rate) / volatility

    def calmar_ratio(
        self,
        select: str = "portfolio",
        annualize: int = 252,
    ) -> float:
        """
        Compute the Calmar Ratio of the portfolio.

        Returns
        -------
        float
            Calmar Ratio.
        """
        returns = self.annualized_returns(select, annualize)
        mdd = self.max_drawdown(select)
        return returns / mdd

    def omega_ratio(self, select: str = "portfolio", thresh: float = 0.05) -> float:
        """
        Compute the Omega Ratio of the portfolio.

        Returns
        -------
        float
            Omega Ratio.
        """
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        pos = sum(ret - thresh for ret in data if ret > thresh)
        neg = sum(ret - thresh for ret in data if ret < thresh)
        return pos / - neg if neg != 0 else float('inf')

    def sortino_ratio(self, select: str = "portfolio", thresh: float = 0.05):
        """
        Compute the Sortino Ratio of the portfolio.

        Returns
        -------
        float
            Sortino Ratio.
        """
        data = self.select_data(
            select,
            self.benchmark,
            self.backtest()[self.key]
        )
        mean_returns = self.mean_returns(select)
        squared_diff = [
            min(0, ret - thresh) ** 2 for ret in data
        ]
        downside_deviation = np.sqrt(np.mean(squared_diff))
        return mean_returns / downside_deviation
