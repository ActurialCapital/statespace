from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

from statespace.reporting.accessors import WrapAccessor


@dataclass
class BaseScore:
    """Abstract base class template to allocate score to investment strategy"""
    short: int
    neutral: int
    long: int


@dataclass
class LongShort(BaseScore):
    """Long-Short based strategy"""
    short: int = -1
    neutral: int = 0
    long: int = 1


@dataclass
class LongOnly(BaseScore):
    """Long obly based strategy"""
    short: int = 0
    neutral: int = 0
    long: int = 1


@dataclass
class ShortOnly(BaseScore):
    """Short only based strategy"""
    short: int = -1
    neutral: int = 0
    long: int = 0


class BaseReport(ABC):
    """
    Abstract base class for strategy report methods.

    This class provides an interface for summarizing the results of investment
    strategies. Implementations of this class should define the `apply` method
    to generate a summary of the strategy's performance or characteristics.

    Methods
    -------
    apply(**kwargs) -> pd.DataFrame
        Abstract method for applying the summary method.
    """

    def __repr__(self):
        """
        Special method to return the string representation of the instance,
        dynamically using the class name of the subclass.
        """
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def apply(self, **kwargs) -> pd.DataFrame | WrapAccessor:
        """
        Abstract method for applying the report method.

        Subclasses should implement this method to generate a summary of the
        strategy's performance or characteristics based on the provided data
        and parameters.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments containing data and parameters relevant
            to the strategy.

        Returns
        -------
        pd.DataFrame | WrapAccessor
            A DataFrame or accessor summarizing the strategy's records or 
            characteristics.
        """
        pass
