from dataclasses import asdict
import pandas as pd

from statespace.reporting.base import BaseReport, BaseScore, LongShort
from statespace.reporting.accessors import WrapAccessor


class Tabular(BaseReport):
    """
    Helpers for summarizing strategy results.

    This class provides a default implementation for summarizing investment
    strategy results using tabular data, including signals, predictions, and 
    directions.

    Attributes
    ----------
    accessor : bool, optional
        If True, the apply returns an WrapAccessor object, instead 
        of a pandas DataFrame.
    scores : dict
        A dictionary mapping strategy scores to their numeric representations.
        It defaults to:

        * Short: -1
        * Neutral: 0
        * Long: 1

    Methods
    -------
    apply
        Summarizes strategy results based on signals and predictions.
    """

    def __init__(self, accessor: bool = False, scores: BaseScore = None):
        self.accessor = accessor
        self.scores = scores or LongShort()

    def apply(
        self,
        signals: pd.DataFrame,
        predictions: pd.DataFrame | None = None,
        market_returns: pd.DataFrame | None = None,
        **weights
    ) -> pd.DataFrame | WrapAccessor:
        """
        Applies the summary method to strategy results.

        This method summarizes strategy results by concatenating signals, 
        predictions, and directions into a single DataFrame.

        Parameters
        ----------
        signals : pd.DataFrame
            DataFrame containing strategy signals.
        predictions : pd.DataFrame, optional
            DataFrame containing strategy predictions.
        market_returns : pd.DataFrame, optional
            DataFrame containing market returns.
        **weights
            Additional weights components to include in the summary.

        Returns
        -------
        pd.DataFrame | WrapAccessor
            If accessors is set to False, it returns a DataFrame summarizing 
            the strategy results. Otherwise, It returns a pandas accessors.
        """
        summary_components = {
            'Signal': signals.fillna(0),
            'Prediction': (
                predictions
                if predictions is not None and not isinstance(predictions, str)
                else signals.replace({1: predictions, -1: predictions})
            ),
            'Market': (
                market_returns
                if market_returns is not None
                else pd.DataFrame().reindex_like(signals)
            ),
            'Direction': (
                signals
                .replace(dict(map(reversed, asdict(self.scores).items())))
            )
        }
        # Add weights if provided
        summary_components.update(
            {str(name): w for name, w in weights.items()})
        # Concat all
        # Stack with level names (useful if dataframe is a pandas MultiIndex)
        summary = pd.concat([
            comp.stack(signals.columns.names)
            for comp in summary_components.values() if comp is not None
        ],
            keys=summary_components.keys(),
            axis=1
        )
        # Clean summary
        summary = (
            summary
            .set_index('Signal', append=True)
            .sort_index(level=['Date', 'Signal'], ascending=[True, False])
            .reset_index(level='Signal')
        )

        if self.accessor:
            if isinstance(market_returns, pd.DataFrame):
                return summary.statespace(market_returns)

            else:
                raise ValueError(
                    "When `accessor` is True, `market_returns` should be passed."
                )

        return summary
