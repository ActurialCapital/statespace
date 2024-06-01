from dataclasses import dataclass


@dataclass
class Listed:
    """
    A dataclass for storing categorical spaces in the configuration file.

    This class is designed to hold a list of parameters defining
    categorical spaces. It is a simple container for these parameters,
    facilitating access and manipulation of categorical space definitions
    through the configuration file.

    Parameters
    ----------
    params : list
        A list containing the parameters that define the categorical spaces.
        Each element in the list represents a distinct categorical space,
        characterized by its own set of properties or constraints.

    Examples
    --------
    With `sklearn` estimators
    ```pycon
    >>> from sklearn.linear_model import LinearRegression, Ridge
    >>> from statespace.tools import Listes
    >>> Listed([LinearRegression(), Ridge()])
    ```
    """
    params: list

    def __repr__(self):
        return f"Listed(params={self.params})"


@dataclass
class Nested:
    """
    A dataclass for storing layered or tiered parameter spaces in the 
    configuration file.

    This class encapsulates a dictionary of parameters that define a tiered
    space used in optimization problems where parameters have a
    hierarchical or tiered relationship. The class supports aggregation,
    allowing the combination of multiple tiered spaces into a single,
    coherent structure in the configuration file.

    Parameters
    ----------
    params : dict
        A dictionary where each key-value pair represents a parameter and its
        respective space definition. The space definition can be a range,
        set of values, or another structured definition that specifies the
        valid or interesting region for that parameter in the space.
    aggregate : bool, optional
        A flag indicating whether the parameter spaces should be aggregated.
        If set to True, it implies that the spaces defined by the `params`
        dictionary are to be considered collectively as part of a larger,
        aggregated space. Defaults to False, treating each space in `params`
        independently.

    Examples
    --------
    ```pycon
    >>> from optuna.distributions import IntDistribution
    >>> from statespace.tools import Nested
    ```

    With splitter
    ```pycon
    >>> from opendesk import Splitter
    >>> Nested({
    ...     Splitter: {
    ...         'n_train': IntDistribution(1, 10), 
    ...         'n_test': 1
    ...     }
    ... }
    ```
    
    With preprocessors
    ```pycon
    >>> from opendesk import Factors
    >>> Nested({
    ...     Factors: {'shift_by': IntDistribution(1, 4)}
    ... }
    ```

    With transformers
    ```pycon
    >>> from opendesk import QuantileRanks, Signal
    >>> study.Nested({
    ...     QuantileRanks: {'number_q': IntDistribution(2, 5)},
    ...     Signal: {'higher_is_better': True}
    ... },
    ...     aggregate=True
    ... )
    ```

    """
    params: dict
    aggregate: bool = False

    def __repr__(self):
        return f"Nested(params={self.params}, aggregate={self.aggregate})"
