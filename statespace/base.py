import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Callable
import pandas as pd

from optuna import create_study
from optuna.trial import Trial
from optuna.study import Study
from optuna.distributions import BaseDistribution, IntDistribution, CategoricalDistribution

from statespace.tools import Listed, Nested

# Suppress all warnings
warnings.filterwarnings('ignore')


class BaseOptunaHandler(ABC):
    """Abstract builder for creating components."""

    def __init__(self, trial: Trial):
        self.trial = trial

    @abstractmethod
    def handle(self, params):
        """Abstract method to parameters."""
        pass

    def optuna_wrapper(
        self,
        name: str,
        distribution: BaseDistribution
    ) -> None | bool | int | float | str:
        """
        Suggest a parameter value for a given trial based on the distribution 
        type.

        Parameters
        ----------
        trial : Trial
            The trial for which to suggest the parameter.
        name : str
            The name of the parameter.
        distribution : BaseDistribution | list
            The distribution of the parameter. Can be an Optuna distribution 
            or a list of choices.

        Returns
        -------
        None | bool | int | float | str
            The suggested parameter value.
        """
        if isinstance(distribution, list):
            suggested = self.trial.suggest_categorical(
                name, distribution)

        elif isinstance(distribution, IntDistribution):
            suggested = self.trial.suggest_int(
                name, distribution.low, distribution.high)

        elif isinstance(distribution, CategoricalDistribution):
            suggested = self.trial.suggest_categorical(
                name, distribution.choices)

        else:
            suggested = distribution  # Handling constant

        return suggested


class ListedHandler(BaseOptunaHandler):
    """
    Handles building components from lists.

    This handler is responsible for constructing components based on 
    list of parameters. Specifically, it iterates through the provided 
    list (coming from a Listed instance), using the values to determine the 
    hyperparameters and their distributions.

    """

    def handle(self, name: str, params: Any | List[Any]) -> Any:
        """
        Returns a parameter value directly or suggests a categorical choice 
        from a list.

        Parameters
        ----------
        name : str
            The name of the parameter.
        params : Any | List[Any]
            The parameter values or a list of values to choose from.

        Returns
        -------
        Any
            The constant value or a suggested value from the list.
        """
        return (
            self.optuna_wrapper(name, params)
            if isinstance(params, list)
            else params
        )


class NestedHandler(BaseOptunaHandler):
    """
    Handles building components from dictionaries.

    This handler is responsible for constructing components based on 
    dictionaries of parameters. Specifically, it iterates through the provided 
    dictionary (coming from a Nested instance), using the keys to identify 
    components and the values to determine the hyperparameters and their 
    distributions. If a value in this dictionary is another Nested instance,
    NestedHandler recognizes this nested structure and processes it 
    accordingly, allowing for recursive handling of nested layers.
    """

    def handle(
        self,
        params: Dict[Type, Dict[str, Any]],
        aggregate: bool = False
    ) -> Any:
        """
        Builds components from a dictionary of parameters, optionally 
        aggregating them.

        The integration point between NestedHandler and Nested occurs in the 
        NestedHandler.handle method, where it checks if a distribution 
        (hyperparameter space) is an instance of Nested. If so, it processes the 
        nested Nested by iterating over its params, applying hyperparameter 
        suggestions through Optuna, and constructing the sub-components 
        recursively. This allows for complex, hierarchical model configurations
        where components can have their sub-components, each with its own 
        hyperparameter space defined through Nested instances.

        Parameters
        ----------
        params : Dict[Type, Dict[str, BaseDistribution]]
            The dictionary mapping component classes to their parameters.
        aggregate : bool, optional
            Whether to aggregate the components into a list or return the first 
            component, by default False.

        Returns
        -------
        List[Any] | Any
            The list of components if aggregated, otherwise a single component.
        """
        components = [
            self.create_component(model, values)
            for model, values in params.items()
        ]
        return tuple(components) if aggregate else components[0]

    def create_component(self, model: Type, values: Dict[str, Any]) -> Any:
        """
        Create a single component (model instance) from given parameters, 
        handling nested parameters recursively.

        Parameters
        ----------
        model : Type
            The class of the model to instantiate.
        params : Dict[str, Any]
            The parameters for the model, potentially including nested 
            parameters for nested models.

        Returns
        -------
        Any
            An instance of the model initialized with the given parameters.
        """
        comps = {}
        for name, dist in values.items():
            # If dist is a nested layer
            if isinstance(dist, Nested):
                for sub_model, v in dist.params.items():
                    sub_comps = {
                        n: self.optuna_wrapper(n, d) for n, d in v.items()
                    }
                suggested = sub_model(**sub_comps)
            else:
                suggested = self.optuna_wrapper(name, dist)
            comps[name] = suggested
        return model(**comps)


class BaseStudy(ABC):
    """
    Base class to  configure components for a trial based on provided 
    parameters space. 

    This class encapsulates the process of setting up, executing an 
    optimization study using Optuna and a predefined objective function
    for model evaluation.

    Parameters
    ----------
    config : Dict[str, Listed | Nested]
        The configuration dictionary for the trial components. This dictionary 
        specifies how each component of the study should be configured. The 
        configuration values are either `Listed`, a list of categorical 
        spaces, `Nested`, a dict of layered parameter spaces or any other 
        constants.
    strategy : Any
        Stategy implementation function.
    *data : tuple
        Variable length argument list for the data to be used in the study. 
        This could include any form of data necessary for the blueprint and 
        summarizer, such as datasets for training and testing.
    **create_study_kwargs
        Optuna `create_study` arguments:

        * `storage`: (`str | storages.BaseStorage | None`)
        * `sampler`: (`'samplers.BaseSampler' | None`)
        * `pruner`: (`pruners.BasePruner | None`)
        * `study_name`: (`str | None`) 
        * `direction`: (`str | StudyDirection | None`)
        * `load_if_exists`: (`bool`)
        * `directions`: (`Sequence[str | StudyDirection] | None`) 

        More information could be found in the 
        [optuna documentation]
        (https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna-study-create-study).

    Methods
    -------
    create
        Creates strategy based on the trial components and additional 
        configuration.
    execute
        Executes the optimization study over a specified number of trials.
    objective
        Abstract method to define the objective function to optimize.
    setup
        Sets up the trial components based on the configuration.

    Examples
    --------
    ```pycon
    >>> from optuna.distributions import IntDistribution
    >>> from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    >>> from sklego.meta import EstimatorTransformer
    >>> from opendesk.transformers import QuantileRanks, Signal
    >>> from opendesk.portfolio import EqualWeighted, MarketCapWeighted
    >>> from opendesk.splitter import Splitter
    >>> from opendesk.strategy import strategy
    >>> from statespace import Listed, Nested, Performance
    ```

    Create a configuration file
    ```pycon
    >>> config = {
    ...     'pipeline': Nested({
    ...         EstimatorTransformer: {
    ...             'estimator': [LinearRegression(), Ridge(), Lasso(), ElasticNet()], # (1)
    ...         },
    ...         QuantileRanks: {'number_q': IntDistribution(2, 5)},
    ...         Signal: {'higher_is_better': True}, # (2)
    ...     },
    ...         aggregate=True # (3)
    ...     ),
    ...     'expected_returns': 'ema_historical_return', # (4)
    ...     'allocators': Listed([EqualWeighted(), MarketCapWeighted(mcaps)]),
    ...     'splitter': Nested({
    ...         Splitter: {
    ...             'n_train': IntDistribution(1, 10),
    ...             'n_test': 1 # (5)
    ...         }
    ...     })
    ... }
    ```

    1.  Estimators in this context are treated as categorical inputs.
    2.  Assume to be a constant.
    3.  Group as one variable in the model, with suggested hyperparamters at each layers.
    4.  Assume to be a constant.
    5.  Assume to be a constant.

    Create a study
    ```pycon
    >>> perf = Performance(config, strategy, X, y, direction="maximize")
    >>> model = perf.execute(n_trials=5)
    [I 2024-04-04 10:31:17,861] A new study created in memory...
    [I 2024-04-04 10:31:28,011] Trial 0 finished with value: 0.81...
    [I 2024-04-04 10:31:37,427] Trial 1 finished with value: 0.52...
    [I 2024-04-04 10:31:47,171] Trial 2 finished with value: 1.20...
    [I 2024-04-04 10:31:52,395] Trial 3 finished with value: 0.90...
    [I 2024-04-04 10:32:00,430] Trial 4 finished with value: 1.10...
    ```

    Get best parameters
    ```pycon
    >>> print(model.best_trial.params)
    {'estimator': LinearRegression(),
     'shift_by': 3,
     'number_q': 3,
     'allocators': MarketCapWeighted,
     'n_train': 7}
    ```

    Get best value
    ```pycon
    >>> print(model.best_value)
    1.20
    ```

    Visulatize the output
    ```pycon
    >>> from optuna import visualization
    >>> fig = visualization.plot_contour(model, params=["estimator", "number_q"])
    >>> fig.show() # (1)
    ```

    1.  Render a `plotly` chart.

    ```plotly
    {"file_path": "../../../assets/charts/contour_plot.json"}
    ```

    Test objective
    ```pycon
    >>> from optuna.trial import FixedTrial
    >>> params = FixedTrial(
    ...     dict(n_train=3, 
    ...          n_test=1, 
    ...          shift_by=1, 
    ...          number_q=4, 
    ...          higher_is_better=True,
    ...     )
    ... )
    ... Performance(params)
    1.10347850100630805
    ```
    """

    def __init__(
        self,
        config: Dict[str, Listed | Nested | str | float | int | None],
        strategy: Callable,
        *data,
        **create_study_kwargs
    ):
        self.config = config
        self.strategy = strategy
        self.data = data
        self.create_study_kwargs = create_study_kwargs

    def __repr__(self):
        """
        Special method to return the string representation of the instance,
        dynamically using the class name of the subclass.
        """
        return f"{self.__class__.__name__}({self.config})"

    @abstractmethod
    def objective(self, summary: pd.DataFrame) -> float:
        """
        Abstract method to define the objective function to optimize.

        Returns
        -------
        float
            The performance metric of the evaluated trial.
        """
        pass

    def setup(self, trial: Trial) -> Dict[str, Any]:
        """
        Sets up the trial components based on the configuration.

        Optuna Optimization Flow: In the broader context of the code, the
        BaseStudy class sets up an optimization study with Optuna. During the
        setup phase (setup method), it uses the provided configuration
        (which may include both Nested and Listed objects) to determine how to
        construct trial components. For each configuration item that is a
        Nested, it uses NestedHandler to process the parameters and possibly
        nested layers, creating a structured and parameterized model or
        pipeline component ready for evaluation in the optimization process.

        Parameters
        ----------
        trial : Trial
            The trial for which to configure components.

        Returns
        -------
        Dict[str, Any]
            A dictionary of configured components.
        """
        comps = {}
        for name, space in self.config.items():
            if isinstance(space, Listed):
                listed = ListedHandler(trial)
                comps[name] = listed.handle(name, space.params)

            elif isinstance(space, Nested):
                nested = NestedHandler(trial)
                comps[name] = nested.handle(space.params, space.aggregate)

            else:
                comps[name] = space

        return comps

    def execute(self, **optimize_kwargs) -> Study:
        """
        Executes the optimization study.

        Optimization is done by choosing a suitable set of hyperparameter 
        values from a given range. Uses a sampler which implements the task of 
        value suggestion based on a specified distribution. The sampler is 
        specified in `create_study_kwargs` and the default choice for the 
        sampler is `TPE`. See also `TPESampler` for more information.

        Optimization will be stopped when receiving a termination signal such 
        as `SIGINT` and `SIGTERM`. Unlike other signals, a trial is 
        automatically and cleanly failed when receiving `SIGINT` (Ctrl+C). If 
        `n_jobs` is greater than one or if another signal than `SIGINT` is 
        used, the interrupted trial state won’t be properly updated[^1].

        [^1]: Source: Optuna

        Parameters
        ----------
        **optimize_kwargs
            Optuna optimize arguments:

            * n_trials (`int | None`)
            * timeout (`float | None`)
            * n_jobs (`int`)
            * catch (`Iterable[type[Exception]] | type[Exception]`) 
            * callbacks (`list[Callable[[Study, FrozenTrial], None]] | None`)
            * gc_after_trial (`bool`)
            * show_progress_bar (`bool`) 

        Returns
        -------
        Study
            The completed optimization study.

        Examples
        --------
        To implement a custom objective, subclass `BaseStudy` and define 
        the `optimize` method
        ```pycon
        >>> from statespace import BaseStudy
        >>> from optuna.trial import Trial
        >>> 
        >>> class MyCustomObjective(BaseStudy):
        ...     @run_study
        ...     def objective(self, trial: Trial) -> float:
        ...         return self.model.some_metrics
        >>> 
        >>> custom_objective = MyCustomObjective(custom_params)
        >>> custom_objective.execute(n_trials=100)
        <Study.Study at 0x17427b990>
        ```
        """
        study = create_study(**self.create_study_kwargs)
        study.optimize(self.objective, **optimize_kwargs)
        return study

    def create(self, *args, **kwargs) -> Any:
        """
        Creates an implementation of a strategy based on the trial components.

        This method sets up the weights and block parameters based on the 
        provided components dictionary. It then runs the blueprint with the 
        constructed block and additional weights to generate and return the 
        `trials` DataFrame.

        Parameters
        ----------
        *args, **kwargs
            Data and feature components corresponding to an object or 
            configuration needed for the trial.

        Returns
        -------
        Any
            A summuarizer containing the results of the trials run with the 
            specified components and configurations. Its structure and content 
            will depend on the specifics of the `model.trials`  method and the 
            `summarizer` used.

        """
        return self.strategy(*args, **kwargs)
