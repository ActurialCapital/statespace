from typing import Dict, Any, Callable
from optuna.trial import Trial

from statespace.base import BaseStudy
from statespace.tools import Listed, Nested
from statespace.decorators import run_study

ConfigStudy = Dict[str, Listed | Nested | str | float | int | None]
ModelKwargs = Dict[str, Any]


class Performance(BaseStudy):
    """
    Objective function which optimize for best performance over the period.

    Parameters
    ----------
    config : ConfigStudy
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
    model_kwargs : ModelKwargs
        Model keyword arguments.
    **create_study_kwargs
        Optuna `create_study` arguments:

        * `storage`: (`str | storages.BaseStorage | None`)
        * `sampler`: (`'samplers.BaseSampler' | None`)
        * `pruner`: (`pruners.BasePruner | None`)
        * `study_name`: (`str | None`) 
        * `direction`: (`str | StudyDirection | None`)
        * `load_if_exists`: (`bool`)
        * `directions`: (`Sequence[str | StudyDirection] | None`) 

        More information could be found in the [optuna documentation](https://optuna.readthedocs.io/en/).

    """

    def __init__(
        self,
        config: ConfigStudy,
        strategy: Callable,
        *data,
        model_kwargs: ModelKwargs = None,
        **create_study_kwargs
    ):
        super().__init__(config, strategy, *data, **create_study_kwargs)
        self.model_kwargs = model_kwargs or {}

    @run_study
    def objective(self, trial: Trial):
        """
        Objective function for the optimization performance.

        Returns
        -------
        float
            The performance metric of the evaluated trial.
        """
        return self.summary.performance(**self.model_kwargs).iloc[-1, 0]


# Other objectives can be developed here
# ...
