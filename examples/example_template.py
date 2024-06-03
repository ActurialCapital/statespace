import numpy as np
from optuna.trial import Trial
from statespace.base import BaseStudy
from statespace.decorators import run_study


class Template(BaseStudy):
    @run_study
    def objective(self, trial: Trial) -> float:
        return np.random.rand()


def strategy() -> None:
    return


if __name__ == "__main__":

    config = {}

    # Initialize Template objective
    template_objective = Template(config, strategy)

    # Execute
    template_objective.execute(n_trials=100)
    # <Study.Study at 0x17427b990>
