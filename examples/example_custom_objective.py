import numpy as np
from optuna.trial import Trial
from statespace.base import BaseStudy
from statespace.decorators import run_study


class TemplateObjective(BaseStudy):
    @run_study
    def objective(self, trial: Trial) -> float:
        return np.random.normal()


def template_strategy() -> None:
    return


if __name__ == "__main__":

    config = {}

    # Initialize Template objective
    template_objective = TemplateObjective(config, template_strategy)

    # Execute
    template_objective.execute(n_trials=100)
    # <Study.Study at 0x17427b990>
