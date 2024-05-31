from statespace import BaseStudy, run_study
from optuna.trial import Trial


class MyCustomObjective(BaseStudy):
    def __init__(self, custom_parameter):
        self.custom_parameter = custom_parameter

    @run_study
    def objective(self, trial: Trial) -> float:
        return self.model


if __name__ == "__main__":

    custom_parameter = {...}
    custom_objective = MyCustomObjective(custom_parameter)
    custom_objective.execute(n_trials=100)
    # <Study.Study at 0x17427b990>
