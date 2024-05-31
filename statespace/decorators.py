def run_study(method):
    """
    Decorator to prepare the trial by setting up components and creating the 
    DataFrame and accessing the accessor necessary for the objective function.

    Parameters
    ----------
    method : callable 
        The original objective method of the objective class.
    accessor : bool, optional (WIP)
        if True, set model attribute to the `statespace` an accessor. Ottherwise
        set model attribute to the provided strategy result.
    
    Returns
    -------
    Callable: 
        A wrapped objective method.
    """

    def wrapper(self, trial, *args, **kwargs):
        # Perform the setup and create actions
        comps = self.setup(trial)
        self.model = self.create(**comps)
        return method(self, trial)
    
    return wrapper