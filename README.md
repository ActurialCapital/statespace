<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<p align="center"><img src="docs/static/logo.png" alt="logo" width="80%" height="80%"></p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
        <ul>
            <li><a href="#introduction">Introduction</a></li>
        </ul>
        <ul>
            <li><a href="#why-optuna">Why Optuna?</a></li>
        </ul>
        <ul>
            <li><a href="#built-with">Built With</a></li>
        </ul>
    </li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#reference">Reference</a></li>
    
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

### Introduction

`statespace` is an automated portfolio construction process using advanced `optuna` fine-tuning techniques.

> `optuna` is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.

* **Eager search spaces**: Automated search for optimal hyperparameters using Python conditionals, loops, and syntax
* **State-of-the-art algorithms**: Efficiently search large spaces and prune unpromising trials for faster results
* **Easy parallelization**: Parallelize hyperparameter searches over multiple threads or processes without modifying code

`statespace` is designed to automatically optimize portfolio hyper-parameters base on a configuration file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Why Optuna?

Optuna enables efficient hyperparameter optimization by adopting state-of-the-art algorithms for sampling hyperparameters and pruning efficiently unpromising trials.

* **Sampling Algorithms**: *Samplers basically continually narrow down the search space using the records of suggested parameter values and evaluated objective values, leading to an optimal search space which giving off parameters leading to better objective values. `optuna` provides the following sampling algorithms:*
  * *Grid Search*
  * *Random Search*
  * *Tree-structured Parzen Estimator algorithm*
  * *CMA-ES based algorithm*
  * *Gaussian process-based algorithm*
  * *Algorithm to enable partial fixed parameters*
  * *Nondominated Sorting Genetic Algorithm II*
  * *A Quasi Monte Carlo sampling algorithm*
  
* **Pruning Algorithms**: *Pruners automatically stop unpromising trials at the early stages of the training (a.k.a., automated early-stopping). Optuna provides the following pruning algorithms:*
  * *Median pruning algorithm*
  * *Non-pruning algorithm*
  * *Algorithm to operate pruner with tolerance*
  * *Algorithm to prune specified percentile of trials*
  * *Asynchronous Successive Halving algorithm*
  * *Hyperband algorithm*
  * *Threshold pruning algorithm*
  * *A pruning algorithm based on Wilcoxon signed-rank test*

More information could be found in the [Official Documentation](https://optuna.readthedocs.io/en/stable/tutorial/index.html).

### Built With

* `optuna = "^3.6.1"`
* `numpy = "^1.26.4"`
* `pandas = "^2.2.2"`
* `plotly = "^5.22.0"`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Installation

```sh
$ pip install python-statespace
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Getting Started

Follow these steps to get started with developing an investment strategy using the `statespace` framework.

### Prerequisites

Make sure you have `sklearn` installed to follow this example:

```shell
$ pip install scikit-learn
```

### Steps to Get Started

* Develop an Investment Strategy
* Create a Custom Objective Object
* Set Up a Configuration File with Hyperparameters
* Execute the Trials
* Analyze the Results
* Visualize

### Example

In this example, we create a simple `sklearn` pipeline with a processor and a regressor.

#### Import Required Modules

```python
>>> from statespace.base import BaseStudy, Listed
>>> from statespace.decorators import run_study
```

Note: `sklearn` imports are not shown here. Please see our running examples in the examples folder for more details.

#### Develop an Investment Strategy

```python
>>> def strategy(estimator: BaseEstimator, preprocessor: BaseEstimator) -> Pipeline:
...     return make_pipeline(preprocessor, estimator)
```

#### Create a Custom Objective Object

We then create a custom objective `MyCustomObjective` to minimize the mean squared error on the validation set:

```python 
>>> class MyCustomObjective(BaseStudy):
...     @run_study
...     def objective(self, trial: Trial) -> float:
...         X_train, X_test, y_train, y_test = train_test_split(*self.data, test_size=0.33)
...         pred = self.model.fit(y_train, X_train).predict(y_test)
...         return mean_squared_error(y_test, pred)
```

#### Set Up a Configuration File with Hyperparameters

```python
>>> config = {
...     'preprocessor': Listed([MinMaxScaler(), RobustScaler(), StandardScaler()]),
...     'estimator':    Listed([LinearRegression(), Ridge(), Lasso(), ElasticNet()]),
... }
```

#### Execute the Trials

```python
>>> custom_model = MyCustomObjective(config, strategy, X, y)
>>> model = custom_model.execute(n_trials=20)
# A new study created in memory with name: statespace
```

#### Analyze Results

Get best parameters

```python
>>> print(model.best_trial.params)
# {'preprocessor': MinMaxScaler(), 'estimator': ElasticNet()}
```

Get best value

```python
>>> print(model.best_value)
# 0.9758159549070534.
```

#### Visualize

```python
>>> from optuna import visualization
```

Contour Plot

```python
>>> fig = visualization.plot_contour(model, params=["estimator", "preprocessor"])
>>> fig.show()
```
<p align="center"><img src="docs/static/example_contour_plot.png" alt="chart-1""></p>

Hyperparameter Importance
```python
>>> fig = visualization.plot_param_importances(model)
>>> fig.show()
```
<p align="center"><img src="docs/static/example_importance_plot.png" alt="chart-2""></p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the BSD-3 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Reference -->
## Reference

* Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
* Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.
