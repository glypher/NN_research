"""features.py: Helper classes to evaluate models
and compare results using statistical methods
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"


from .model import ModelHistory, ModelHistorySet, DataType

import seaborn as sns
import pandas as pd


class ModelEvaluation:
    def __init__(self, models: ModelHistorySet):
        self._models = models
        self._filter_params = set()

    def set_filter_params(self, params):
        self._filter_params = set(params)
        return self

    def plot_distributions(self, metric, type:DataType, metric_func=lambda x: max(x)):
        """Plot the distribution of all models
        Get all models with the same parameter that represents multiple train runs
        Apply the metric function to get the corresponding metric from each (e.x. max accuracy)"""

        data = []
        for params, histories in self._models.same_histories(self._filter_params).items():
            for h in histories:
                data.append([params, metric_func(h.history(metric, type))])
        name = "{}_{}".format(type.name, metric)
        sns.displot(pd.DataFrame(data, columns=['type', name]),
                    x=name, hue="type", kde=True)

        return self
