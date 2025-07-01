# Adapted from https://github.com/SquareResearchCenter-AI/BEExAI/src/beexai/training/
"""Training models and evaluating their performance."""

from typing import Callable, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,roc_curve,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score, roc_auc_score)
from beexai.utils.convert import convert_to_numpy
from beexai.utils.path import get_path
from BranchNetFramwork import BranchNetModel


class Trainer:
    """Trainer class
    """

    def __init__(
        self,
        model_name: str,
        task: str,
        model_params: Optional[dict] = None,
        device: str = None,
    ):
        assert task in ["classification"], f"Task must be classification, got {task}"
        self.models = {
            "XGBClassifier": XGBClassifier,
            "BranchNet": BranchNetModel
        }
        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}
        self.task = task
        self.device = device
        assert (
            self.model_name in self.models
        ), f"Model name must be one of {self.models.keys()}"
        if model_name == "BranchNet":
            self.model = BranchNetModel(**self.model_params, device=device, task=task)
        else:
            self.model = self.models[model_name](**self.model_params)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        seed: int,
        loss_file: Optional[str] = None,   
    ) -> Callable:
        """Perform training on the whole training set.
        Returns:
            callable: trained model
        """
        if self.model_name == "BranchNet":
            n_samples, n_features = x_train.shape
            n_labels = len(list(set(list(y_train.ravel()))))
            the_number = round(np.log2( n_features ))+4
            n_estimators = n_labels+round(np.log2( n_features ))
            print(n_estimators,"trees, with ",2**the_number,"leaves (maximum)",the_number)
            tree_ensemble = ExtraTreesClassifier(n_estimators=n_estimators,max_leaf_nodes=2**the_number,random_state=seed)
            tree_ensemble.fit(x_train, y_train.ravel())   
            self.model.build_model_from_ensemble(tree_ensemble)
            self.model = self.model.fit(
                x_train,
                y_train,
                x_val,
                y_val,
                loss_file=loss_file,
            )
        else:
            self.model.fit(x_train, y_train)
        return self.model

    def get_metrics(self, x:np.ndarray, y:np.ndarray) -> dict:
        """Get metrics for the model. Accuracy and f1 score for
        classification, mse and r2 score for regression.
        Returns:
            dict: dictionary of metrics
        """
        pred = self.model.predict(x)
        pred = convert_to_numpy(pred)
        metrics = {
            "accuracy": round(accuracy_score(pred,y),6),
            "f1 score": round(f1_score(pred, y, average="weighted"),6),
        }
        return metrics

    def save_model(self, path: str):
        """Save the model"""
        path = get_path(path, check_dir=True)
        is_nn = self.model_name == "BranchNet"
        if is_nn:
            torch.save(self.model.state_dict(), path)
        else:
            joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load the model"""
        path = get_path(path)
        is_nn = self.model_name == "BranchNet"
        if is_nn:
            fn_dict = torch.load(path,weights_only=True)
            self.model.build_from_dict(fn_dict)
            self.model.load_state_dict(fn_dict)
            self.model.eval()
            self.model.to(self.device)
        else:
            self.model = joblib.load(path)


