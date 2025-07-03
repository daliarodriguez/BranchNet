# Adapted from https://github.com/SquareResearchCenter-AI/BEExAI
import argparse
import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from beexai.dataset.dataset import Dataset
from beexai.dataset.load_data import load_data
from beexai.utils.time_seed import set_seed
from beexai.utils.sampling import stratified_sampling
from beexai.evaluate.metrics.get_results import get_all_metrics
from train import Trainer
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path",
    type=str,
    default="output/benchmarks",
    help="Path to folder to save results",
)
parser.add_argument(
    "--config_path",
    type=str,
    default="config/tabular_openml/multi_class",
    help="Path to folder to save results",
)
args = parser.parse_args()
SAVE_PATH = args.save_path
if not os.path.exists(f"{SAVE_PATH}/models"):
            os.makedirs(f"{SAVE_PATH}/models")

CONFIG_PATH = args.config_path

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

all_config_names = glob.glob(f"{CONFIG_PATH}/*.yaml")

for path in all_config_names:
    DATA_NAME = path.split("/")[-1].split(".")[0].replace("\\", "_")
    print("Dataset: ", DATA_NAME)
    with open(f"{SAVE_PATH}/models/{DATA_NAME}.txt", 'a') as f:
        for i,SEED in enumerate(range(10)):
            set_seed(SEED)
            torch.manual_seed(SEED)
            check_random_state(SEED)
            np.random.seed(SEED)
            data_test, target_col, task, _ = load_data(
                from_cleaned=True, config_path=path, keep_corr_features=True
            )
            data = Dataset(data_test, target_col)
            X_train, X_test, y_train, y_test = data.get_train_test(
                test_size=0.3, scaler_params={"y_scaler_name": "labelencoder"}
            )
            X_train=X_train.values
            X_test=X_test.values
            y_train = y_train.values.reshape(-1)
            y_test =y_test.values.reshape(-1)

            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33,random_state=SEED)

            num_labels = data.get_classes_num(task)

            n_samples, n_features = X_train.shape

            if i ==0:
                print("Labels:",num_labels, "Features:",n_features,"Train:", n_samples,"Test:", y_test.shape[0],"Val:", y_val.shape[0], file=f)
                print("Labels:",num_labels, "Features:",n_features,"Train:", n_samples,"Test:", y_test.shape[0],"Val:", y_val.shape[0])

            for MODEL_NAME in ["XGBClassifier","BranchNet"]:
                PARAMS = {}
                trainer = Trainer(MODEL_NAME, task, PARAMS, device)

                loaded=False
                if MODEL_NAME=="BranchNet":
                    if glob.glob(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.pt"):
                        trainer.load_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.pt")
                        loaded=True
                else:
                    if glob.glob(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.joblib"):
                        trainer.load_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.joblib")
                        loaded=True

                if not loaded:
                    loss_file=f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.png"
                    trainer.train(X_train, y_train,X_val,y_val,SEED,loss_file=loss_file)

                if MODEL_NAME == "BranchNet":
                    trainer.save_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.pt")
                    trainer.model.eval()
                else:
                    trainer.save_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.joblib")

                perf_metric = trainer.get_metrics(X_test, y_test)
                if MODEL_NAME == "BranchNet":
                    print(MODEL_NAME, "performance", perf_metric, "hidden_neurons:", trainer.model.w1.data.shape[0], file=f)
                    print(MODEL_NAME, "performance", perf_metric, "hidden_neurons:", trainer.model.w1.data.shape[0])
                else:
                    print(MODEL_NAME,"performance", perf_metric, file=f)
                    print(MODEL_NAME,"performance", perf_metric)
                torch.cuda.empty_cache()
        f.close()
