import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Any, Optional, Union
from beexai.utils.convert import convert_to_tensor
from tqdm import tqdm
from BranchNet import BranchNet
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        if self.y is not None:
            y = self.y[idx]
            return x, y
        else:
            return x

    def __len__(self):
        return self.X.shape[0]


class BranchNetModel(BranchNet):
    """Inherit from BranchNet to inherit forward and
    create fit and predict methods.
    """

    def __init__(
        self,
        task: str="classification",
        device: str = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.task = task
        self.dtype = dtype
        
    def train_step(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        criterion: nn.modules.loss,
        optimizer: torch.optim.Optimizer,
        p: float = 0.4
    ) -> float:
        """Train the model for one epoch.
        """
        dataset = TabularDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=min(256,dataset.__len__()), shuffle=True,drop_last=True)#630
        loss_sum = 0
        for x, y in dataloader:
            y = y.to(self.device)
            y_pred = self.forward(x.to(self.device)) 
            y = y.squeeze(1).long()
            ce = criterion(y_pred,y) 
            pt = torch.exp(-ce)  # pt = prob of correct class
            f = 0.5 * (1 - pt) ** 2.5 * ce
            loss = p*f.mean() + (1-p)*ce.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        return loss_sum / len(dataloader) #(i+1)

    def val_step(
        self,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        criterion: nn.modules.loss,
        p: float = 0.4
    ) -> float:
        """Validate the model for one epoch.
        """
        dataset = TabularDataset(x_val, y_val)
        dataloader = DataLoader(dataset, batch_size=min(256,dataset.__len__()), shuffle=True,drop_last=True)
        loss_sum = 0
        self.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                y = y.to(self.device)
                y_pred = self.forward(x.to(self.device))
                y = y.squeeze(1).long()
                ce = criterion(y_pred,y) 
                pt = torch.exp(-ce)
                f = 0.5 * (1 - pt) ** 2.5 * ce
                loss = p*f.mean() + (1-p)*ce.mean()
                loss_sum += loss.item()
        self.train()
        return loss_sum / len(dataloader)#(i+1)
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        learning_rate = 0.01,
        epochs = 1500,
        loss_file: Optional[str] = None,
    ) -> Any:
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).reshape(-1,1)
        x_val = torch.from_numpy(x_val).float()
        y_val = torch.from_numpy(y_val).reshape(-1,1)
        y_train = y_train.float()
        y_val = y_val.float()
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        min_val_loss=10000000
        patience = 0
        max_patience=100
        progress_bar = tqdm(range(epochs))
        loss_history = []
        val_loss_history = []
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=180)
        for i,_ in enumerate(progress_bar):
            loss = self.train_step(x_train, y_train, criterion, optimizer)
            progress_bar.set_description(f"Loss: {loss:.6f}")
            loss_history.append(loss)
            val_loss = self.val_step(x_val, y_val, criterion)
            val_loss_history.append(val_loss)
            scheduler.step(val_loss)
            if val_loss<min_val_loss:
                min_val_loss = val_loss
                torch.save(self.state_dict(), "temporal.pt")
                patience = 0
            else:
                patience+=1
            if patience==max_patience:
                break
        del scheduler
        
        _ = plt.figure(figsize=(12, 8))
        plt.plot(loss_history[5:], label="train")
        if i<epochs-1:
            self.load_state_dict(torch.load("temporal.pt",weights_only=True))
        plt.plot(val_loss_history[5:], label="val")
        plt.legend()
        if loss_file is not None:
            plt.savefig(loss_file)
        plt.close()

        return self

    def predict(
        self, x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        x_test_copy = convert_to_tensor(x_test).float()
        dataset = TabularDataset(x_test_copy)
        dataloader = DataLoader(dataset, batch_size=min(200,dataset.__len__()), shuffle=False)
        res = []
        with torch.no_grad():
            for x in dataloader:
                res1 = torch.softmax(self.forward(x.to(self.device)),dim=1)
                res.append(res1.to('cpu'))
            res = torch.vstack(res)
            res= torch.argmax(res,dim=1)
        return res

    def predict_proba(self, x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        x_test_copy = convert_to_tensor(x_test).float().to(self.device)
        dataset = TabularDataset(x_test_copy)
        dataloader = DataLoader(dataset, batch_size=min(200,dataset.__len__()), shuffle=False)
        res = []
        with torch.no_grad():
            for x in dataloader:
                res1 = torch.softmax(self.forward(x.to(self.device)),dim=1)
                res.append(res1.to('cpu'))
            res = torch.vstack(res)
        return res
