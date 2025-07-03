import torch
from torch import Tensor
from sklearn import ensemble
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
class BranchNet(nn.Module):
    def __init__(
            self,
            task: str="classification",
            device=None,
            dtype=torch.float,
    ) -> None:
        assert (
            task == "classification"
        ), f"""BranchNet is only implemented for classification,
            found {task}"""
        super().__init__()
        self.task = task
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.hidden_neurons = 0
        self.in_features = None
        self.out_features = None
        self.bn0 = None
        self.w1 = None
        self.m1 = None
        self.bn1 = None
        self.bn2 = None
        self.w2 = None 
        
    def forward(self, input: Tensor) -> Tensor:
        x = self.bn0(input)
        if self.training: 
            x = F.linear(x, self.w1 * self.m1)
        else:
            x = F.linear(x, self.w1)
        x = self.bn1(x)
        x = F.sigmoid(x)
        x = self.bn2(x)
        x = F.linear(x, self.w2)
        return x

    def build_model_from_ensemble(self,tree_ensemble:ensemble) -> nn.Module:
        """
        gets all the necessary info from the tree ensemble
        """
        def bf_search(index,path,is_leaf):
            
            left_i = tree.children_left[index]
            right_i = tree.children_right[index]
            has_left_leaf = left_i != -1 and is_leaf[left_i]
            has_right_leaf = right_i != -1 and is_leaf[right_i]
            
            new_path=path[:]
            if tree.feature[index]>=0:
                new_path.append(tree.feature[index])
                
            if has_left_leaf or has_right_leaf:
                n_samples=tree.n_node_samples[0]
                node_samples=tree.n_node_samples[index]
                parents.append(index)  
                path_to_parent.append(new_path)
                factor = node_samples/n_samples #portion of samples in branch
                dist = factor*tree.value[index][0]
                class_proportion.append(dist)
                
            if not has_left_leaf: bf_search(left_i, new_path, is_leaf)
            if not has_right_leaf: bf_search(right_i, new_path, is_leaf)
                
                
        def get_w1(size,device,dtype):
            w1 = torch.zeros(size, dtype=dtype)
            i=0
            for t,paths_in_tree in enumerate(all_path_to_parent):
                for path in paths_in_tree:
                    w1[i][path] = feature_importance[t][path]
                    i += 1
            w1*=1 / np.sqrt(self.in_features)
            return w1.to(device)
            
        def get_w2(size,device,dtype):
            w2 = torch.zeros(size,dtype=dtype)
            i=0
            for t,proportions_in_tree in enumerate(all_proportions):
                for p,classes_involved_in_branch in enumerate(proportions_in_tree):
                    w2[:,i] = torch.from_numpy(classes_involved_in_branch)
                    i += 1       
            w2*=1 / np.sqrt(self.in_features)
            return w2.to(device)
            
        self.in_features = tree_ensemble.n_features_in_
        self.out_features = tree_ensemble.n_classes_

        all_path_to_parent = []
        feature_importance = []
        all_proportions = []
        
        estimators = tree_ensemble.estimators_
        for estimator in estimators:
            tree = estimator.tree_
            is_leaf = (tree.children_left == -1) & (tree.children_right == -1)
            path_to_parent = []
            parents = []
            class_proportion = []
            bf_search(0,[],is_leaf)
            all_path_to_parent.append(path_to_parent)
            self.hidden_neurons += len(parents)
            all_proportions.append(class_proportion)
            importance=torch.zeros(self.in_features).float()
            for path in path_to_parent:
                for feat in path:
                    importance[feat]+=1
            feature_importance.append(importance/importance.max())

        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        w1 = get_w1((self.hidden_neurons, self.in_features), device=self.device,dtype=self.dtype)
        self.m1 = (w1!=0)
        self.w1 = nn.Parameter(w1)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons,device=self.device)
        w2 = get_w2((self.out_features, self.hidden_neurons), device=self.device,dtype=self.dtype)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.w2 = nn.Parameter(w2,requires_grad=False)
        print(self.hidden_neurons,"hidden")       

    def build_from_dict(self, fn_dict) -> nn.Module:
        self.w1 = nn.Parameter(fn_dict['w1'])
        self.m1 = (self.w1 != 0)
        self.hidden_neurons = self.w1.shape[0]
        self.w2 = nn.Parameter(fn_dict['w2'],requires_grad=False)
        self.out_features = self.w2.shape[0]
        self.in_features = self.w1.shape[1]
        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)

        

