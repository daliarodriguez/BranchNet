import pandas as pd
import re
import os
import scipy.stats as stats
import statistics
import glob
import torch

performance_summary = []
architecture_summary =[]
glob.glob("*.txt")

for filename in glob.glob("*.txt"):

    dataset_name = filename.replace(".txt", "")
    
    if not os.path.exists(filename):
        print(f"Warning: File not found for {filename}. Skipping.")
        continue

    with open(filename, 'r') as f:
        content = f.read()

    lines = content.strip().split('\n')

    # Temporary lists to hold performance for each run (seed)
    xgb_accs = []
    xgb_f1s = []
    bn_accs = []
    bn_f1s = []
    bn_hns = []

    for line in lines:
        if "Train" in line:
            L = int(re.search(r"Labels: (\d+)", line).group(1))
            F = int(re.search(r"Features: (\d+)", line).group(1))
            Tr = int(re.search(r"Train: (\d+)", line).group(1))
            Te = int(re.search(r"Test: (\d+)", line).group(1))
            V = int(re.search(r"Val: (\d+)", line).group(1))
        if "XGBClassifier performance" in line:
            acc_match = re.search(r"'accuracy': (\d+\.\d+)", line)
            f1_match = re.search(r"'f1 score': (\d+\.\d+)", line)
            if acc_match and f1_match:
                xgb_accs.append(float(acc_match.group(1)))
                xgb_f1s.append(float(f1_match.group(1)))
        elif "BranchNet performance" in line:
            acc_match = re.search(r"'accuracy': (\d+\.\d+)", line)
            f1_match = re.search(r"'f1 score': (\d+\.\d+)", line)
            hn_match = re.search(r"hidden_neurons: (\d+)", line)
            if acc_match and f1_match:
                bn_accs.append(float(acc_match.group(1)))
                bn_f1s.append(float(f1_match.group(1)))
            if hn_match:
                bn_hns.append(int(hn_match.group(1)))

    num_runs = len(bn_accs)

    for i in range(num_runs):
        model_dict = torch.load(dataset_name+"_BranchNet_"+str(i)+".pt",weights_only=True)
        w1 = model_dict['w1'] 
        s1=(w1==0).sum().item()/(w1.shape[0]*w1.shape[1])
        w2 = model_dict['w2'] 
        s2=(w2==0).sum().item()/(w2.shape[0]*w2.shape[1])
        if i==0:
            xgb_acc = []
            bn_acc = []
            xgb_f1 = []
            bn_f1 = []
            sparsity1 = []
            sparsity2 = []
            hidden = []
        hidden.append(bn_hns[i])
        xgb_acc.append(xgb_accs[i])
        bn_acc.append(bn_accs[i])
        xgb_f1.append(xgb_f1s[i])
        bn_f1.append(bn_f1s[i])
        sparsity1.append(s1)
        sparsity2.append(s2)
        if i==num_runs-1:
            p_acc = round(stats.wilcoxon(xgb_acc, bn_acc)[1],3)
            p_f1 = round(stats.wilcoxon(xgb_f1, bn_f1)[1],3)
            xgb_m_acc = round(statistics.mean(xgb_acc),3)
            xgb_m_f1 = round(statistics.mean(xgb_f1),3)
            xgb_s_acc = round(statistics.stdev(xgb_acc),3)
            xgb_s_f1 = round(statistics.stdev(xgb_f1),3)
            bn_m_acc = round(statistics.mean(bn_acc),3)
            bn_m_f1 = round(statistics.mean(bn_f1),3)
            bn_s_acc = round(statistics.stdev(bn_acc),3)
            bn_s_f1 = round(statistics.stdev(bn_f1),3)
            s_w1_m = round(min(sparsity1),3)
            s_w2_m = min(sparsity2)
            s_w1_M = round(max(sparsity1),3)
            s_w2_M = max(sparsity2)
            h_min = min(hidden)
            h_max = max(hidden)
            
            if xgb_m_acc > bn_m_acc: 
                winner = "XGBoost"
            else:
                winner = "BranchNet"
                
            performance_summary.append({
                'Dataset': dataset_name,
                'model': "BranchNet",
                'mean': bn_m_acc,
                'std': bn_s_acc,
                'p_val': p_acc,
                'winer': winner
            })
            performance_summary.append({
                'Dataset': '',
                'model': "XGBoost",
                'mean': xgb_m_acc,
                'std': xgb_s_acc,
                'p_val': '',
                'winer': '' 
            })
            architecture_summary.append({
                'Dataset': dataset_name,
                '# feats': F,
                '# samples': Tr+Te+V,
                'h_min': h_min,
                'h_max': h_max,
                'w1_s_min': s_w1_m*100,
                'w1_s_max': s_w1_M*100,
                'w2_s_min': s_w2_m*100,
                'w2_s_max': s_w2_M*100
            })

# Create Pandas DataFrames and save to CSV
df = pd.DataFrame(performance_summary)
csv_output_filename = "performance_summary.csv"
df.to_csv(csv_output_filename, index=False)
print(f"CSV file '{csv_output_filename}' generated successfully with {len(df)} rows.")
df = pd.DataFrame(architecture_summary)
csv_output_filename = "branchnet_summary.csv"
df.to_csv(csv_output_filename, index=False)
print(f"CSV file '{csv_output_filename}' generated successfully with {len(df)} rows.")
