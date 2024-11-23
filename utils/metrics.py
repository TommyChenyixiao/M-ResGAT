import torch
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from numpy import interp
import matplotlib.pyplot as plt

@torch.no_grad()
def evaluate(model, data, idx, return_roc=False):
    model.eval()
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)
    pred = out[idx].argmax(dim=1)
    
    # Basic metrics
    acc = accuracy_score(data.y[idx].cpu(), pred.cpu())
    f1 = f1_score(data.y[idx].cpu(), pred.cpu(), average='macro')
    
    if return_roc:
        # Calculate ROC and AUC for multi-class
        probs_cpu = probs[idx].cpu().numpy()
        labels_cpu = data.y[idx].cpu().numpy()
        n_classes = probs_cpu.shape[1]
        
        # Binarize the labels
        labels_bin = label_binarize(labels_cpu, classes=np.arange(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs_cpu[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            labels_bin.ravel(), 
            probs_cpu.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'n_classes': n_classes
        }
        
        return acc, f1, roc_data
    
    return acc, f1

def plot_roc_curves(roc_data, model_name, save_path=None):
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curves
    # Plot micro-average ROC curve
    plt.plot(
        roc_data['fpr']["micro"],
        roc_data['tpr']["micro"],
        label=f'micro-average ROC curve (AUC = {roc_data["roc_auc"]["micro"]:0.2f})',
        color='deeppink',
        linestyle=':',
        linewidth=4,
    )
    
    # Plot macro-average ROC curve
    plt.plot(
        roc_data['fpr']["macro"],
        roc_data['tpr']["macro"],
        label=f'macro-average ROC curve (AUC = {roc_data["roc_auc"]["macro"]:0.2f})',
        color='navy',
        linestyle=':',
        linewidth=4,
    )
    
    # Plot ROC curve for each class
    n_classes = roc_data['n_classes']
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            roc_data['fpr'][i],
            roc_data['tpr'][i],
            color=color,
            linewidth=2,
            label=f'ROC curve of class {i} (AUC = {roc_data["roc_auc"][i]:0.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {model_name}')
    plt.legend(loc="lower right", fontsize='small')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()