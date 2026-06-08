import torch
import numpy as np

def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        (n_class * label_true[mask].astype(int) +
        label_pred[mask]), minlength=n_class ** 2).reshape(n_class, n_class).reshape(n_class, n_class)
    return hist

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.epsilon = np.finfo(np.float32).eps

    def pixel_accuracy(self, hist):
        pa=np.diag(hist).sum() / hist.sum()
        return pa

    def kappa(self, hist):
        total_sum = hist.sum()
        oa=np.diag(hist).sum() / total_sum
        pe = ((hist.sum(axis=0)[0] * hist.sum(axis=1)[0]) + (hist.sum(axis=0)[1] * hist.sum(axis=1)[1])) / (total_sum**2)
        kappa = (oa - pe) / (1 - pe)
        return kappa

    def mean_pixel_accuracy(self, hist):
        cpa = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        mpa = np.nanmean(cpa)
        return mpa

    def precision(self, hist):
        precision = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        precision = np.nanmean(precision)
        return precision

    def recall(self, hist):
        recall = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + self.epsilon)
        recall = np.nanmean(recall)
        return recall

    def f1_score(self, hist):
        f1 = (np.diag(hist) + self.epsilon) * 2 / (hist.sum(axis=1) * 2 + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        f1 = np.nanmean(f1)
        return f1

    def mean_intersection_over_union(self, hist):
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        miou = np.nanmean(iou)
        return miou

    def frequency_weighted_intersection_over_union(self, hist):
        freq = hist.sum(axis=1) / hist.sum()
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou

    def cpa(self, hist):
        cpa = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        return cpa
    
    def compute_iou(self, confusion_matrix, num_classes, ignore_class=0):
        iou = []
        for i in range(num_classes):
            if i == ignore_class:
                continue

            intersection = confusion_matrix[i, i].item()
            union = confusion_matrix[i, :].sum().item() + confusion_matrix[:, i].sum().item() - intersection
            if union == 0:
                iou.append(0)
            else:
                iou.append(intersection / union)
        
        return np.mean(iou)

    def compute_kappa(self, confusion_matrix):
        total = confusion_matrix.sum().item()
        diag_sum = confusion_matrix.diag().sum().item()
        oa = diag_sum / total
    
        row_sum = confusion_matrix.sum(dim=1).float()
        col_sum = confusion_matrix.sum(dim=0).float()
        ea = (row_sum @ col_sum) / (total ** 2)

        kappa = (oa - ea) / (1 - ea) if (1 - ea) != 0 else 0.0
        return kappa

    def compute_kappa_multiclass(self, preds, targets, ignore_class=0):
        preds = preds.astype(int)
        targets = targets.astype(int)

        num_classes = max(preds.max().item(), targets.max().item()) + 1
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

        for t, p in zip(targets, preds):
            confusion_matrix[t, p] += 1

        if ignore_class < num_classes:
            confusion_matrix = np.delete(confusion_matrix.numpy(), ignore_class, axis=0)
            confusion_matrix = np.delete(confusion_matrix, ignore_class, axis=1)
            confusion_matrix = torch.tensor(confusion_matrix, dtype=torch.int64)

        kappa = self.compute_kappa(confusion_matrix)
        iou = self.compute_iou(confusion_matrix, confusion_matrix.size(0), ignore_class)
        sek = kappa * (np.exp(iou) - 1)

        return sek


