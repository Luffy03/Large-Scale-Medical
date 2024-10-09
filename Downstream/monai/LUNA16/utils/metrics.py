import threading
import torch
import numpy as np
import torch.nn.functional as F

class SegMetric_Numpy(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = 0.0
        self.SE = 0.0
        self.SP = 0.0
        self.PC = 0.0
        self.F1 = 0.0
        self.JS = 0.0
        self.DC = 0.0
        self.length = 0.0

        self.acc_i = 0.0
        self.SE_i = 0.0
        self.SP_i = 0.0
        self.PC_i = 0.0
        self.F1_i = 0.0
        self.JS_i = 0.0
        self.DC_i = 0.0

    def update(self, SR, GT, threshold=0.5):
        """
            SR:  Segmentation Result, a ndarray with the shape of [D, H, W]
            GT:  Ground Truth , a ndarray with the shape of [D, H, W]

        return:
            acc, SE, SP, PC, F1, js, dc
        """
        assert SR.shape == GT.shape

        SR = (SR > threshold).astype(int)
        GT = (GT == np.max(GT)).astype(int)
        corr = np.sum(SR == GT)
        #tensor_size = SR.shape[0] * SR.shape[1] * SR.shape[2]
        if len(SR.shape) == 2:
            tensor_size = SR.shape[0] * SR.shape[1]
        else:
            tensor_size =  SR.shape[0] * SR.shape[1] * SR.shape[2]
        self.acc_i = float(corr) / float(tensor_size)

        # TP : True Positive
        # FN : False Negative
        TP = ((SR == 1) & (GT == 1))
        FN = ((SR == 0) & (GT == 1))
        # TN : True negative
        # FP : False Positive
        TN = ((SR == 0) & (GT == 0))
        FP = ((SR == 1) & (GT == 0))

        self.SE_i = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)
        self.SP_i = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)
        self.PC_i = float(np.sum(TP)) / (float(np.sum(TP + FP)) + 1e-6)
        self.F1_i = 2 * self.SE_i * self.PC_i / (self.SE_i + self.PC_i + 1e-6)

        Inter = np.sum((SR + GT) == 2)
        Union = np.sum((SR + GT) >= 1)

        self.JS_i = float(Inter) / (float(Union) + 1e-6)
        self.DC_i = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)

        self.acc += self.acc_i
        self.SE += self.SE_i
        self.SP += self.SP_i
        self.PC += self.PC_i
        self.F1 += self.F1_i
        self.JS += self.JS_i
        self.DC += self.DC_i
        self.length += 1


    def update_in_FOV(self, SR, GT, FOV_Mask, threshold=0.5):
        """
            SR:  Segmentation Result, a ndarray with the shape of [D, H, W]
            GT:  Ground Truth , a ndarray with the shape of [D, H, W]

        return:
            acc, SE, SP, PC, F1, js, dc
        """
        assert SR.shape == GT.shape == FOV_Mask.shape
        FOV_Mask = (FOV_Mask==1).astype(int)
        SR = (SR > threshold).astype(int)*FOV_Mask
        GT = (GT == np.max(GT)).astype(int)*FOV_Mask
        corr = np.sum((SR == GT)&(FOV_Mask==1))
        tensor_size = np.sum(FOV_Mask)
        self.acc_i = float(corr) / float(tensor_size)

        # TP : True Positive
        # FN : False Negative
        TP = ((SR == 1) & (GT == 1))*FOV_Mask
        FN = ((SR == 0) & (GT == 1))*FOV_Mask
        # TN : True negative
        # FP : False Positive
        TN = ((SR == 0) & (GT == 0))*FOV_Mask
        FP = ((SR == 1) & (GT == 0))*FOV_Mask

        self.SE_i = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)
        self.SP_i = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)
        self.PC_i = float(np.sum(TP)) / (float(np.sum(TP + FP)) + 1e-6)
        self.F1_i = 2 * self.SE_i * self.PC_i / (self.SE_i + self.PC_i + 1e-6)

        Inter = np.sum((SR + GT) == 2)
        Union = np.sum((SR + GT) >= 1)

        self.JS_i = float(Inter) / (float(Union) + 1e-6)
        self.DC_i = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)

        self.acc += self.acc_i
        self.SE += self.SE_i
        self.SP += self.SP_i
        self.PC += self.PC_i
        self.F1 += self.F1_i
        self.JS += self.JS_i
        self.DC += self.DC_i
        self.length += 1

    @property
    def get_current(self):
        return self.acc_i, self.SE_i, self.SP_i, self.PC_i, self.F1_i, self.JS_i, self.DC_i

    @property
    def get_avg(self):
        return self.acc / self.length, self.SE / self.length, \
               self.SP / self.length, self.PC / self.length, self.F1 / self.length, \
               self.JS / self.length, self.DC / self.length


class SegMetric_Tensor(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = 0.0
        self.SE = 0.0
        self.SP = 0.0
        self.PC = 0.0
        self.F1 = 0.0
        self.JS = 0.0
        self.DC = 0.0
        self.length = 0.0

        self.acc_i = 0.0
        self.SE_i = 0.0
        self.SP_i = 0.0
        self.PC_i = 0.0
        self.F1_i = 0.0
        self.JS_i = 0.0
        self.DC_i = 0.0

    def update(self, SR, GT, threshold=0.5):
        """
        SR:  Segmentation Result, a tensor with the size of [D, H, W]
        GT:  Ground Truth , a tensor with the size of [D, H, W]

        return:
            acc, SE, SP, PC, F1, js, dc
        """
        assert SR.size() == GT.size()
        SR = (SR > threshold).int()
        GT = (GT == torch.max(GT)).int()
        corr = torch.sum(SR == GT)
        tensor_size = SR.numel()

        # Acc: accuracy
        self.acc_i = float(corr) / float(tensor_size)

        # TP : True Positive
        # FN : False Negative
        TP = ((SR == 1) & (GT == 1))
        FN = ((SR == 0) & (GT == 1))
        # TN : True negative
        # FP : False Positive
        TN = ((SR == 0) & (GT == 0))
        FP = ((SR == 1) & (GT == 0))

        self.SE_i = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
        self.SP_i = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
        self.PC_i = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
        self.F1_i = 2 * self.SE_i * self.PC_i / (self.SE_i + self.PC_i + 1e-6)

        Inter = torch.sum((SR + GT) == 2)
        ## torch.sum((SR + GT) == 2) == torch.sum(((SR + GT) == 2).int()) is True

        Union = torch.sum((SR + GT) >= 1)

        self.JS_i = float(Inter) / (float(Union) + 1e-6)
        self.DC_i = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

        self.acc += self.acc_i
        self.SE += self.SE_i
        self.SP += self.SP_i
        self.PC += self.PC_i
        self.F1 += self.F1_i
        self.JS += self.JS_i
        self.DC += self.DC_i
        self.length += 1

    @property
    def get_current(self):
        return self.acc_i, self.SE_i, self.SP_i, self.PC_i, self.F1_i, self.JS_i, self.DC_i

    @property
    def get_avg(self):
        return self.acc / self.length, self.SE / self.length, \
               self.SP / self.length, self.PC / self.length, self.F1 / self.length, \
               self.JS / self.length, self.DC / self.length




def iou(im1, im2):
    overlap = (im1 > 0.5) * (im2 > 0.5)
    union = (im1 > 0.5) + (im2 > 0.5)
    print('overlap', overlap.sum())
    print('union', float(union.sum()))
    return overlap.sum() / float(union.sum())


def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    print('ier sum', intersection.sum(), im_sum)

    return 2. * intersection.sum() / im_sum


class ClsEstimator():
    """
    Adpated from https://github.com/YijinHuang/pytorch-classification/
    """

    def __init__(self, criterion, num_classes, labels, thresholds=None):
        self.criterion = criterion
        self.num_classes = num_classes
        self.labels = labels
        self.thresholds = [-0.5 + i for i in range(num_classes)] if not thresholds else thresholds

        self.reset()  # intitialization

    def update(self, predictions, targets):
        targets = targets.data.cpu()
        predictions = predictions.data.cpu()
        predictions = self.to_prediction(predictions)

        # update metrics
        self.num_samples += len(predictions)
        self.correct += (predictions == targets).sum().item()
        for i, p in enumerate(predictions):
            self.conf_mat[int(targets[i])][int(p.item())] += 1

    def get_accuracy(self, digits=-1):
        acc = self.correct / self.num_samples
        acc = acc if digits == -1 else round(acc, digits)
        return acc

    def get_kappa(self, digits=-1):
        kappa = quadratic_weighted_kappa(self.conf_mat)
        kappa = kappa if digits == -1 else round(kappa, digits)
        return kappa

    def reset(self):
        self.correct = 0
        self.num_samples = 0
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def to_prediction(self, predictions):
        if self.criterion in ['ce', 'focal_loss', 'kappa']:
            predictions = torch.tensor(
                [torch.argmax(p) for p in predictions]
            ).long()
        elif self.criterion in ['mse', 'mae', 'smooth_L1']:
            predictions = torch.tensor(
                [self.classify(p.item()) for p in predictions]
            ).float()
        else:
            raise NotImplementedError('Not implemented criterion.')

        return predictions

    def classify(self, predict):
        thresholds = self.thresholds
        predict = max(predict, thresholds[0])
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                return i

    def summary(self): 
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.conf_mat)
        for i in range(self.num_classes):
            sum_TP += self.conf_mat[i, i]  
        acc = sum_TP / n 
        print("The model accuracy is ", acc)
        
        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.conf_mat[0])):
            sum_po += self.conf_mat[i][i]
            row = np.sum(self.conf_mat[i, :])
            col = np.sum(self.conf_mat[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        kappa = round((po - pe) / (1 - pe), 3)
        print("The model kappa is: ", kappa)
        print("The confusion matrix is: ", self.conf_mat)

        # precision, recall, specificity
        table = PrettyTable() 
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  
            TP = self.conf_mat[i, i]
            FP = np.sum(self.conf_mat[:, i]) - TP
            FN = np.sum(self.conf_mat[i, :]) - TP
            TN = np.sum(self.conf_mat) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        return str(acc)

    def plot(self, path): 
        matrix = self.conf_mat
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig(path + '/confusion_matrix.png')


def quadratic_weighted_kappa(conf_mat):
    print('confusion matrix', conf_mat)
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j) ** 2) / ((cate_num - 1) ** 2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()

    return (observed - expected) / (1 - expected)


