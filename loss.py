import torch
import torch.nn.functional as F


def generalized_InfoNCE(feat_dist, labels):

    labels_soft = F.softmax(labels, dim=0)
    feat_dist_soft = F.softmax(feat_dist, dim=0)
    # labels_numpy = labels_soft.detach().cpu().numpy()
    # feat_dist_numpy = feat_dist_soft.detach().cpu().numpy()

    weighted_cross_entropy = -labels_soft * torch.log(feat_dist_soft)
    InfoNCE_loss = torch.sum(weighted_cross_entropy)
    return InfoNCE_loss


if __name__ == '__main__':
    pass
