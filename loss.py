import torch
import torch.nn.functional as F


def generalized_InfoNCE(feat_dist, labels):
    # feat_dist_numpy = feat_dist.detach().cpu().numpy()
    # label_numpy = labels.detach().cpu().numpy()

    labels_soft = F.softmax(labels, dim=0)
    feat_dist_soft = F.softmax(feat_dist, dim=0)

    # labels_numpy = labels_soft.detach().cpu().numpy()
    # feat_dist_numpy = feat_dist_soft.detach().cpu().numpy()

    weighted_cross_entropy = -labels_soft * torch.log(feat_dist_soft)
    InfoNCE_loss = torch.sum(weighted_cross_entropy)
    return InfoNCE_loss


def generalized_info_nce(feat_dist, labels, debug, temperature=0.1):
    feat_dist = feat_dist / temperature
    feat_dist_soft = F.softmax(feat_dist, dim=-1)

    weighted_cross_entropy = - labels * torch.log(feat_dist_soft)
    if debug == 1:
        labels_numpy = labels.detach().cpu().numpy()
        feat_dist_numpy = feat_dist.detach().cpu().numpy()
        feat_dist_soft_numpy = feat_dist_soft.detach().cpu().numpy()
        cross_entropy_numpy = weighted_cross_entropy.detach().cpu().numpy()
    InfoNCE_loss = torch.sum(weighted_cross_entropy)
    return InfoNCE_loss


    # sim = torch.matmul()

if __name__ == '__main__':
    feat_dist = torch.randn(5, 5)
    label = torch.randn(5, 5)
    label_temperature = 0.1
    label_soft = F.softmax(label / label_temperature, dim=-1)
    label_soft_numpy = label_soft.detach().cpu().numpy()
    loss = generalized_info_nce(feat_dist, label_soft)