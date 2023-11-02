import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import preprocess as p
import models as m
import metrics as me
import loss as l
import trainer as t
import mlflow


def save_params(epoch, file_path, model):
    file_name = "model_{:04d}.pth".format(epoch)
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    file_path_ = os.path.join(file_path, file_name)
    torch.save(
        model.state_dict(),
        file_path_
    )


def train(data_loader, model, criterion, optimizer, device, opt):
    losses = []
    mlflow.set_experiment(experiment_name=opt.experiment_name)
    mlflow.set_tracking_uri('./mlruns')
    with mlflow.start_run():
        for epoch in range(opt.n_epochs):
            running_loss = 0
            iter_size = len(data_loader)
            for frames, all_speed, y_angle in tqdm.tqdm(data_loader):
                #
                # split all speed
                # 2 * M -> M, M
                #
                num_arguments = frames.shape[1]
                half_of_num_arguments = int(num_arguments // 2)
                all_speed1 = all_speed[:, :half_of_num_arguments]
                all_speed2 = all_speed[:, half_of_num_arguments:]
                #
                # label distance
                #
                all_labels, label_index = me.label_distance(all_speed1, all_speed2,
                                               opt.label_dist_fn, opt.label_temperature)
                batch_size = frames.shape[0]
                shape = frames.shape[2:]
                transform_shape = (batch_size * num_arguments, *shape)
                frames_transformed = frames.view(transform_shape)
                #
                # inference
                #
                model.zero_grad()
                frames_transformed = frames_transformed.to(device)
                all_z = model(frames_transformed, 'f')
                all_z = all_z.view(batch_size, num_arguments, -1)

                loss = 0
                true_label_list = []
                feat_dist_list = []
                # for feats, labels in zip(all_z, label_index):
                for feats, labels in zip(all_z, all_labels):
                    labels = labels.to(device)
                    # print(feats, labels)
                    feat1 = feats[:half_of_num_arguments]
                    feat2 = feats[half_of_num_arguments:]
                    if opt.feat_dist_fn == 'max_corr':
                        feat_dist = me.batched_max_cross_corr(feat1, feat2, device)
                        # feat_dist = me.max_cross_corr(feat1, feat2, device)
                    else:
                        feat_dist = 9999
                    # gen_infoNCE_loss = l.generalized_InfoNCE(feat_dist, labels)
                    gen_infoNCE_loss = l.generalized_info_nce(feat_dist, labels, 1)
                    loss += gen_infoNCE_loss

                    # labels_soft = F.softmax(labels, dim=0)
                    # feat_dist_soft = F.softmax(feat_dist, dim=0)
                    #
                    # _cross_entropy = -labels_soft * torch.log(feat_dist_soft)
                    # cross_entropy = torch.sum(_cross_entropy)
                    # loss += cross_entropy

                    # labels_numpy = labels.detach().cpu().numpy()
                    # feat_dist_numpy = feat_dist.detach().cpu().numpy()
                    # _cross_entropy_numpy = -labels_numpy * np.log(feat_dist_numpy)
                    # cross_entropy_numpy = np.sum(_cross_entropy_numpy)
                    #
                    # feat_dist_list.append(feat_dist)
                    # true_label_list.append(labels)
                # feat_tensor = torch.stack(feat_dist_list)
                # label_tensor = torch.stack(true_label_list)
                # label_tensor = label_tensor.float()
                #
                # feat_tensor = feat_tensor.to(device)
                # label_tensor = label_tensor.to(device)
                #
                # loss += criterion(feat_tensor, label_tensor)
                loss /= batch_size
                loss /= half_of_num_arguments
                loss.backward()
                optimizer.step()
                mlflow.log_metric(key='loss', value=loss.item(), step=1)
                running_loss += loss.item() / iter_size
                # print('loss', loss.item())
                # save_params(epoch, 'params', model)
            losses.append(running_loss)
            mlflow.log_metric(key='running loss', value=running_loss, step=1)
            print(f'running loss: {running_loss}')
            save_params(epoch, 'params', model)


def plot_umap(model, data_loader, device, opt):
    import umap
    z_arr = []
    speed_arr = []
    for frames, all_speed, y_angle in tqdm.tqdm(data_loader):
        #
        # split all speed
        # 2 * M -> M, M
        #
        num_arguments = frames.shape[1]
        half_of_num_arguments = int(num_arguments // 2)
        all_speed1 = all_speed[:, :half_of_num_arguments]
        all_speed2 = all_speed[:, half_of_num_arguments:]
        #
        # label distance
        #
        all_labels, label_index = me.label_distance(all_speed1, all_speed2,
                                                    opt.label_dist_fn, opt.label_temperature)
        batch_size = frames.shape[0]
        shape = frames.shape[2:]
        transform_shape = (batch_size * num_arguments, *shape)
        frames_transformed = frames.view(transform_shape)
        #
        # inference
        #
        mini_batch_size = frames.shape[0]
        frames_transformed = frames_transformed.to(device)
        all_z = model(frames_transformed, 'f')
        all_z = all_z.view(batch_size, num_arguments, -1)
        all_speed_transformed = all_speed.view((mini_batch_size * num_arguments, -1))
        speed_arr_ = all_speed_transformed.detach().cpu().numpy().flatten()
        data = all_z.detach().cpu().numpy()
        z_arr.append(data.reshape(-1, opt.extract_time_frames))
        speed_arr.append(speed_arr_)

    z_arr = np.vstack(z_arr)
    speed_arr = np.concatenate(speed_arr)

    embedding = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(z_arr)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=speed_arr, cmap='Blues', s=5)
    plt.colorbar()
    plt.show()
