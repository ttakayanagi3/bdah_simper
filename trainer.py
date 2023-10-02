import tqdm
import torch
import torch.nn as nn

import preprocess as p
import models as m
import metrics as me
import trainer as t


def save_params(epoch, file_path, model):
    file_name = "model_{:04d}.pth".format(epoch)
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    file_path_ = os.path.join(file_path, file_name)
    torch.save(
        model.state_dict(),
        file_path_
    )


def train(data_loader, model, criterion, optimizer, opt):
    losses = []
    for epoch in tqdm.tqdm(range(opt.n_epochs)):
        running_loss = 0
        for frames, all_speed, y_angle in data_loader:
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
            all_labels = me.label_distance(all_speed1, all_speed2,
                                           opt.label_dist_fn, opt.label_temperature)
            batch_size = frames.shape[0]
            shape = frames.shape[2:]
            transform_shape = (batch_size * num_arguments, *shape)
            frames_transformed = frames.view(transform_shape)
            #
            # inference
            #
            model.zero_grad()
            all_z = model(frames_transformed, 'f')
            all_z = all_z.view(batch_size, num_arguments, -1)

            loss = 0
            for feats, labels in zip(all_z, all_labels):
                # print(feats, labels)
                feat1 = feats[:half_of_num_arguments]
                feat2 = feats[half_of_num_arguments:]
                if opt.feat_dist_fn == 'max_corr':
                    feat_dist = me.max_cross_corr(feat1, feat2)
                else:
                    pass
                loss += criterion(feat_dist, labels)
            loss /= batch_size
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss)
        save_params(epoch, 'params', model)
