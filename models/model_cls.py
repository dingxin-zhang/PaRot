import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from util import cal_loss
from models.model_util import get_graph_feature_three_dir, get_random_rot, index_points, get_dir_loss, \
    get_orth_loss, get_orientation 
from pytorch3d.ops import sample_farthest_points as fps
from pytorch3d.ops import knn_points

class Classifier(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Classifier, self).__init__()
        self.posembed = nn.Sequential(
            nn.Conv1d(9 + 6 + 1, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.inter_layer = nn.Sequential(
            nn.Conv1d(256 + 32, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, args.emb_dims, 1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.cls = nn.Sequential(
            nn.Linear(args.emb_dims, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, output_channels)
        )

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()

        #restore pose information
        pos = self.posembed(x[:,256:])

        x = self.inter_layer(torch.cat((x[:,:256], pos), dim=1)).max(dim=-1)[0]
        x = self.cls(x)
        return x


class DisentangleNet(nn.Module):
    def __init__(self, args, dim=128):
        super(DisentangleNet, self).__init__()
        self.pn_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(True),
        )

        if args.invar_block == 2:
            self.inv_encoder = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(True),
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(True)
            )
        elif args.invar_block == 3:
            self.inv_encoder = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(True),
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(True),
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(True)
            )

        self.equi_encoder1 = nn.Sequential(
            nn.Linear(dim, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 3, bias=False)
        )

        self.equi_encoder2 = nn.Sequential(
            nn.Linear(dim, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 3, bias=False)
        )

    def forward(self, x):
        """
        :param x: input feature: (BN, 3, K)
        :return: x_inv: (BN, C_dim); dir_vec: (BN, 2, 3)
        """
        x_inter = self.pn_encoder(x).max(dim=-1)[0]  # [BN, C_dim]

        # invariance disentanglement
        x_inv = self.inv_encoder(x_inter)

        # equivariance disentanglement
        dir_vec1 = self.equi_encoder1(x_inter)
        dir_vec1 = F.normalize(dir_vec1, dim=-1)  # (BN, 3)

        dir_vec2 = self.equi_encoder2(x_inter)
        dir_vec2 = F.normalize(dir_vec2, dim=-1)  # (BN, 3)

        dir_vec = torch.stack([dir_vec1, dir_vec2], dim=1)  # (BN, 2, 3)
        return x_inv, dir_vec


class IntraLayer(nn.Module):
    def __init__(self, C_in, C_inter=128, C_out=128):
        super(IntraLayer, self).__init__()
        self.pos = nn.Conv2d(9 + 6 + 1, 32, 1, bias=False)
        self.bnpos = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(C_in+32, C_inter, 1, bias=False)
        self.conv2 = nn.Conv2d(C_inter, C_out, 1, bias=False)
        self.conv3 = nn.Conv2d(C_out, C_out, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(C_inter)
        self.bn2 = nn.BatchNorm2d(C_out)
        self.bn3 = nn.BatchNorm2d(C_out)

    def forward(self, x):
        #restore pose information
        pos = F.relu(self.bnpos(self.pos(x[:,256:])))

        x = F.relu(self.bn1(self.conv1(torch.cat((x[:,:256], pos), dim=1))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=-1)[0]  # (B, C_out, N)
        return x


class LocalEncoder(nn.Module):
    def __init__(self, args, C_dim=128):
        super(LocalEncoder, self).__init__()
        self.k = args.k_local
        self.k_local_layer = args.k_local_layer
        self.S = args.local_S
        self.num_points = args.num_points
        self.use_ball_query = args.use_ball_query

        # Siamese Training Procedure
        self.disNet = DisentangleNet(args, dim=C_dim)
        self.intra_layer = IntraLayer(C_in=256, C_inter=128, C_out=128)

    def forward(self, xyz, train=True):
        xyz = xyz.contiguous()
        if self.num_points != 1024:
            xyz, fps_idx = fps(xyz, K=self.num_points, random_start_point=True)      # (B, 1024, 3)

        new_xyz, fps_idx = fps(xyz, K=self.S, random_start_point=True)       # (B, 128, 3)

        if self.use_ball_query:
            idx = ball_query(xyz, new_xyz, radius, self.k)
            x = index_points(xyz, idx)
            x -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, self.k, 1)
        else:
            dist, knn_idx, grouped_xyz = knn_points(new_xyz, xyz, K=self.k, return_nn=True)  # (B, S, K, 3)
            x = grouped_xyz - new_xyz.unsqueeze(2)

        M_xyz = x.reshape(-1, self.k, 3)  # (B * S, K, 3)
        trot1 = get_random_rot(M_xyz.size()[0], xyz.device, train)
        trot2 = get_random_rot(M_xyz.size()[0], xyz.device, train)
        rot_data1 = trot1.transform_points(M_xyz) if train else M_xyz
        rot_data2 = trot2.transform_points(M_xyz) if train else M_xyz

        # shared invariance/equivariance
        x1, dir1 = self.disNet(rot_data1.transpose(1, 2))  # (BN, C_dim), (BN, 2, 3)
        x2, dir2 = self.disNet(rot_data2.transpose(1, 2))  # (BN, C_dim), (BN, 2, 3)

        # get the original pose
        dir1_origin = trot1.inverse().transform_points(dir1) if train else dir1
        dir2_origin = trot2.inverse().transform_points(dir2) if train else dir2

        feature = torch.cat([x1, dir1_origin.view(-1, 6)], dim=1).view(-1, self.S, 128 + 6)
        feature = get_graph_feature_three_dir(new_xyz, feature, k=self.k_local_layer)
        feature = self.intra_layer(feature)  # (B, C, N)

        return feature, new_xyz, (x1, x2), (trot1, trot2), (dir1, dir2), (dir1_origin, dir2_origin)


class GlobalEncoder(nn.Module):
    def __init__(self, args, C_dim=128):
        super(GlobalEncoder, self).__init__()
        self.k = args.k_global
        self.S = args.local_S
        self.num_points = args.num_points
        self.use_ball_query = args.use_ball_query

        # Siamese Training Procedure
        self.disNet = DisentangleNet(args, dim=C_dim)


    def forward(self, xyz, local_xyz, train=True):
        local_xyz = local_xyz.contiguous()
        B, N, _ = local_xyz.size()

        if self.S == self.k:
            grouped_xyz = local_xyz.unsqueeze(1).repeat(1, self.S, 1, 1)
        else:
            global_xyz, fps_idx = fps(xyz, K=self.k, random_start_point=True)    # (B, 1024, 3)
            grouped_xyz = global_xyz.unsqueeze(1).repeat(1, self.S, 1, 1)

        grouped_xyz = grouped_xyz - local_xyz.unsqueeze(2)

        M_xyz = grouped_xyz.view(-1, self.k, 3)  # (B * N, K, 3)
        batch_size = M_xyz.size()[0]  # B * N


        trot1 = get_random_rot(batch_size, xyz.device, train)
        trot2 = get_random_rot(batch_size, xyz.device, train)

        if train:
            rot_data1 = trot1.transform_points(M_xyz)
            rot_data2 = trot2.transform_points(M_xyz)
        else:
            rot_data1 = M_xyz
            rot_data2 = M_xyz

        # shared invariance/equivariance
        x1, dir1 = self.disNet(rot_data1.transpose(1, 2))  # (BN, C_dim), (BN, 2, 3)
        x2, dir2 = self.disNet(rot_data2.transpose(1, 2))  # (BN, C_dim), (BN, 2, 3)

        # get the original pose
        dir1_origin = trot1.inverse().transform_points(dir1) if train else dir1
        dir2_origin = trot2.inverse().transform_points(dir2) if train else dir2

        # get the updated global feature
        feature = x1.view(B, N, -1).transpose(1, 2).contiguous()

        return feature, (x1, x2), (trot1, trot2), (dir1, dir2), (dir1_origin, dir2_origin)


class Model(nn.Module):
    def __init__(self, args, C_dim=128, output_channels=40):
        super(Model, self).__init__()
        self.local_enc = LocalEncoder(args, C_dim)
        self.global_enc = GlobalEncoder(args, C_dim)
        self.classifier = Classifier(args, output_channels)
        self.criterion_cls = cal_loss
        self.device = args.device

        self.loss_dir_l = args.loss_dir_l
        self.loss_dir_g = args.loss_dir_g

        self.loss_orth_l = args.loss_orth_l
        self.loss_orth_g = args.loss_orth_g

        self.loss_feat_l = args.loss_feat_l
        self.loss_feat_g = args.loss_feat_g

        self.loss_cls = args.loss_cls


    def forward(self, xyz, label, train=True):
        local_feat, new_xyz, (feat1_l, feat2_l), (trot1_l, trot2_l), (dir1_l, dir2_l), (dir1_origin_l, dir2_origin_l) = self.local_enc(xyz, train)
        global_feat, (feat1_g, feat2_g), (trot1_g, trot2_g), (dir1_g, dir2_g), (dir1_origin_g, dir2_origin_g) = self.global_enc(xyz, new_xyz, train)


        fused_feat = torch.cat([local_feat, global_feat], dim=1)  # (B, 128 + 128, N)

        B, N, _ = new_xyz.size()

        ori1_l = get_orientation(dir1_origin_l) # (BN, 3, 3)
        ori2_l = get_orientation(dir2_origin_l)
        ori1_g = get_orientation(dir1_origin_g)
        ori2_g = get_orientation(dir2_origin_g)

        R_l2g = ori1_l @ ori1_g.transpose(1, 2)
        R_l2g_prime = ori2_l @ ori2_g.transpose(-1, -2)

        dist = torch.norm(new_xyz, dim=-1, keepdim=True)
        new_xyz = F.normalize(new_xyz, dim=-1)

        T_l = (ori1_l @ new_xyz.view(-1, 3, 1)).view(B, N, 3)
        T_g = (ori1_g @ new_xyz.view(-1, 3, 1)).view(B, N, 3)

        # fuse all features
        feat = torch.cat([fused_feat.transpose(-1, -2).contiguous(),
                       dist,
                       R_l2g.view(B, N, -1), T_l, T_g], dim=-1)

        logits = self.classifier(feat)

        loss_cls = self.criterion_cls(logits, label)

        # version change
        loss_dir_local = get_dir_loss(dir1_l, dir2_l, trot1_l, trot2_l)
        loss_dir_global = get_dir_loss(dir1_g, dir2_g, trot1_g, trot2_g)
        loss_feat_local = F.mse_loss(feat1_l, feat2_l)
        loss_feat_global = F.mse_loss(feat1_g, feat2_g)
        loss_orth_local = get_orth_loss(dir1_l)
        loss_orth_global = get_orth_loss(dir1_g)
        loss_list = [loss_cls, loss_dir_local, loss_dir_global, loss_orth_local, loss_orth_global, loss_feat_local, loss_feat_global]

        weighted_loss = torch.Tensor([0]).to(self.device)
        if self.loss_dir_l != 0:
            weighted_loss += loss_dir_local * self.loss_dir_l

        if self.loss_dir_g != 0:
            weighted_loss += loss_dir_global * self.loss_dir_g


        if self.loss_orth_l != 0:
            weighted_loss += loss_orth_local * self.loss_orth_l

        if self.loss_orth_g != 0:
            weighted_loss += loss_orth_global * self.loss_orth_g


        if self.loss_feat_l != 0:
            weighted_loss += loss_feat_local * self.loss_feat_l

        if self.loss_feat_g != 0:
            weighted_loss += loss_feat_global * self.loss_feat_g


        if self.loss_cls != 0:
            weighted_loss += loss_cls * self.loss_cls


        return weighted_loss, logits, loss_list
