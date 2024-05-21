import torch
import torch.nn.functional as F
from pytorch3d.transforms import Rotate, random_rotations
from pytorch3d.ops import knn_points


def index_points(points, idx) -> torch.Tensor:
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # [B, 1]/[B, 1, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # [1, S]/[1, S, K]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def get_graph_feature_three_dir(center_xyz, feat, k=20):
    # x: (B, N, 3), feat: (B, N, C)
    center_xyz = center_xyz.contiguous()
    batch_size, num_points = center_xyz.size()[:2]

    __, knn_idx, grouped_xyz = knn_points(center_xyz, center_xyz, K=k, return_nn=True) #  __ ,(B, N, K), (B, N, K, 3)
    
    # grouped_xyz = index_points(center_xyz, knn_idx)  # (B, N, K, 3)
    grouped_feat = index_points(feat, knn_idx)  # (B, N, K, C)

    dir_xy = feat[..., -6:].view(batch_size, num_points, 1, 2, 3)  # (B, N, 1, 2, 3)
    dir_z = dir_xy[:, :, :, 0].cross(dir_xy[:, :, :, 1])  # (B, N, 1, 3)
    dir_z = F.normalize(dir_z, dim=-1).unsqueeze(-2)  # (B, N, 1, 1, 3)
    dir_xyz = torch.cat((dir_xy, dir_z), dim=-2).repeat(1, 1, k, 1, 1)  # (B, N, K, 3, 3)

    grouped_dir_xy = grouped_feat[..., -6:].view(batch_size, num_points, k, 2, 3)  # (B, N, K, 2, 3)
    grouped_dir_z = grouped_dir_xy[:, :, :, 0].cross(grouped_dir_xy[:, :, :, 1])  # (B, N, K, 3)
    grouped_dir_z = F.normalize(grouped_dir_z, dim=-1).unsqueeze(-2)  # (B, N, K, 1, 3)
    grouped_dir_xyz = torch.cat((grouped_dir_xy, grouped_dir_z), dim=-2)  # (B, N, K, 3, 3)

    R_ji = (grouped_dir_xyz @ dir_xyz.transpose(-1, -2)).view(batch_size, num_points, k, 9)

    # normed grouped feature
    grouped_feat = grouped_feat[..., :128]  # (B, N, K, 128)
    feat = feat[..., :128]  # (B, N, 128)
    normed_group_feat = grouped_feat - feat.unsqueeze(-2)  # (B, N, K, 128)

    # relative point distance
    rel_xyz = grouped_xyz - center_xyz.unsqueeze(-2)  # (B, N, K, 3)
    rel_dist = torch.norm(rel_xyz, dim=-1, keepdim=True)  # (B, N, K, 1)
    rel_xyz = F.normalize(rel_xyz, dim=-1).unsqueeze(-1)  # (B, N, K, 3, 1)
    dot_pos = (dir_xyz @ rel_xyz).squeeze(-1)  # (B, N, K, 3)
    dot_pos2 = (grouped_dir_xyz @ rel_xyz).squeeze(-1)  # (B, N, K, 3)
    feature = torch.cat([grouped_feat, normed_group_feat, R_ji, dot_pos, dot_pos2, rel_dist], dim=-1)

    return feature.permute(0, 3, 1, 2).contiguous()

def get_orientation(dir_vectors):
    Ori_x = dir_vectors[:, 0]  # (BN, 3)
    Ori_y = dir_vectors[:, 1]  # (BN, 3)
    Ori_z = F.normalize(torch.cross(Ori_x, Ori_y), dim=-1).detach()
    Ori_xyz = torch.stack([Ori_x, Ori_y, Ori_z], dim=-2)  # (BN, 3, 3)
    return Ori_xyz

def get_random_rot(batch_size, device, train):
    if train:
        rot = random_rotations(batch_size)
    else:
        rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        rot = rot.view(1, 3, 3).repeat(batch_size, 1, 1)

    trot = Rotate(R=rot).to(device)
    return trot


def get_feat_loss(feat1, feat2):
    loss = F.mse_loss(feat1, feat2)
    return loss


def get_dir_loss(theta1, theta2, trot1, trot2):
    re_theta2 = trot2.inverse().transform_points(theta2)
    re_theta2 = trot1.transform_points(re_theta2)
    loss = F.mse_loss(theta1, re_theta2)
    return loss


def constraint_dir_loss(theta, trot, num_point):
    theta = trot.inverse().transform_points(theta)
    theta = theta.view(-1, num_point, 2, 3)
    theta_mean = F.normalize(theta.mean(dim=1, keepdim=True), dim=-1)
    loss = F.mse_loss(theta, theta_mean.repeat(1, num_point, 1, 1))
    return loss


def get_orth_loss(dir_vectors):
    Ori_x = dir_vectors[:, 0]  # (BN, 3)
    Ori_y = dir_vectors[:, 1]  # (BN, 3)
    dot_product = (Ori_x * Ori_y).sum(-1)  # (BN)
    zero_vector = torch.zeros(dot_product.size()[0], device=dot_product.device)
    loss = F.mse_loss(dot_product, zero_vector)
    return loss
