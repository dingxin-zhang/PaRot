import torch
import torch.nn as nn
import torch.nn.functional as F

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()


def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

def three_nn(xyz1, xyz2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)

    dists, inds = dists[:, :, :3], inds[:, :, :3]
    return dists, inds



def get_k_nn(xyz1, xyz2, k):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)

    dists, inds = dists[:, :, :k], inds[:, :, :k]
    return dists, inds


def interpolate_angle(xyz1, xyz2, feature, k):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)   N1>N2
    :param feature: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    _, _, C2 = feature.shape
    dists, inds = get_k_nn(xyz1, xyz2, k)

    inversed_dists = 1.0 / (dists + 1e-8)

    weight = inversed_dists / torch.sum(inversed_dists, dim=-1, keepdim=True) # shape=(B, N1, 3)

    weight = torch.unsqueeze(weight, -1)

    interpolated_feature = gather_points(feature, inds)  # shape=(B, N1, 3, C2)

    return interpolated_feature, inds, weight



class FP_Module_root(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(FP_Module_root, self).__init__()
        self.posembed = nn.Sequential(
            nn.Conv1d(9 + 6 + 1, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.backbone = nn.Sequential()
        bias = False if bn else True
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv1d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm1d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz1, xyz2, feat1, feat2):
        '''

        :param xyz1: shape=(B, N1, 3)
        :param xyz2: shape=(B, N2, 3)   (N1 >= N2)
        :param points1: shape=(B, N1, C1)
        :param points2: shape=(B, N2, C2)
        :return: new_points2: shape = (B, N1, mlp[-1])
        '''

        interpolated_points = feat2.repeat(1, feat1.size(1), 1)

        if feat1 is not None:
            cat_interpolated_points = torch.cat([interpolated_points, feat1], dim=-1).permute(0, 2, 1).contiguous()
        else:
            cat_interpolated_points = interpolated_points.permute(0, 2, 1).contiguous()

        pos = self.posembed(cat_interpolated_points[:,1280:])
        new_points = self.backbone(torch.cat((cat_interpolated_points[:,:1280], pos), dim=1))
        return new_points.permute(0, 2, 1).contiguous()

class FP_Module_root_combine(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(FP_Module_root_combine, self).__init__()
        self.backbone = nn.Sequential()
        bias = False if bn else True
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv1d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm1d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz1, xyz2, feat1, feat2):
        '''

        :param xyz1: shape=(B, N1, 3)
        :param xyz2: shape=(B, N2, 3)   (N1 >= N2)
        :param points1: shape=(B, N1, C1)
        :param points2: shape=(B, N2, C2)
        :return: new_points2: shape = (B, N1, mlp[-1])
        '''

        interpolated_points = feat2.repeat(1, feat1.size(1), 1)

        if feat1 is not None:
            cat_interpolated_points = torch.cat([interpolated_points, feat1], dim=-1).permute(0, 2, 1).contiguous()
        else:
            cat_interpolated_points = interpolated_points.permute(0, 2, 1).contiguous()

        new_points = self.backbone(cat_interpolated_points)
        return new_points.permute(0, 2, 1).contiguous()


class FP_Module_angle_label(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(FP_Module_angle_label, self).__init__()
        self.posembed = nn.Sequential(
            nn.Conv2d(6 + 1, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.backbone = nn.Sequential()
        bias = False if bn else True
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv2d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz1, xyz2, points1, feat2, dir1, dir2, label, k=3):
        '''
        :param xyz1: shape=(B, N1, 3)
        :param xyz2: shape=(B, N2, 3)   (N1 >= N2)
        :param points1: shape=(B, N1, C1)
        :param points2: shape=(B, N2, C2)
        :return: new_points2: shape = (B, N1, mlp[-1])
        '''

        B, N1, C1 = points1.shape
        _, N2, C2 = feat2.shape

        interpolated_feature, inds, inversed_dists = interpolate_angle(xyz1, xyz2, feat2, k)

        dir1 = dir1.view(-1, N2, 9)
        dir2 = dir2.view(-1, N2, 9)
        close_lrf = gather_points(xyz2, inds)

        dir1 = gather_points(dir1, inds).view(-1, 3, 3)
        dir2 = gather_points(dir2, inds).view(-1, 3, 3)

        relate_position = points1.unsqueeze(2).repeat(1, 1, k, 1)- close_lrf

        for_dot = F.normalize(relate_position.view(-1, 3), dim=-1).unsqueeze(2)
        angle1 = dir1.matmul(for_dot)
        angle2 = dir2.matmul(for_dot)

        angle1 = angle1.view(B, N1, k, -1)
        angle2 = angle2.view(B, N1, k, -1)

        label = label.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, k, N1)

        relative_pos = torch.cat((torch.norm(relate_position, dim=-1, keepdim=True), angle1, angle2), dim=3)

        pos = self.posembed(relative_pos.permute(0, 3, 2, 1))

        interpolated_feature = interpolated_feature.permute(0, 3, 2, 1)

        cat_interpolated_points = torch.cat((interpolated_feature, pos, label), dim=1)

        new_points = self.backbone(cat_interpolated_points)

        new_points = torch.sum(new_points, dim=2)

        return new_points.permute(0, 2, 1).contiguous()




class FP_Module_root_v2(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(FP_Module_root_v2, self).__init__()
        self.backbone = nn.Sequential()
        bias = False if bn else True
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv2d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz1, xyz2, feat1, feat2, label):
        '''
        :param xyz1: shape=(B, N1, 3)
        :param xyz2: shape=(B, N2, 3)   (N1 >= N2)
        :param points1: shape=(B, N1, C1)
        :param points2: shape=(B, N2, C2)
        :return: new_points2: shape = (B, N1, mlp[-1])
        '''

        label = label.unsqueeze(1).repeat(1, feat1.size(1), 1)

        interpolated_points = feat2.repeat(1, feat1.size(1), 1)

        if feat1 is not None:
            cat_interpolated_points = torch.cat([interpolated_points, feat1, label], dim=-1).permute(0, 2, 1).contiguous()
        else:
            cat_interpolated_points = interpolated_points.permute(0, 2, 1).contiguous()

        new_points = torch.squeeze(self.backbone(torch.unsqueeze(cat_interpolated_points, -1)), dim=-1)
        return new_points.permute(0, 2, 1).contiguous()


class FP_Module_angle_label_v2(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(FP_Module_angle_label_v2, self).__init__()
        self.backbone = nn.Sequential()
        bias = False if bn else True
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv2d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz1, xyz2, points1, feat2, dir1, dir2, k=3):
        '''
        :param xyz1: shape=(B, N1, 3)
        :param xyz2: shape=(B, N2, 3)   (N1 >= N2)
        :param points1: shape=(B, N1, C1)
        :param points2: shape=(B, N2, C2)
        :return: new_points2: shape = (B, N1, mlp[-1])
        '''

        B, N1, C1 = points1.shape
        _, N2, C2 = feat2.shape

        interpolated_feature, inds, inversed_dists = interpolate_angle(xyz1, xyz2, feat2, k)

        dir1 = dir1.view(-1, N2, 6)
        dir2 = dir2.view(-1, N2, 6)
        close_lrf = gather_points(xyz2, inds)

        dir1 = gather_points(dir1, inds).view(-1, 2, 3)
        dir2 = gather_points(dir2, inds).view(-1, 2, 3)

        relate_position = points1.unsqueeze(2).repeat(1, 1, k, 1)- close_lrf

        for_dot = F.normalize(relate_position.view(-1, 3), dim=-1).unsqueeze(2)
        angle1 = dir1.matmul(for_dot)
        angle2 = dir2.matmul(for_dot)

        angle1 = angle1.view(B, N1, k, -1)
        angle2 = angle1.view(B, N1, k, -1)


        comb_relate = torch.cat((interpolated_feature, torch.sum(torch.pow(relate_position, 2), dim=-1, keepdim=True), angle1, angle2), dim = 3)

        cat_interpolated_points = comb_relate.permute(0, 3, 2, 1)#.contiguous()

        new_points = self.backbone(cat_interpolated_points)

        new_points = torch.sum(new_points, dim=2)

        return new_points.permute(0, 2, 1).contiguous()
