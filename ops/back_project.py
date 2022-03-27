import torch
from torch.nn.functional import grid_sample


def back_project(coords, origin, voxel_size, feats, KRcam):
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (num of voxels,)
    '''
    n_views, bs, c, h, w = feats.shape

    feature_volume_all = torch.zeros(coords.shape[0], c + 1).cuda()  # [Nv, C+1]
    count = torch.zeros(coords.shape[0]).cuda()  # [Nv]

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1)
        coords_batch = coords[batch_ind][:, 1:]

        coords_batch = coords_batch.view(-1, 3)  # [N_b, 3]
        origin_batch = origin[batch].unsqueeze(0)  # [1, 3]
        feats_batch = feats[:, batch]  # [9, C, H, W]
        proj_batch = KRcam[:, batch]  # [9, 4, 4]

        grid_batch = coords_batch * voxel_size + origin_batch.float()
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)  # [9, N_b, 3]
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()  # [9, 3, N_b]
        nV = rs_grid.shape[-1]  # number of voxels
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).cuda()], dim=1)  # [9, 4, N_b]

        # Project grid
        im_p = proj_batch @ rs_grid  # [9, 4, N_b]
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]  # [9, N_b]
        im_x = im_x / im_z
        im_y = im_y / im_z

        # convert to [-1, 1] for grid_sampling
        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)  # [9, N_b, 2]
        mask = im_grid.abs() <= 1  # within [-1, 1] range
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)  # [9, N_b]

        feats_batch = feats_batch.view(n_views, c, h, w)  # [9, C, H, W]
        im_grid = im_grid.view(n_views, 1, -1, 2)  # [9, 1, N_b, 2]
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)  # [9, C, 1, N_b]

        features = features.view(n_views, c, -1)  # [9, C, N_b]
        mask = mask.view(n_views, -1)  # [9, N_b]
        im_z = im_z.view(n_views, -1)  # [9, N_b]
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        count[batch_ind] = mask.sum(dim=0).float()  # [N_b]

        # aggregate multi view
        features = features.sum(dim=0)  # [C, N_b]
        mask = mask.sum(dim=0)  # [N_b]
        invalid_mask = mask == 0
        mask[invalid_mask] = 1
        in_scope_mask = mask.unsqueeze(0)  # [1, N_b]
        features /= in_scope_mask  # average
        features = features.permute(1, 0).contiguous()  # [N_b, C]

        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()  # [1, N_b], averaged observed depth value at each voxel
        im_z_mean = im_z[im_z > 0].mean()
        im_z_std = torch.norm(im_z[im_z > 0] - im_z_mean) + 1e-5
        im_z_norm = (im_z - im_z_mean) / im_z_std
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=1)  # [N_b, C + 1]

        feature_volume_all[batch_ind] = features
    return feature_volume_all, count
