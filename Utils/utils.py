import argparse

import torch
import torch.nn.functional as fn
import torch.distributions as dist
class hyperpm(object):
    routit = 4
    dropout = 0.05
    lr = 0.05
    reg = 0.08
    nlayer = 1
    ncaps = 4
    nhidden = 64 // ncaps
    latent_nnb_k = 4
    gm_update_rate = 0.35
    space_lambda = 0.88
    div_lambda = 0.033


def LGD_loss(output, args):
    h__gm_reg_loss = 0.0
    h__div_reg_loss = 0.0
    disen_y = torch.arange(args.ncaps).long().unsqueeze(dim=0).repeat(args.num_stations, 1).flatten().to(args.device)
    for i in range(len(output)):
        for j in range(len(output[i])):
            h_pred_feat_list = [output[i][j]]
            h__means_list, h__covmats_list = [], []
            for idx, h_pred_feat in enumerate(h_pred_feat_list):
                tp__means, tp__covmats = initialize_mean_cov(args, h_pred_feat)
                h__means_list.append(tp__means)
                h__covmats_list.append(tp__covmats)


            for idx, (h_pred_feat, h__means, h__covmats) in enumerate(
                    zip(h_pred_feat_list[::-1], h__means_list[::-1], h__covmats_list[::-1])):
                # lik-reg-loss
                h__gm_reg_loss += (10 ** -idx) * compute_gm_reg_loss(
                    x=h_pred_feat.view(-1, args.nhidden), y=disen_y, means=h__means, cov_mats=h__covmats)
                # div-reg-loss
                h__div_reg_loss += (10 ** -idx) * compute_div_loss(args,
                                                                   disen_likeli=compute_gm_likeli_(args,
                                                                                                   disen_x=h_pred_feat.view(-1,
                                                                                                                            args.nhidden),
                                                                                                   means=h__means,
                                                                                                   inv_covmats=h__covmats))
    return h__gm_reg_loss+h__div_reg_loss

def haversine_distance(coord1, coord2):
    # 将经纬度转换为弧度
    lat1, lon1 = torch.deg2rad(coord1[:, :2]).t()
    lat2, lon2 = torch.deg2rad(coord2).t()

    # Haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(a))

    # 地球半径（单位：千米）
    radius_earth = 6371.0

    # 计算距离
    distance = radius_earth * c

    return distance
def radians(degrees):
    """
    Convert degrees to radians.
    """
    return degrees * (torch.pi / 180.0)

def haversine_distance_loss_(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Radius of earth in kilometers. Use 3956 for miles
    distance = 6371 * c
    return distance
def haversine_distance_loss(coords_pred, coords_target):
    """
    Calculate the loss based on the Haversine distance
    between predicted and target coordinates.
    """
    lat1, lon1 = coords_pred[:, 0], coords_pred[:, 1]
    lat2, lon2 = coords_target[:, 0], coords_target[:, 1]

    predicted_distance = haversine_distance_loss_(lat1, lon1, lat2, lon2)
    loss = torch.mean((predicted_distance - torch.mean(predicted_distance)) ** 2)  # Mean squared error
    return loss

def scatter_add(values, index, dim, dim_size):
    output = torch.zeros(size=[dim_size, ]).to(values.device)
    return output.scatter_add(dim=dim, index=index, src=values)


def dense_to_sparse(tensor):
    assert tensor.dim() == 2
    index = tensor.nonzero().t().contiguous()
    value = tensor[index[0], index[1]]
    return index, value


def compute_covmat_from_feat(self, feat, label, nclass):
    cov_mats = torch.zeros(nclass, feat.shape[1], feat.shape[1]).float().to(feat.device)
    for i in range(nclass):
        cur_index = torch.where(label == i)[0]
        cur_feat = feat.index_select(dim=0, index=cur_index)
        cur_dfeat = cur_feat - cur_feat.mean(dim=0, keepdim=True)
        cov_mats[i] = torch.mm(cur_dfeat.t(), cur_dfeat) / cur_index.shape[0]
    return cov_mats


def compute_mean_from_feat(feat, label, nclass):
    means = torch.zeros(nclass, feat.shape[1]).float().to(feat.device)
    for i in range(nclass):
        means[i] = feat.index_select(dim=0, index=torch.where(label == i)[0]).mean(dim=0)
    return means


def compute_inv_mat(input_mat):
    cov_mats = fn.normalize(input_mat, dim=2, p=2)  # 1st row normalization
    small_covar_diags = 1e-15 * torch.eye(200).float().repeat(2, 1, 1).to(
        'cuda')  # offset for inverse-mat computation
    try:
        # inverse
        inv_cov_mats = torch.pinverse(
            cov_mats + small_covar_diags * cov_mats.reshape(cov_mats.size()[0], -1).mean(dim=1).unsqueeze(
                dim=1).unsqueeze(dim=2))
    except:
        inv_cov_mats = 1 / cov_mats.diagonal(dim1=-2, dim2=-1).diag_embed()
    inv_cov_mats = fn.normalize(inv_cov_mats, dim=2, p=2)  # 2nd row normalization
    return inv_cov_mats


def compute_gm_term(disen_x, disen_y, disen_means, disen_cov_mats):
    batch_size = disen_x.size()[0]
    inv_cov_mats = compute_inv_mat(input_mat=disen_cov_mats)
    # get the batch samples
    means_batch = torch.index_select(disen_means, dim=0, index=disen_y)
    invcovmat_bath = torch.index_select(inv_cov_mats, dim=0, index=disen_y)
    diff_batch = disen_x - means_batch
    gm_term_batch = torch.matmul(torch.matmul(diff_batch.view(batch_size, 1, -1), invcovmat_bath),
                                 diff_batch.view(batch_size, -1, 1)).squeeze()
    return gm_term_batch


def initialize_mean_cov(args, input_feat):
    disen_y = torch.arange(args.ncaps).long().unsqueeze(dim=0).repeat(args.num_stations, 1).flatten().to('cuda')
    means = compute_mean_from_feat(input_feat.detach().clone().view(-1, args.nhidden), disen_y, args.ncaps)
    cov_mats = compute_covmat_from_feat(args, input_feat.detach().clone().view(-1, args.nhidden), disen_y, args.ncaps)
    return means, cov_mats


def compute_gm_reg_loss(x, y, means, cov_mats):
    return compute_gm_term(x, y, means, cov_mats).mean()


def compute_div_loss(args, disen_likeli):
    tmp = disen_likeli.view(-1, args.ncaps, args.ncaps)
    mat = torch.bmm(tmp, tmp.transpose(dim0=1, dim1=2))
    return (-torch.logdet(
        mat + args.det_offset * torch.eye(args.ncaps).to(mat.device).unsqueeze(dim=0).repeat(mat.shape[0], 1,
                                                                                             1))).mean()


def compute_gm_likeli_(args, disen_x, means, inv_covmats):
    batch_diffs = disen_x.unsqueeze(dim=1) - means.unsqueeze(dim=0).repeat(disen_x.shape[0], 1, 1)
    batch_inv_covmatns = inv_covmats.unsqueeze(dim=0).repeat(disen_x.shape[0], 1, 1, 1)
    batch_gm_term = torch.bmm(
        torch.bmm(batch_diffs.view(-1, 1, args.nhidden), batch_inv_covmatns.view(-1, args.nhidden, args.nhidden)),
        batch_diffs.view(-1, args.nhidden, 1)).view(-1, args.ncaps)
    # remove inf caused by exp(89) --> inf
    z = -0.5 * batch_gm_term
    z = (z.masked_fill(z > 80, 80)).exp()
    gm_likeli_ = fn.normalize(z, dim=-1, p=2)  # l2-norm
    return gm_likeli_

def KL_loss(out, label):
    # 归一化概率分布矩阵
    p_matrix = fn.softmax(out, dim=2)
    q_matrix = fn.softmax(label, dim=2)

    # 创建分布对象
    p_dist = dist.Categorical(probs=p_matrix)
    q_dist = dist.Categorical(probs=q_matrix)

    # 计算KL散度
    kl_divergence = dist.kl_divergence(p_dist, q_dist)

    # 求整个batchsize的平均散度
    average_kl_divergence = torch.mean(kl_divergence, dim=0)

    return average_kl_divergence

def uniform(a, b, *args):
    return a + (b - a) * torch.rand(*args)



# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='[SINE] Location & Magnitude')

    parser.add_argument('--continue_training', type=bool, default=False,
                        help='Whether to continue training')

    parser.add_argument('--train_dataPath', type=str, default=r'DataSet/SCEDC/train_new.txt',
                        help='训练集目录路径')
    parser.add_argument('--test_dataPath', type=str, default=r'DataSet/SCEDC/test_new.txt',
                        help='测试集目录路径')
    parser.add_argument('--train_data_value_Path', type=str,
                        default=r'D:\地震k-net日本数据\南加州地震目录\data_15s.npy',
                        help='训练集数据路径')
    parser.add_argument('--test_data_value_Path', type=str, default=r'D:\地震k-net日本数据\南加州地震目录\data_15s.npy',
                        help='测试集数据路径')

    parser.add_argument('--num_stations', type=int, default=5, help='参与计算的台站个数')
    parser.add_argument('--channel_input', type=int, default=1, help='参与计算的通道个数')
    parser.add_argument('--channel_select', type=int, default=2, help='0: E, 1: N, 2: Z')
    parser.add_argument('--channel_output', type=int, default=64, help='计算输出的通道个数')
    parser.add_argument('--data_input_size', type=int, default=1000, help='输入的数据长度')
    parser.add_argument('--conv_split_size', type=int, default=100, help='一个时间步的长度')
    parser.add_argument('--graph_input_size', type=int, default=100, help='图神经网络的特征长度')
    parser.add_argument('--graph_hidden_size', type=int, default=128, help='图神经网络的隐藏层特征长度')
    parser.add_argument('--graph_output_size', type=int, default=64, help='图神经网络的输出特征长度')
    parser.add_argument('--num_heads', type=int, default=5, help='自注意力头数')
    parser.add_argument('--num_result', type=int, default=5, help='输出为几维')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--aligning_P', type=bool, default=False, help='是否对齐P波')
    parser.add_argument('--normalized', type=bool, default=True, help='是否归一化')
    parser.add_argument('--writer', type=str, default='logs/application_to_seismicity_in_sc_10s_nograph',
                        help='tensorboardX logs dir')
    parser.add_argument('--Continuous_model_path', type=str, default='review')
    parser.add_argument('--writer_log', type=str,
                        default='application_to_seismicity_in_SCEDC',
                        help='tensorboardX logs Description')
    parser.add_argument('--nfeat', type=int, default=100, help='特征数量')
    parser.add_argument('--nclass', type=int, default=2, help='聚类数量')
    parser.add_argument('--ncaps', type=int, default=4, help='邻居数量阈值')
    parser.add_argument('--nhidden', type=int, default=300, help='隐藏层神经元长度')
    parser.add_argument('--device', type=str, default='cuda:0', help='驱动')
    parser.add_argument('--graph_type', type=str, default='knn', help='聚类类型')
    parser.add_argument('--det_offset', type=float, default=1e-6, help='隐藏层神经元长度')
    parser.add_argument('--time_steps', type=int, default=10, help='分割时间步数量')
    parser.add_argument('--std_dev', type=int, default=100, help='生成P波分布的标准差')
    parser.add_argument('--times', type=int, default=10, help='使用多长时间的数据')
    parser.add_argument('--d_model', type=int, default=512, help='降维')

    return parser.parse_args()

