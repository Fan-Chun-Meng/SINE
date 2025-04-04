import argparse
import torch
import os
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from DataSet.MyDataset import MyDataset
from Model.Model_P import Model_P
from Model.SINE import SINE
from Utils.utils import haversine_distance, parse_args
import torch.nn.functional as F

from test import test_model
import configparser

# 创建配置解析器
config = configparser.ConfigParser()

# 读取配置文件
config.read('Utils\config.ini')

# 访问文件中的数据
lat_max = float(config['Label']['LatMax'])
lat_min = float(config['Label']['LatMin'])
lon_max = float(config['Label']['LonMax'])
lon_min = float(config['Label']['LonMin'])
depth_max = float(config['Label']['DepthMax'])
depth_min = float(config['Label']['DepthMin'])
mag_max = float(config['Label']['MagMax'])
mag_min = float(config['Label']['MagMin'])

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"





# Initialize models
def initialize_models(args):
    model = SINE(args).to(args.device)
    model_pick_detection = Model_P(args).to(args.device)

    Lamda1 = torch.ones((1,)).cuda()
    Lamda2 = torch.ones((1,)).cuda()
    Lamda3 = torch.ones((1,)).cuda()
    Lamda4 = torch.ones((1,)).cuda()

    params = ([p for p in model.parameters()] + [Lamda1, Lamda2, Lamda3, Lamda4])
    model_optim = optim.Adam(params, lr=args.learning_rate)
    model_optim_pick_detection = optim.Adam(model_pick_detection.parameters(), lr=args.learning_rate)

    return model, model_pick_detection, model_optim, model_optim_pick_detection, Lamda1, Lamda2, Lamda3, Lamda4


# Loss function for comparison
def loss_fn(output, target, type):
    criterion_arrive = nn.BCEWithLogitsLoss()
    criterion_loc = nn.L1Loss()
    criterion_dep_mag = nn.MSELoss()
    if type == 'detection':
        return criterion_arrive(output, target)

    elif type == 'loc':
        return criterion_loc(output, target)

    else:
        return criterion_dep_mag(output, target)


# Load model from checkpoint
def load_checkpoint(args, model, model_optim, model_pick_detection, model_optim_pick_detection):
    if args.continue_training:
        checkpoint = torch.load(args.writer + '.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        model_pick_detection.load_state_dict(checkpoint['model_state_dict_pick_detection'])
        model_optim_pick_detection.load_state_dict(checkpoint['optimizer_state_dict_pick_detection'])


# Training loop
def train(model, model_pick_detection, model_optim, model_optim_pick_detection, train_loader, args):
    model.train()
    mae_lat_list = []
    mae_lon_list = []
    mae_dep_list = []
    mae_mag_list = []
    mae_arrive_list = []
    rmse_lat_list = []
    rmse_lon_list = []
    rmse_dep_list = []
    rmse_mag_list = []
    r2_lat_list = []
    r2_lon_list = []
    r2_dep_list = []
    r2_mag_list = []
    train_loss = []
    train_dis = []

    model.train()

    for i, (input_data, input_data_fft, edge_index, edge_weight_dis, edge_weight_similar, label, station_list, p_point,
            P_point_label) in enumerate(train_loader):
        input_data, input_data_fft, edge_index, edge_weight_dis, edge_weight_similar, label, station_list, p_point, P_point_label = input_data.to(
            args.device), input_data_fft.to(args.device), edge_index.to(
            args.device), edge_weight_dis.to(args.device), edge_weight_similar.to(args.device), label.to(
            args.device), station_list.to(args.device), p_point.to(
            args.device), P_point_label.to(args.device)

        model_optim.zero_grad()
        model_optim_pick_detection.zero_grad()

        detection_labels = torch.zeros((args.batch_size, args.num_stations, args.data_input_size)).to(
            args.device)
        p_wave_labels = torch.zeros((args.batch_size, args.num_stations, args.data_input_size)).to(
            args.device)

        pick_prob = []
        total_diff = []
        for station_i in range(args.num_stations):
            pick_detection_input = input_data[:, station_i, :].unsqueeze(1)
            pick_detection_out = model_pick_detection(pick_detection_input)
            detection_output = pick_detection_out[0]
            predictions_detected = (detection_output > 0.4).float()
            pick_output = pick_detection_out[1]
            log_probs = F.log_softmax(pick_output, dim=1)

            for batch_i in range(args.batch_size):
                p_wave_idx = int(p_point[batch_i, station_i].item())  # p_point:[batchsize, station]
                if p_wave_idx < args.data_input_size:
                    detection_labels[batch_i, station_i, p_wave_idx:] = 1  # P波起点之后标为地震事件
                    p_wave_labels[batch_i, station_i, p_wave_idx] = 1  # P波起点标为1
                else:
                    p_point[batch_i, station_i] = args.data_input_size

            p_index_output = torch.argmax(pick_output, dim=1).unsqueeze(1)

            # 计算当前台站的损失
            kl_loss = F.kl_div(log_probs, p_wave_labels[:, station_i, :], reduction='batchmean')
            loss_pick_detection = 0.03 * loss_fn(detection_labels[:, station_i, :],
                                                 predictions_detected, type='detection') + 0.4 * kl_loss

            loss_pick_detection.backward()
            model_optim_pick_detection.step()

            pick_prob.append(pick_output)

            abs_diff = torch.abs(p_point[:, station_i] - p_index_output)
            abs_diff[abs_diff > 990] = 0
            total_diff.append(torch.mean(abs_diff.float()))
        total_diff = torch.mean(torch.stack(total_diff))

        pick_prob = torch.sigmoid(torch.stack(pick_prob, dim=1).detach()) + 1

        lat_output, lon_output, dep_output, mag_output = model(input_data, edge_index, edge_weight_dis, station_list,
                                                               pick_prob)

        label_lat = label[:, :1]
        label_lon = label[:, 1].unsqueeze(1)
        label_dep = label[:, 2].unsqueeze(1)
        label_mag = label[:, 3].unsqueeze(1)
        loss_lat = loss_fn(label_lat, lat_output, type='loc')
        loss_lon = loss_fn(label_lon, lon_output, type='loc')
        loc_output = torch.cat([lat_output, lon_output], dim=1)
        # loss_loc += criterion_loc_mse(label[:,:2],loc_output)
        loss_dep = loss_fn(label_dep, dep_output, type='dep')
        loss_mag = loss_fn(label_mag, mag_output, type='mag')
        loss_loc_mag = Lamda1 * (loss_lat) + Lamda2 * loss_lon + Lamda3 * loss_mag + Lamda4 * loss_dep
        loss_loc_mag.backward()
        model_optim.step()

        label[:, 0] = label[:, 0] * (lat_max - lat_min) + lat_min
        # label[:, 1] = label[:, 1] * ((-116.001) - (-119.994)) + (-119.994)
        label[:, 1] = label[:, 1] * (lon_max - lon_min) + lon_min
        label[:, 2] = label[:, 2] * (depth_max - depth_min) + depth_min
        label[:, 3] = label[:, 3] * (mag_max - mag_min) + mag_min
        loc_output[:, 0] = loc_output[:, 0] * (lat_max - lat_min) + lat_min
        # loc_output[:, 1] = loc_output[:, 1] * ((-116.001) - (-119.994)) + (-119.994)
        loc_output[:, 1] = loc_output[:, 1] * (lon_max - lon_min) + lon_min
        dep_output = dep_output * (depth_max - depth_min) + depth_min
        mag_output = mag_output * (mag_max - mag_min) + mag_min

        output = torch.cat([loc_output, dep_output, mag_output], dim=1)

        distance = haversine_distance(label, loc_output)

        train_loss.append(loss_loc_mag.item())
        outputs_ = torch.cat((loc_output, dep_output, mag_output), dim=1)
        outputs = outputs_.to('cpu').detach().clone()
        label = label.to('cpu').detach().clone()
        distance = distance.to('cpu').detach()

        average_max_values = total_diff.to('cpu').detach().clone()
        mae_arrive_list.append(average_max_values.item())

        # 计算 MAE
        train_dis.append(distance.mean().item())
        mae_i = torch.abs(outputs[:, 0] - label[:, 0]).mean().item()
        mae_lat_list.append(mae_i)
        mae_i = torch.abs(outputs[:, 1] - label[:, 1]).mean().item()
        mae_lon_list.append(mae_i)
        mae_i = torch.abs(outputs[:, 2] - label[:, 2]).mean().item()
        mae_dep_list.append(mae_i)
        mae_i = torch.abs(outputs[:, 3] - label[:, 3]).mean().item()
        mae_mag_list.append(mae_i)
        # 计算 RMSE
        rmse_i = torch.sqrt(torch.mean((outputs[:, 0] - label[:, 0]) ** 2)).item()
        rmse_lat_list.append(rmse_i)
        rmse_i = torch.sqrt(torch.mean((outputs[:, 1] - label[:, 1]) ** 2)).item()
        rmse_lon_list.append(rmse_i)

        rmse_i = torch.sqrt(torch.mean((outputs[:, 2] - label[:, 2]) ** 2)).item()
        rmse_dep_list.append(rmse_i)
        rmse_i = torch.sqrt(torch.mean((outputs[:, 3] - label[:, 3]) ** 2)).item()
        rmse_mag_list.append(rmse_i)
        # 计算 R2
        r2_i = 1 - torch.sum((outputs[:, 0] - label[:, 0]) ** 2) / torch.sum(
            (label[:, 0] - torch.mean(label[:, 0])) ** 2).item()
        r2_lat_list.append(r2_i)
        r2_i = 1 - torch.sum((outputs[:, 1] - label[:, 1]) ** 2) / torch.sum(
            (label[:, 1] - torch.mean(label[:, 1])) ** 2).item()
        r2_lon_list.append(r2_i)

        r2_i = 1 - torch.sum((outputs[:, 2] - label[:, 2]) ** 2) / torch.sum(
            (label[:, 2] - torch.mean(label[:, 2])) ** 2).item()
        r2_dep_list.append(r2_i)
        r2_i = 1 - torch.sum((outputs[:, 3] - label[:, 3]) ** 2) / torch.sum(
            (label[:, 3] - torch.mean(label[:, 3])) ** 2).item()
        r2_mag_list.append(r2_i)

        if i % 10 == 0 and i != 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, MAE_dis:{:.4f},MAE_arrive:{:.4f}, MAE_lat: {:.4f}, MAE_lon: {:.4f}, MAE_dep: {:.4f}, MAE_mag: {:.4f}, RMSE_lat: {:.4f}, RMSE_lon: {:.4f}, RMSE_dep: {:.4f}, RMSE_mag: {:.4f}, R2_lat: {:.4f}, R2_lon: {:.4f}, R2_dep: {:.4f}, R2_mag: {:.4f}'
                .format(epoch + 1, args.train_epochs, i + 1, len(train_loader), sum(train_loss) / len(train_loss),
                        sum(train_dis) / len(train_dis), sum(mae_arrive_list) / (len(mae_arrive_list) * 100),
                        sum(mae_lat_list) / len(mae_lat_list),
                        sum(mae_lon_list) / len(mae_lon_list), sum(mae_dep_list) / len(mae_dep_list),
                        sum(mae_mag_list) / len(mae_mag_list), sum(rmse_lat_list) / len(rmse_lat_list),
                        sum(rmse_lon_list) / len(rmse_lon_list),
                        sum(rmse_dep_list) / len(rmse_dep_list), sum(rmse_mag_list) / len(rmse_mag_list),
                        sum(r2_lat_list) / len(r2_lat_list), sum(r2_lon_list) / len(r2_lon_list),
                        sum(r2_dep_list) / len(r2_dep_list), sum(r2_mag_list) / len(r2_mag_list)))
            writer.add_scalar('train_loss', sum(train_loss) / len(train_loss), epoch)
            writer.add_scalar('train_mae_lat', sum(mae_lat_list) / len(mae_lat_list), epoch)
            writer.add_scalar('train_mae_lon', sum(mae_lon_list) / len(mae_lon_list), epoch)
            writer.add_scalar('train_mae_dep', sum(mae_dep_list) / len(mae_dep_list), epoch)
            writer.add_scalar('train_mae_mag', sum(mae_mag_list) / len(mae_mag_list), epoch)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model_optim.state_dict(),
        'model_state_dict_pick_detection': model_pick_detection.state_dict(),
        'optimizer_state_dict_pick_detection': model_optim_pick_detection.state_dict()
    }, args.writer + '.pth')


# Main execution
if __name__ == "__main__":

    torch.manual_seed(42)

    args = parse_args()
    model, model_pick_detection, model_optim, model_optim_pick_detection, Lamda1, Lamda2, Lamda3, Lamda4 = initialize_models(
        args)

    load_checkpoint(args, model, model_optim, model_pick_detection, model_optim_pick_detection)

    # Data loading
    train_loader = torch.utils.data.DataLoader(
        MyDataset(dataPath=args.train_dataPath, dataVluePath=args.train_data_value_Path,
                  data_input_size=args.data_input_size, station_num=args.num_stations, aligning_P=args.aligning_P,
                  channel_select=args.channel_select, normalized=args.normalized, std_dev=args.std_dev),
        batch_size=args.batch_size, shuffle=True, drop_last=True)  # 训练集

    test_loader = torch.utils.data.DataLoader(
        MyDataset(dataPath=args.test_dataPath, dataVluePath=args.test_data_value_Path,
                  data_input_size=args.data_input_size,
                  station_num=args.num_stations, aligning_P=args.aligning_P, channel_select=args.channel_select,
                  normalized=args.normalized, std_dev=args.std_dev),
        batch_size=args.batch_size, shuffle=True, drop_last=True)  # 训练集

    writer = SummaryWriter(log_dir=args.writer)
    writer.add_text('Description', args.writer_log)

    for epoch in range(args.train_epochs):
        train(model, model_pick_detection, model_optim, model_optim_pick_detection, train_loader, args)
        test_model(model, model_pick_detection, model_optim, model_optim_pick_detection, test_loader, epoch, writer, args)