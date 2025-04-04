import configparser

import torch
from Utils.utils import haversine_distance
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
def test_model(model, model_pick_detection, model_optim, model_optim_pick_detection, test_loader, epoch, writer, args):
    with torch.no_grad():
        model.eval()
        model_pick_detection.eval()
        # Initialize lists to store metrics
        mae_lat_list, mae_lon_list, mae_dep_list, mae_mag_list = [], [], [], []
        mae_arrive_list, rmse_lat_list, rmse_lon_list = [], [], []
        rmse_dep_list, rmse_mag_list, r2_lat_list = [], [], []
        r2_lon_list, r2_dep_list, r2_mag_list = [], [], []
        train_loss, train_dis= [], []


          # Set model to training mode

        for i, (
                input_data, input_data_fft, edge_index, edge_weight_dis, edge_weight_similar,
                label, station_list, p_point, P_point_label) in enumerate(test_loader):

            # Move data to device
            input_data, input_data_fft, edge_index, edge_weight_dis, edge_weight_similar = (
                input_data.to(args.device), input_data_fft.to(args.device), edge_index.to(args.device),
                edge_weight_dis.to(args.device), edge_weight_similar.to(args.device)
            )
            label, station_list, p_point, P_point_label = (
                label.to(args.device), station_list.to(args.device), p_point.to(args.device),
                P_point_label.to(args.device)
            )

            # Zero the gradients
            model_optim.zero_grad()
            model_optim_pick_detection.zero_grad()

            # Initialize labels for detection
            detection_labels = torch.zeros((args.batch_size, args.num_stations, args.data_input_size)).to(args.device)
            p_wave_labels = torch.zeros((args.batch_size, args.num_stations, args.data_input_size)).to(args.device)

            pick_prob = []
            total_diff = []

            # Loop through each station
            for station_i in range(args.num_stations):
                pick_detection_input = input_data[:, station_i, :].unsqueeze(1)
                pick_detection_out = model_pick_detection(pick_detection_input)
                pick_output = pick_detection_out[1]

                # Process each batch
                for batch_i in range(args.batch_size):
                    p_wave_idx = int(p_point[batch_i, station_i].item())
                    if p_wave_idx < args.data_input_size:
                        detection_labels[batch_i, station_i, p_wave_idx:] = 1
                        p_wave_labels[batch_i, station_i, p_wave_idx] = 1
                    else:
                        p_point[batch_i, station_i] = args.data_input_size

                p_index_output = torch.argmax(pick_output, dim=1).unsqueeze(1)
                pick_prob.append(pick_output)

                # Calculate absolute differences
                abs_diff = torch.abs(p_point[:, station_i] - p_index_output)
                abs_diff[abs_diff > 990] = 0
                total_diff.append(torch.mean(abs_diff.float()))

            # Calculate total difference across all stations
            total_diff = torch.mean(torch.stack(total_diff))

            # Apply sigmoid and stack pick probabilities
            pick_prob = torch.sigmoid(torch.stack(pick_prob, dim=1).detach()) + 1

            # Get model outputs
            lat_output, lon_output, dep_output, mag_output = model(input_data, edge_index, edge_weight_dis,
                                                                   station_list, pick_prob)
            loc_output = torch.cat([lat_output, lon_output], dim=1)

            # Rescale labels and outputs to original range
            label[:, 0] = label[:, 0] * (lat_max - lat_min) + lat_min
            label[:, 1] = label[:, 1] * (lon_max - lon_min) + lon_min
            label[:, 2] = label[:, 2] * (depth_max - depth_min) + depth_min
            label[:, 3] = label[:, 3] * (mag_max - mag_min) + mag_min
            loc_output[:, 0] = loc_output[:, 0] * (lat_max - lat_min) + lat_min
            loc_output[:, 1] = loc_output[:, 1] * (lon_max - lon_min) + lon_min
            dep_output = dep_output * (depth_max - depth_min) + depth_min
            mag_output = mag_output * (mag_max - mag_min) + mag_min

            # Calculate distance
            distance = haversine_distance(label, loc_output)

            # Collect results for metric calculation
            outputs_ = torch.cat((loc_output, dep_output, mag_output), dim=1)
            outputs = outputs_.to('cpu').detach().clone()
            label = label.to('cpu').detach().clone()
            distance = distance.to('cpu').detach()

            average_max_values = total_diff.to('cpu').detach().clone()
            mae_arrive_list.append(average_max_values.item())

            # MAE
            mae_lat_list.append(torch.abs(outputs[:, 0] - label[:, 0]).mean().item())
            mae_lon_list.append(torch.abs(outputs[:, 1] - label[:, 1]).mean().item())
            mae_dep_list.append(torch.abs(outputs[:, 2] - label[:, 2]).mean().item())
            mae_mag_list.append(torch.abs(outputs[:, 3] - label[:, 3]).mean().item())

            # RMSE
            rmse_lat_list.append(torch.sqrt(torch.mean((outputs[:, 0] - label[:, 0]) ** 2)).item())
            rmse_lon_list.append(torch.sqrt(torch.mean((outputs[:, 1] - label[:, 1]) ** 2)).item())
            rmse_dep_list.append(torch.sqrt(torch.mean((outputs[:, 2] - label[:, 2]) ** 2)).item())
            rmse_mag_list.append(torch.sqrt(torch.mean((outputs[:, 3] - label[:, 3]) ** 2)).item())

            # R2
            r2_lat_list.append(1 - torch.sum((outputs[:, 0] - label[:, 0]) ** 2) /
                               torch.sum((label[:, 0] - torch.mean(label[:, 0])) ** 2).item())
            r2_lon_list.append(1 - torch.sum((outputs[:, 1] - label[:, 1]) ** 2) /
                               torch.sum((label[:, 1] - torch.mean(label[:, 1])) ** 2).item())
            r2_dep_list.append(1 - torch.sum((outputs[:, 2] - label[:, 2]) ** 2) /
                               torch.sum((label[:, 2] - torch.mean(label[:, 2])) ** 2).item())
            r2_mag_list.append(1 - torch.sum((outputs[:, 3] - label[:, 3]) ** 2) /
                               torch.sum((label[:, 3] - torch.mean(label[:, 3])) ** 2).item())

        print(
            'Test MAE_arrive:{:.4f}, MAE_lat: {:.4f}, MAE_lon: {:.4f}, MAE_dep: {:.4f}, MAE_mag: {:.4f}, RMSE_lat: {:.4f}, RMSE_lon: {:.4f}, RMSE_dep: {:.4f}, RMSE_mag: {:.4f}, R2_lat: {:.4f}, R2_lon: {:.4f}, R2_dep: {:.4f}, R2_mag: {:.4f}'
            .format(sum(mae_arrive_list)/(len(mae_arrive_list)*100),
                    sum(mae_lat_list) / len(mae_lat_list),
                    sum(mae_lon_list) / len(mae_lon_list), sum(mae_dep_list) / len(mae_dep_list),
                    sum(mae_mag_list) / len(mae_mag_list), sum(rmse_lat_list) / len(rmse_lat_list),
                    sum(rmse_lon_list) / len(rmse_lon_list),
                    sum(rmse_dep_list) / len(rmse_dep_list), sum(rmse_mag_list) / len(rmse_mag_list),
                    sum(r2_lat_list) / len(r2_lat_list), sum(r2_lon_list) / len(r2_lon_list),
                    sum(r2_dep_list) / len(r2_dep_list), sum(r2_mag_list) / len(r2_mag_list)))


        writer.add_scalar('Test MAE_lat', sum(mae_lat_list) / len(mae_lat_list), epoch)
        writer.add_scalar('Test MAE_lon', sum(mae_lon_list) / len(mae_lon_list), epoch)
        writer.add_scalar('Test MAE_dep', sum(mae_dep_list) / len(mae_dep_list), epoch)
        writer.add_scalar('Test MAE_mag', sum(mae_mag_list) / len(mae_mag_list), epoch)
