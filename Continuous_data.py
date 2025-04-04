import configparser
import os
import time

import numpy as np
import torch
from Model.Model_P import Model_P
from Model.SINE import SINE
from Utils.utils import parse_args
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load model from checkpoint
def load_checkpoint(args, model, model_pick_detection):
    checkpoint = torch.load(args.Continuous_model_path+'.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model_pick_detection.load_state_dict(checkpoint['model_pick_detection_state_dict'])


# Initialize models
def initialize_models(args):
    model = SINE(args).to(args.device)
    model_pick_detection = Model_P(args).to(args.device)

    return model, model_pick_detection

def Data_preprocessing(value):
    for i in range(value.shape[0]):
        ori_list = value[i, :]  # 取出第 i 行
        a = np.polyfit(range(len(ori_list)), ori_list, 10)
        b = np.poly1d(a)
        c = b(range(len(ori_list)))
        value[i, :] = ori_list - c  # 去势后的结果
    return value

def load_npy_files_from_directory(directory):
    datas = np.load(directory, allow_pickle=True).tolist()
    return datas


def load_data(args):
    value = dict_total
    data = value['input_data']
    flag_success = True
    if offset+sampling*window_size <= data.shape[1]:
        data = data[:, int(offset):int(offset+sampling*window_size)]
    else:
        flag_success = False
    try:
        data = Data_preprocessing(data)
        return torch.tensor(data.astype(np.float32)).to(args.device), torch.tensor(value['edge_index']).to(
            args.device), torch.tensor(
            value['edge_weight_dis']).to(args.device), torch.tensor(value['station_list']).to(args.device),flag_success
    except:
        flag_success = False
        return '', '', '', '', flag_success

def is_majority_in_range(lst, lower=900, upper=1000, threshold=0.6):
    count = sum(lower <= x <= upper for x in lst)
    return count / len(lst) >= threshold

def Pick_detection(model, model_pick_detection, WaveFormData, edge_index, edge_weight_dis, station_list):
    with torch.no_grad():
        model.eval()
        model_pick_detection.eval()
        pick_prob = []
        pick_point = []
        for station_id in range(len(WaveFormData)):
            pick_detection_out = model_pick_detection(WaveFormData[station_id,:].unsqueeze(0).unsqueeze(0))
            pick_output = pick_detection_out[1]
            p_index_output = torch.argmax(pick_output, dim=1).unsqueeze(1)
            pick_prob.append(pick_output)
            pick_point.append(p_index_output.item())

        not_P = is_majority_in_range(pick_point)
        if not not_P:
            print('Seismic P-phase detected')
            pick_prob = torch.sigmoid(torch.stack(pick_prob, dim=1).detach()) + 1
            lat_output, lon_output, dep_output, mag_output = model(WaveFormData.unsqueeze(0), edge_index.unsqueeze(0), edge_weight_dis.unsqueeze(0),
                                                                   station_list.unsqueeze(0), pick_prob)
            loc_output = torch.cat([lat_output, lon_output], dim=1)
            loc_output[:, 0] = loc_output[:, 0] * (lat_max - lat_min) + lat_min
            loc_output[:, 1] = loc_output[:, 1] * (lon_max - lon_min) + lon_min
            dep_output = dep_output * (depth_max - depth_min) + depth_min
            mag_output = mag_output * (mag_max - mag_min) + mag_min
            # Print results
            print("Seismic phases at stations:", [x + offset for x in pick_point])
            print(f"Epicenter location: Latitude {loc_output[:,0].item()}, Longitude {loc_output[:,1].item()}")
            print(f"Magnitude: {mag_output.item()}")
            print(f"Depth: {dep_output.item()} km")
            print("=" * 20 + " Calculation completed " + "=" * 20)

        else:
            print('No seismic phase detected ')

# Main execution
if __name__ == "__main__":

    # External Settings
    config = configparser.ConfigParser()
    # Model Settings
    args = parse_args()

    config.read('Utils\config.ini')

    # 创建配置解析器
    config = configparser.ConfigParser()

    # 读取配置文件
    config.read('config.ini')

    # model
    model, model_pick_detection = initialize_models(args)
    load_checkpoint(args, model, model_pick_detection)

    # data
    # data Access
    accessType = config['Access']['type']
    if accessType == 'files':
        data_path = config['Access']['dataPath']
        dict_total = load_npy_files_from_directory(data_path)

    # Label range
    lat_max = float(config['Label']['LatMax'])
    lat_min = float(config['Label']['LatMin'])
    lon_max = float(config['Label']['LonMax'])
    lon_min = float(config['Label']['LonMin'])
    depth_max = float(config['Label']['DepthMax'])
    depth_min = float(config['Label']['DepthMin'])
    mag_max = float(config['Label']['MagMax'])
    mag_min = float(config['Label']['MagMin'])

    # Continuous
    window_size = float(config['Continuous']['window_size'])
    sampling = float(config['Continuous']['sampling'])
    step = float(config['Continuous']['step'])

    flag = True
    offset = 0
    chances = 5
    chance_index = 0

    while flag:
        # Data loading
        WaveFormData, edge_index, edge_weight_dis, station_list, read_flag = load_data(args)

        if read_flag:
            Pick_detection(model, model_pick_detection, WaveFormData, edge_index, edge_weight_dis, station_list)
            offset += step * sampling
            chance_index = 0
        else:
            chance_index += 1
            time.sleep(1)

        if chance_index > chances:
            flag = False




