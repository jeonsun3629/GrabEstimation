import glob
import pandas as pd
import numpy as np
import os
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

import TinyHAR as TinyHAR


# 타겟 객체와 관련된 columns_to_use_all 동적 생성 함수
def get_columns_to_use_all():
    base_columns = [
        'Timestamp', 'GrabEventOccurred', 
        'HeadPosX', 'HeadPosY', 'HeadPosZ', 'HeadRotX', 'HeadRotY', 'HeadRotZ', 
        'EyePosX', 'EyePosY', 'EyePosZ', 'EyeRotX', 'EyeRotY', 'EyeRotZ', 
        'LeftShoulderX_Delta', 'LeftShoulderY_Delta', 'LeftShoulderZ_Delta', 
        'RightShoulderX_Delta', 'RightShoulderY_Delta', 'RightShoulderZ_Delta',
        'LeftElbowX_Delta', 'LeftElbowY_Delta', 'LeftElbowZ_Delta', 
        'RightElbowX_Delta', 'RightElbowY_Delta', 'RightElbowZ_Delta', 
        'LeftWristX_Delta', 'LeftWristY_Delta', 'LeftWristZ_Delta', 
        'RightWristX_Delta', 'RightWristY_Delta', 'RightWristZ_Delta', 
        'LeftHipX_Delta', 'LeftHipY_Delta', 'LeftHipZ_Delta', 
        'RightHipX_Delta', 'RightHipY_Delta', 'RightHipZ_Delta', 
        'LeftKneeX_Delta', 'LeftKneeY_Delta', 'LeftKneeZ_Delta', 
        'RightKneeX_Delta', 'RightKneeY_Delta', 'RightKneeZ_Delta', 
        'LeftAnkleX_Delta', 'LeftAnkleY_Delta', 'LeftAnkleZ_Delta', 
        'RightAnkleX_Delta', 'RightAnkleY_Delta', 'RightAnkleZ_Delta',
        'Angle_RS_LS_LE', 'Angle_LS_LE_LW', 'Angle_LS_RS_RE', 
        'Angle_RS_RE_RW', 'Angle_LS_LH_LK', 'Angle_RS_RH_RK', 
        'Angle_LH_LK_LA', 'Angle_RH_RK_RA', 
        'RedBowl_Position_X', 'RedBowl_Position_Y', 'RedBowl_Position_Z', 
        'RedBowl_Rotation_X', 'RedBowl_Rotation_Y', 'RedBowl_Rotation_Z', 
        'RedBowl_Distance', 'RedBowl_Angle', 'RedBowl_HeadAngle', 
        'WhiteBowl_Position_X', 'WhiteBowl_Position_Y', 'WhiteBowl_Position_Z', 
        'WhiteBowl_Rotation_X', 'WhiteBowl_Rotation_Y', 'WhiteBowl_Rotation_Z', 
        'WhiteBowl_Distance', 'WhiteBowl_Angle', 'WhiteBowl_HeadAngle', 
        'BlueBowl_Position_X', 'BlueBowl_Position_Y', 'BlueBowl_Position_Z', 
        'BlueBowl_Rotation_X', 'BlueBowl_Rotation_Y', 'BlueBowl_Rotation_Z', 
        'BlueBowl_Distance', 'BlueBowl_Angle', 'BlueBowl_HeadAngle', 
        'BronzeBottle_Position_X', 'BronzeBottle_Position_Y', 'BronzeBottle_Position_Z', 
        'BronzeBottle_Rotation_X', 'BronzeBottle_Rotation_Y', 'BronzeBottle_Rotation_Z', 
        'BronzeBottle_Distance', 'BronzeBottle_Angle', 'BronzeBottle_HeadAngle', 
        'WhiteBottle_Position_X', 'WhiteBottle_Position_Y', 'WhiteBottle_Position_Z', 
        'WhiteBottle_Rotation_X', 'WhiteBottle_Rotation_Y', 'WhiteBottle_Rotation_Z', 
        'WhiteBottle_Distance', 'WhiteBottle_Angle', 'WhiteBottle_HeadAngle', 
        'CeladonBottle_Position_X', 'CeladonBottle_Position_Y', 'CeladonBottle_Position_Z', 
        'CeladonBottle_Rotation_X', 'CeladonBottle_Rotation_Y', 'CeladonBottle_Rotation_Z', 
        'CeladonBottle_Distance', 'CeladonBottle_Angle', 'CeladonBottle_HeadAngle', 
        'BlueCup_Position_X', 'BlueCup_Position_Y', 'BlueCup_Position_Z', 
        'BlueCup_Rotation_X', 'BlueCup_Rotation_Y', 'BlueCup_Rotation_Z', 
        'BlueCup_Distance', 'BlueCup_Angle', 'BlueCup_HeadAngle', 
        'WhiteCup_Position_X', 'WhiteCup_Position_Y', 'WhiteCup_Position_Z', 
        'WhiteCup_Rotation_X', 'WhiteCup_Rotation_Y', 'WhiteCup_Rotation_Z', 
        'WhiteCup_Distance', 'WhiteCup_Angle', 'WhiteCup_HeadAngle', 
        'RedCup_Position_X', 'RedCup_Position_Y', 'RedCup_Position_Z', 
        'RedCup_Rotation_X', 'RedCup_Rotation_Y', 'RedCup_Rotation_Z', 
        'RedCup_Distance', 'RedCup_Angle', 'RedCup_HeadAngle'
    ]

    # target_columns = [
    #     f'{target_object}_Position_X', f'{target_object}_Position_Y', f'{target_object}_Position_Z',
    #     f'{target_object}_Rotation_X', f'{target_object}_Rotation_Y', f'{target_object}_Rotation_Z',
    #     f'{target_object}_Distance', f'{target_object}_Angle', f'{target_object}_HeadAngle'
    # ]

    return base_columns

# 데이터 그룹 정의
def get_columns_group(group_name):
    if group_name == "head_eye":
        return [
                'Timestamp',
                'GrabEventOccurred',
                'HeadPosX', 'HeadPosY', 'HeadPosZ', 'HeadRotX', 'HeadRotY', 'HeadRotZ', 
                'EyePosX', 'EyePosY', 'EyePosZ', 'EyeRotX', 'EyeRotY', 'EyeRotZ', 
                'RedBowl_Position_X', 'RedBowl_Position_Y', 'RedBowl_Position_Z', 
                'RedBowl_Rotation_X', 'RedBowl_Rotation_Y', 'RedBowl_Rotation_Z', 
                'RedBowl_Distance', 'RedBowl_Angle', 'RedBowl_HeadAngle', 
                'WhiteBowl_Position_X', 'WhiteBowl_Position_Y', 'WhiteBowl_Position_Z', 
                'WhiteBowl_Rotation_X', 'WhiteBowl_Rotation_Y', 'WhiteBowl_Rotation_Z', 
                'WhiteBowl_Distance', 'WhiteBowl_Angle', 'WhiteBowl_HeadAngle', 
                'BlueBowl_Position_X', 'BlueBowl_Position_Y', 'BlueBowl_Position_Z', 
                'BlueBowl_Rotation_X', 'BlueBowl_Rotation_Y', 'BlueBowl_Rotation_Z', 
                'BlueBowl_Distance', 'BlueBowl_Angle', 'BlueBowl_HeadAngle', 
                'BronzeBottle_Position_X', 'BronzeBottle_Position_Y', 'BronzeBottle_Position_Z', 
                'BronzeBottle_Rotation_X', 'BronzeBottle_Rotation_Y', 'BronzeBottle_Rotation_Z', 
                'BronzeBottle_Distance', 'BronzeBottle_Angle', 'BronzeBottle_HeadAngle', 
                'WhiteBottle_Position_X', 'WhiteBottle_Position_Y', 'WhiteBottle_Position_Z', 
                'WhiteBottle_Rotation_X', 'WhiteBottle_Rotation_Y', 'WhiteBottle_Rotation_Z', 
                'WhiteBottle_Distance', 'WhiteBottle_Angle', 'WhiteBottle_HeadAngle', 
                'CeladonBottle_Position_X', 'CeladonBottle_Position_Y', 'CeladonBottle_Position_Z', 
                'CeladonBottle_Rotation_X', 'CeladonBottle_Rotation_Y', 'CeladonBottle_Rotation_Z', 
                'CeladonBottle_Distance', 'CeladonBottle_Angle', 'CeladonBottle_HeadAngle', 
                'BlueCup_Position_X', 'BlueCup_Position_Y', 'BlueCup_Position_Z', 
                'BlueCup_Rotation_X', 'BlueCup_Rotation_Y', 'BlueCup_Rotation_Z', 
                'BlueCup_Distance', 'BlueCup_Angle', 'BlueCup_HeadAngle', 
                'WhiteCup_Position_X', 'WhiteCup_Position_Y', 'WhiteCup_Position_Z', 
                'WhiteCup_Rotation_X', 'WhiteCup_Rotation_Y', 'WhiteCup_Rotation_Z', 
                'WhiteCup_Distance', 'WhiteCup_Angle', 'WhiteCup_HeadAngle', 
                'RedCup_Position_X', 'RedCup_Position_Y', 'RedCup_Position_Z', 
                'RedCup_Rotation_X', 'RedCup_Rotation_Y', 'RedCup_Rotation_Z', 
                'RedCup_Distance', 'RedCup_Angle', 'RedCup_HeadAngle'
                ]
    elif group_name == "bodypose_deltas_angles":
        return [
                'Timestamp',
                'GrabEventOccurred',
                
                'LeftShoulderX_Delta', 'LeftShoulderY_Delta', 'LeftShoulderZ_Delta', 
                'RightShoulderX_Delta', 'RightShoulderY_Delta', 'RightShoulderZ_Delta',
                'LeftElbowX_Delta', 'LeftElbowY_Delta', 'LeftElbowZ_Delta', 
                'RightElbowX_Delta', 'RightElbowY_Delta', 'RightElbowZ_Delta',
                'LeftWristX_Delta', 'LeftWristY_Delta', 'LeftWristZ_Delta', 
                'RightWristX_Delta', 'RightWristY_Delta', 'RightWristZ_Delta',    
                'LeftHipX_Delta', 'LeftHipY_Delta', 'LeftHipZ_Delta', 
                'RightHipX_Delta', 'RightHipY_Delta', 'RightHipZ_Delta',
                'LeftKneeX_Delta', 'LeftKneeY_Delta', 'LeftKneeZ_Delta', 
                'RightKneeX_Delta', 'RightKneeY_Delta', 'RightKneeZ_Delta',
                'LeftAnkleX_Delta', 'LeftAnkleY_Delta', 'LeftAnkleZ_Delta', 
                'RightAnkleX_Delta', 'RightAnkleY_Delta', 'RightAnkleZ_Delta',
                'Angle_RS_LS_LE', 'Angle_LS_LE_LW', 'Angle_LS_RS_RE', 
                'Angle_RS_RE_RW', 'Angle_LS_LH_LK', 'Angle_RS_RH_RK',
                'Angle_LH_LK_LA', 'Angle_RH_RK_RA',

                'RedBowl_Position_X', 'RedBowl_Position_Y', 'RedBowl_Position_Z', 
                'RedBowl_Rotation_X', 'RedBowl_Rotation_Y', 'RedBowl_Rotation_Z', 
                'RedBowl_Distance', 'RedBowl_Angle', 'RedBowl_HeadAngle', 
                'WhiteBowl_Position_X', 'WhiteBowl_Position_Y', 'WhiteBowl_Position_Z', 
                'WhiteBowl_Rotation_X', 'WhiteBowl_Rotation_Y', 'WhiteBowl_Rotation_Z', 
                'WhiteBowl_Distance', 'WhiteBowl_Angle', 'WhiteBowl_HeadAngle', 
                'BlueBowl_Position_X', 'BlueBowl_Position_Y', 'BlueBowl_Position_Z', 
                'BlueBowl_Rotation_X', 'BlueBowl_Rotation_Y', 'BlueBowl_Rotation_Z', 
                'BlueBowl_Distance', 'BlueBowl_Angle', 'BlueBowl_HeadAngle', 
                'BronzeBottle_Position_X', 'BronzeBottle_Position_Y', 'BronzeBottle_Position_Z', 
                'BronzeBottle_Rotation_X', 'BronzeBottle_Rotation_Y', 'BronzeBottle_Rotation_Z', 
                'BronzeBottle_Distance', 'BronzeBottle_Angle', 'BronzeBottle_HeadAngle', 
                'WhiteBottle_Position_X', 'WhiteBottle_Position_Y', 'WhiteBottle_Position_Z', 
                'WhiteBottle_Rotation_X', 'WhiteBottle_Rotation_Y', 'WhiteBottle_Rotation_Z', 
                'WhiteBottle_Distance', 'WhiteBottle_Angle', 'WhiteBottle_HeadAngle', 
                'CeladonBottle_Position_X', 'CeladonBottle_Position_Y', 'CeladonBottle_Position_Z', 
                'CeladonBottle_Rotation_X', 'CeladonBottle_Rotation_Y', 'CeladonBottle_Rotation_Z', 
                'CeladonBottle_Distance', 'CeladonBottle_Angle', 'CeladonBottle_HeadAngle', 
                'BlueCup_Position_X', 'BlueCup_Position_Y', 'BlueCup_Position_Z', 
                'BlueCup_Rotation_X', 'BlueCup_Rotation_Y', 'BlueCup_Rotation_Z', 
                'BlueCup_Distance', 'BlueCup_Angle', 'BlueCup_HeadAngle', 
                'WhiteCup_Position_X', 'WhiteCup_Position_Y', 'WhiteCup_Position_Z', 
                'WhiteCup_Rotation_X', 'WhiteCup_Rotation_Y', 'WhiteCup_Rotation_Z', 
                'WhiteCup_Distance', 'WhiteCup_Angle', 'WhiteCup_HeadAngle', 
                'RedCup_Position_X', 'RedCup_Position_Y', 'RedCup_Position_Z', 
                'RedCup_Rotation_X', 'RedCup_Rotation_Y', 'RedCup_Rotation_Z', 
                'RedCup_Distance', 'RedCup_Angle', 'RedCup_HeadAngle'
                ]
    elif group_name == "all":
        return get_columns_to_use_all()
    else:
        raise ValueError(f"Unknown group name: {group_name}")


class TinyHAR_Model(nn.Module):
    def __init__(
        self,
        input_shape,
        number_class,
        filter_num,
        nb_conv_layers=4,
        filter_size=5,
        cross_channel_interaction_type="attn",
        cross_channel_aggregation_type="filter",
        temporal_info_interaction_type="gru",
        temporal_info_aggregation_type="FC",
        dropout=0.5,
        activation="ReLU",
    ):
        super(TinyHAR_Model, self).__init__()
        
        self.cross_channel_interaction_type = cross_channel_interaction_type
        self.cross_channel_aggregation_type = cross_channel_aggregation_type
        self.temporal_info_interaction_type = temporal_info_interaction_type
        self.temporal_info_aggregation_type = temporal_info_aggregation_type
        
        """
        PART 1 , ============= 채널 별 특징 추출 =============================        
        입력 형식:  Batch, filter_num, length, Sensor_channel        
        출력 형식:  Batch, filter_num, downsampling_length, Sensor_channel
        """
        filter_num_list = [input_shape[1]]  # 첫 번째 레이어의 in_channels를 input_shape[1]로 설정
        filter_num_step = int(filter_num / nb_conv_layers)
        for i in range(nb_conv_layers - 1):
            filter_num_list.append(filter_num)
        filter_num_list.append(filter_num)

        layers_conv = []
        for i in range(nb_conv_layers):
            in_channel = filter_num_list[i]
            out_channel = filter_num_list[i + 1]
            if i % 2 == 1:
                layers_conv.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (filter_size, 1), (2, 1), padding=(1, 0)),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channel)))
            else:
                layers_conv.append(nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, (filter_size, 1), (1, 1), padding=(1, 0)),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channel)))
        self.layers_conv = nn.ModuleList(layers_conv)
        downsampling_length = self.get_the_shape(input_shape)      
        
        """
        PART2 , ================ 채널 간 상호작용  =================================
        선택 가능:  attn, transformer, identity
        출력 형식:  Batch, filter_num, downsampling_length, Sensor_channel
        """
        self.channel_interaction = TinyHAR.crosschannel_interaction[cross_channel_interaction_type](input_shape[3], filter_num)
        
        """
        PART3 , =============== 채널 융합  ====================================
        선택 가능:  filter, naive, FC
        출력 형식:  Batch, downsampling_length, filter_num
        """
        if cross_channel_aggregation_type == "FC":
            self.channel_fusion = TinyHAR.crosschannel_aggregation[cross_channel_aggregation_type](input_shape[3] * filter_num, filter_num)
        elif cross_channel_aggregation_type in ["SFCC", "SFCF"]:
            self.channel_fusion = TinyHAR.crosschannel_aggregation[cross_channel_aggregation_type](input_shape[3], filter_num)
        else:
            self.channel_fusion = TinyHAR.crosschannel_aggregation[cross_channel_aggregation_type](input_shape[3], filter_num)
        
        self.activation = nn.ReLU()

        """
        PART4 , ============= 시간 정보 추출 =========================
        선택 가능:  gru, lstm, attn, transformer, identity
        출력 형식:  Batch, downsampling_length, filter_num
        """
        self.temporal_interaction = TinyHAR.temporal_interaction[temporal_info_interaction_type](input_shape[3], filter_num)
        
        """
        PART 5 , =================== 시간 정보 융합 ================
        출력 형식:  Batch, downsampling_length, filter_num
        """
        self.dropout = nn.Dropout(dropout)
        
        if temporal_info_aggregation_type == "FC":
            self.flatten = nn.Flatten()
            self.temporal_fusion = TinyHAR.temmporal_aggregation[temporal_info_aggregation_type](downsampling_length * filter_num, filter_num)
        else:
            self.temporal_fusion = TinyHAR.temmporal_aggregation[temporal_info_aggregation_type](input_shape[3], filter_num)
            
        # PART 6 , ==================== 예측 ==============================
        self.prediction = nn.Linear(filter_num, number_class)
    def get_the_shape(self, input_shape):
        x = torch.rand(input_shape)
        for layer in self.layers_conv:
            x = layer(x)
        return x.shape[2]

    def forward(self, x):
        # Batch Filter_num Length Channel
        for layer in self.layers_conv:
            x = layer(x)
            # print(f"After conv layer: {x.shape}")
        x = x.permute(0, 3, 2, 1)
        
        """ =============== 채널 간 상호작용 =============== """
        x = torch.cat([self.channel_interaction(x[:, :, t, :]).unsqueeze(3) for t in range(x.shape[2])], dim=-1)
        x = self.dropout(x)
        
        """ =============== 채널 융합 =============== """
        if self.cross_channel_aggregation_type == "FC":
            x = x.permute(0, 3, 1, 2)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = self.activation(self.channel_fusion(x))
        elif self.cross_channel_aggregation_type in ["SFCC", "SFCF", "SFCF2"]:
            x = x.permute(0, 3, 1, 2)
            x = self.activation(self.channel_fusion(x))
        else:
            x = torch.cat([self.channel_fusion(x[:, :, :, t]).unsqueeze(2) for t in range(x.shape[3])], dim=-1)
            x = x.permute(0, 2, 1)
            x = self.activation(x)
        
        """ =============== 시간 정보 추출 =============== """
        x = self.temporal_interaction(x)

        """ =============== 시간 정보 융합 =============== """
        if self.temporal_info_aggregation_type == "FC":
            x = self.flatten(x)
            x = self.activation(self.temporal_fusion(x))
        else:
            x = self.temporal_fusion(x)
        
        y = self.prediction(x)
        return y

""" =============== 데이터 전처리 =============== """

# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 데이터 전처리 함수
def preprocess_data(data):
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    return data

# 데이터 정규화 함수
def normalize_position(data, columns_to_normalize):
    min_val = data[columns_to_normalize].min().min()
    max_val = data[columns_to_normalize].max().max()
    data[columns_to_normalize] = (data[columns_to_normalize] - min_val) / (max_val - min_val)
    return data

def normalize_rotation(data, columns_to_normalize):
    min_value = 0
    max_value = 360
    data[columns_to_normalize] = (data[columns_to_normalize] - min_value) / (max_value - min_value)
    return data

# 시계열 데이터 생성 함수
def create_time_series_data(data, time_steps, step_size, fps=90):
    X, y = [], []
    feature_columns = data.columns.difference(['Timestamp', 'GrabEventOccurred'])

    sequence_length = fps  # 1초간의 프레임 수

    for start in range(0, len(data) - sequence_length * time_steps + 1, step_size):
        for i in range(time_steps):
            end = start + sequence_length * (i + 1)
            if end <= len(data):
                X_sample = data.iloc[start + sequence_length * i:end][feature_columns].values
                y_sample = data.iloc[end - 1]['GrabEventOccurred']
                X.append(X_sample)
                y.append(y_sample)

    X = np.array(X)
    y = np.array(y)

    return X, y

# 시계열 데이터를 유지하면서 언더샘플링하는 함수
def time_series_under_sample(X, y, target_class_count):
    zero_indices = np.where(y == 0)[0]
    non_zero_indices = np.where(y != 0)[0]

    # 타겟 클래스 수에 맞추어 0 데이터를 샘플링
    zero_indices_under = []
    step_size = max(1, len(zero_indices) // target_class_count)

    for start_idx in range(0, len(zero_indices) - step_size + 1, step_size):
        zero_indices_under.extend(zero_indices[start_idx:start_idx + step_size])
        
        if len(zero_indices_under) >= target_class_count:
            break
    
    zero_indices_under = zero_indices_under[:target_class_count]
    under_sample_indices = np.concatenate([non_zero_indices, zero_indices_under])
    under_sample_indices = under_sample_indices.astype(int)  # 정수형으로 변환
    np.random.shuffle(under_sample_indices)

    X_res = X[under_sample_indices]
    y_res = y[under_sample_indices]

    return X_res, y_res

# 파일을 로드하고 전처리하는 함수
def load_and_preprocess_file(file, time_steps, step_size, columns_to_use, window_size, fps=90):
    df = pd.read_csv(file, usecols=lambda col: col in columns_to_use or col in ['Timestamp', 'GrabEventOccurred'])
    missing_cols = set(columns_to_use) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    df = preprocess_data(df)
    
    position_columns = [col for col in columns_to_use if 'Position' in col]
    rotation_columns = [col for col in columns_to_use if 'Rotation' in col]
    
    df = normalize_position(df, position_columns)
    df = normalize_rotation(df, rotation_columns)

    last_true_index = df[df['GrabEventOccurred'] != 0].index.max()

    if last_true_index is not None and not np.isnan(last_true_index):
        df = df.iloc[:last_true_index + fps + 1]

    try:
        X, y = create_time_series_data(df, time_steps, step_size, fps)
    except Exception as e:
        print(f"Error creating time series data from {file}: {e}")
        return np.array([]), np.array([])

    return X, y

# 데이터 로더 함수
def get_dataloader(folder_pattern, time_steps, batch_size, target_objects, train_patients, test_patients, window_size, step_size, num_workers=4, group=None):
    all_files = glob.glob(folder_pattern)
    X_train, y_train, X_test, y_test = [], [], [], []

    print(f"Total files found: {len(all_files)}")

    for file in all_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) < 4:
            print(f"Unexpected file name format: {filename}")
            continue

        patient_id = int(parts[0].replace('patient', ''))
        target_object = parts[2]

        if target_object not in target_objects:
            continue

        columns_to_use = get_columns_group(group)

        if not columns_to_use:
            print(f"No columns found for group {group} in file {filename}")
            continue

        X, y = load_and_preprocess_file(file, time_steps, step_size, columns_to_use, window_size)
        if X.size > 0 and y.size > 0:
            if patient_id in train_patients:
                X_train.append(X)
                y_train.append(y)
            elif patient_id in test_patients:
                X_test.append(X)
                y_test.append(y)

    if not X_train or not y_train or not X_test or not y_test:
        raise ValueError("No data found after processing all files.")

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # print(f"X_train shape: {X_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"y_test shape: {y_test.shape}")

    # 클래스 1~9의 데이터 수 계산
    non_zero_train_counts = [np.sum(y_train == i) for i in range(1, 10)]
    non_zero_test_counts = [np.sum(y_test == i) for i in range(1, 10)]

    # 클래스 1~9의 평균 수 계산
    train_target_class_count = int(np.mean(non_zero_train_counts))
    test_target_class_count = int(np.mean(non_zero_test_counts))

    # 클래스 0의 데이터 수를 조정
    X_train, y_train = time_series_under_sample(X_train, y_train, train_target_class_count)
    X_test, y_test = time_series_under_sample(X_test, y_test, test_target_class_count)

    # 데이터 프레임으로 변환
    num_features = X_train.shape[2]  # 각 time step 당 feature 수
    sequence_length = window_size  # 90

    columns_to_use = [col for col in columns_to_use if col not in ['Timestamp', 'GrabEventOccurred']]
    columns = [f'{col}_t{i}' for i in range(sequence_length) for col in columns_to_use]

    assert len(columns) == num_features * sequence_length, f"Length of columns list ({len(columns)}) does not match the reshaped data dimensions ({num_features * sequence_length})"
    print(f"Columns: {columns[:10]}...")  # 일부 열 이름만 출력

    train_data = pd.DataFrame(X_train.reshape(X_train.shape[0], -1), columns=columns)
    train_data['Label'] = y_train

    test_data = pd.DataFrame(X_test.reshape(X_test.shape[0], -1), columns=columns)
    test_data['Label'] = y_test

    # # CSV로 저장
    # train_data.to_csv('train_data.csv', index=False)
    # test_data.to_csv('test_data.csv', index=False)

    # print("Train and test data saved to 'train_data.csv' and 'test_data.csv'")

    X_train = np.transpose(X_train, (0, 2, 1))
    X_train = np.expand_dims(X_train, axis=1)

    X_test = np.transpose(X_test, (0, 2, 1))
    X_test = np.expand_dims(X_test, axis=1)

    print(f"X_train shape after transpose: {X_train.shape}")
    print(f"X_test shape after transpose: {X_test.shape}")

    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Train class distribution: {class_distribution}")

    unique, counts = np.unique(y_test, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Test class distribution: {class_distribution}")

    return DataLoader(CustomDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           DataLoader(CustomDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers), \
           num_features

""" =============== 모델 학습 =============== """

# 객체에 숫자 배정
object_to_number = {
    'RedBowl': 1,
    'WhiteBowl': 2,
    'BlueBowl': 3,
    'BronzeBottle': 4,
    'WhiteBottle': 5,
    'CeladonBottle': 6,
    'BlueCup': 7,
    'WhiteCup': 8,
    'RedCup': 9
}

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 예측 시각화 함수
def visualize_predictions(model, test_loader, epoch, num_samples=10):
    model.eval()
    all_inputs = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_inputs.append(inputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    indices = np.random.choice(len(all_labels), num_samples, replace=False)
    selected_inputs = all_inputs[indices]
    selected_labels = all_labels[indices]
    selected_predictions = all_predictions[indices]

    # reshape 입력 데이터로 각 샘플의 feature를 행, time step을 열로 맞춤
    selected_inputs = selected_inputs.reshape(num_samples, selected_inputs.shape[2], selected_inputs.shape[3])

    for i in range(num_samples):
        print(f'Epoch {epoch+1} - Test Input {i + 1}:')
        print(pd.DataFrame(selected_inputs[i]))  # 시퀀스의 처음 10개 프레임만 출력
        print(f'True Label: {selected_labels[i]}, Predicted: {selected_predictions[i]}')
        print('-' * 50)

# 모델 학습 함수
def train_model(train_loader, test_loader, input_dim, time_steps, result_dir, group, fold_index):
    # 입력 형식을 (Batch, Channel, Features, Length)로 설정합니다.
    input_shape = (train_loader.batch_size, 1, input_dim, train_loader.dataset[0][0].shape[2])
    print(f"Input shape for model: {input_shape}")
    
    model = TinyHAR_Model(input_shape=input_shape, number_class=10, filter_num=64).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    test_accuracies = []
    best_accuracy = 0
    f1_scores = []
    precisions = []
    recalls = []
    confusion_matrices = []

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/50 (Training)', leave=False):
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'Epoch {epoch + 1}/50 (Testing)', leave=False):
                inputs, labels = inputs.to(device).float(), labels.to(device).long()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        confusion_matrices.append(conf_matrix)

        print(f'Epoch {epoch+1}/50, Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

        # Best 모델 저장
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(result_dir, f"result_{group}_{time_steps}_best_model_fold{fold_index}_all_objects.pth"))

        scheduler.step()

    # 결과 저장
    test_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'f1_scores': f1_scores,
        'precisions': precisions,
        'recalls': recalls,
        'confusion_matrices': confusion_matrices,
    }
    output_file = os.path.join(result_dir, f"result_{group}_{time_steps}_test_data_fold{fold_index}_all_objects.pth")
    torch.save(test_data, output_file)
    torch.save(model.state_dict(), os.path.join(result_dir, f"result_{group}_{time_steps}_last_model_fold{fold_index}_all_objects.pth"))
    print(f"-----Test data and models saved to {output_file} and last_model_fold{fold_index}_all_objects.pth-----")

    # 성능 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Test Accuracy')
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"result_{group}_{time_steps}_performance_fold{fold_index}_all_objects.png"))
    plt.close()

def main():
    print(f'Using device: {device}')
    folder_pattern = 'D:/Python/ARGradProject/initial_test/processed_data_withPosRot/*_processed.csv'
    time_steps_list = [1, 2, 3] 
    target_objects = ['RedBowl', 'WhiteBowl', 'BlueBowl', 'BronzeBottle', 'WhiteBottle', 'CeladonBottle', 'BlueCup', 'WhiteCup', 'RedCup']
    groups = ['all', 'head_eye', 'bodypose_deltas_angles']

    all_patients = list(range(5, 23))
    random.seed(42)
    random.shuffle(all_patients)

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for time_steps in time_steps_list:
        print(f"-----Training model with {time_steps}s time steps...-----")
        for fold_index, (train_index, test_index) in enumerate(kf.split(all_patients)):
            train_patients = [all_patients[i] for i in train_index]
            test_patients = [all_patients[i] for i in test_index]

            for group in groups:
                print(f"Training with group: {group}, Fold: {fold_index + 1}")
                train_loader, test_loader, input_dim = get_dataloader(
                    folder_pattern, time_steps, 128, target_objects, train_patients, test_patients, 90, 5, 4, group
                )
                train_model(train_loader, test_loader, input_dim, time_steps, 'final_data', group, fold_index)

if __name__ == "__main__":
    main()