import pandas as pd
import numpy as np
import glob
import os
from scipy.spatial.transform import Rotation as R
from scipy import stats


# 파일 이름에서 타겟 객체를 추출하는 함수
def extract_target_object(file_name):
    parts = file_name.split('_')
    for part in parts:
        if part in ['RedBowl', 'WhiteBowl', 'BlueBowl', 'BronzeBottle', 'WhiteBottle', 'CeladonBottle', 'BlueCup', 'WhiteCup', 'RedCup']:
            return part
    return None

# 각도 계산 함수 (Quaternion 사용)
def calculate_angle_quaternion(head_rot, target_pos, head_pos):
    head_rot = R.from_euler('xyz', head_rot, degrees=True)
    head_dir = head_rot.apply([0, 0, 1])  # 기본 방향 벡터 (앞쪽을 보는 벡터)
    
    target_vector = target_pos - head_pos
    target_dir = target_vector / np.linalg.norm(target_vector)
    
    dot_product = np.dot(head_dir, target_dir)
    angle = np.arccos(dot_product) * 180 / np.pi
    return angle

# 모든 객체에 대한 head angle을 계산하는 함수
def calculate_head_angles(data, object_names, file_name):
    for obj in object_names:
        pos_cols = [f'{obj}_Position_X', f'{obj}_Position_Y', f'{obj}_Position_Z']
        angle_col = f'{obj}_Angle'

        # 필요한 열이 있는지 확인
        if all(col in data.columns for col in pos_cols):
            # 헤드 앵글 계산
            head_angles = data.apply(lambda row: calculate_angle_quaternion(
                [row['HeadRotX'], row['HeadRotY'], row['HeadRotZ']],
                np.array([row[col] for col in pos_cols]),
                np.array([row['HeadPosX'], row['HeadPosY'], row['HeadPosZ']])
            ), axis=1)

            data[f'{obj}_HeadAngle'] = head_angles
            print(f"{file_name} - {obj}의 HeadAngle 계산 완료")
        else:
            print(f"{file_name} - {obj}에 필요한 데이터가 부족합니다. (필요한 열: {pos_cols})")
    return data

# 손에 잡은 객체가 타겟 객체일 때 Grab 이벤트 식별하는 함수
def identify_grab_event_hand(data, target_obj):
    grab_event = data[(data['LHandObjName'] == target_obj) | (data['RHandObjName'] == target_obj)]
    if not grab_event.empty:
        print(f"Hand grab event found for {target_obj} at {grab_event['Timestamp'].iloc[0]}")
        return grab_event['Timestamp']
    return pd.Series([], dtype='datetime64[ns]')

# 타겟 객체의 움직임을 기준으로 Grab 이벤트를 식별하는 함수
def identify_grab_event_obj(data, target_obj, threshold=0.01):
    pos_cols = [f'{target_obj}_Position_X', f'{target_obj}_Position_Y', f'{target_obj}_Position_Z']
    if all(col in data.columns for col in pos_cols):
        movement_data = data[pos_cols].diff().fillna(0).pow(2).sum(axis=1).pow(0.5)
        grab_event_time = data[movement_data > threshold]['Timestamp']
        if not grab_event_time.empty:
            return grab_event_time
    return pd.Series([], dtype='datetime64[ns]')

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

# Grab 이벤트를 설정하는 함수
def set_grab_events(data, object_to_number):
    data['GrabEventOccurred'] = 0  # 초기값은 0으로 설정 (아무것도 안잡았을 때)
    
    # 각 객체에 대해 Grab 이벤트 시간 식별 및 설정
    for obj, num in object_to_number.items():
        grab_time = identify_grab_event_hand(data, obj)
        if grab_time is None or grab_time.empty:
            grab_time = identify_grab_event_obj(data, obj)
        if not (grab_time is None or grab_time.empty):
            data.loc[data['Timestamp'].isin(grab_time), 'GrabEventOccurred'] = num

    return data

# 지정한 body pose 값들의 delta값을 계산하는 함수
def calculate_body_pose_deltas(data, body_pose_cols):
    for col in body_pose_cols:
        delta_col = f'{col}_Delta'
        data[delta_col] = data[col].diff().fillna(0)
    return data

# 지정된 열을 제거하고, 필요한 열의 값을 수정하는 함수
def clean_data(data, columns_to_remove):
    data = data.drop(columns=columns_to_remove, errors='ignore')

    # 각 열이 존재하는지 확인하고 값 수정
    if 'GazeObjName' in data.columns:
        data.loc[data['GazeObjName'].isna(), ['GazeDuration', 'GazeObjName']] = 0
    if 'LHandObjName' in data.columns:
        data.loc[data['LHandObjName'].isna(), ['LeftGrabDuration', 'LHandObjName']] = 0
    if 'RHandObjName' in data.columns:
        data.loc[data['RHandObjName'].isna(), ['RightGrabDuration', 'RHandObjName']] = 0

    return data


# 각도 계산 함수
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.einsum('ij,ij->i', ba, bc) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# 각도를 계산하는 함수
def calculate_body_angles(data):
    angles = pd.DataFrame()

    angles['Angle_RS_LS_LE'] = calculate_angle(data[['RightShoulderX', 'RightShoulderY', 'RightShoulderZ']].values,
                                               data[['LeftShoulderX', 'LeftShoulderY', 'LeftShoulderZ']].values,
                                               data[['LeftElbowX', 'LeftElbowY', 'LeftElbowZ']].values)

    angles['Angle_LS_LE_LW'] = calculate_angle(data[['LeftShoulderX', 'LeftShoulderY', 'LeftShoulderZ']].values,
                                               data[['LeftElbowX', 'LeftElbowY', 'LeftElbowZ']].values,
                                               data[['LeftWristX', 'LeftWristY', 'LeftWristZ']].values)

    angles['Angle_LS_RS_RE'] = calculate_angle(data[['LeftShoulderX', 'LeftShoulderY', 'LeftShoulderZ']].values,
                                               data[['RightShoulderX', 'RightShoulderY', 'RightShoulderZ']].values,
                                               data[['RightElbowX', 'RightElbowY', 'RightElbowZ']].values)

    angles['Angle_RS_RE_RW'] = calculate_angle(data[['RightShoulderX', 'RightShoulderY', 'RightShoulderZ']].values,
                                               data[['RightElbowX', 'RightElbowY', 'RightElbowZ']].values,
                                               data[['RightWristX', 'RightWristY', 'RightWristZ']].values)

    angles['Angle_LS_LH_LK'] = calculate_angle(data[['LeftShoulderX', 'LeftShoulderY', 'LeftShoulderZ']].values,
                                               data[['LeftHipX', 'LeftHipY', 'LeftHipZ']].values,
                                               data[['LeftKneeX', 'LeftKneeY', 'LeftKneeZ']].values)

    angles['Angle_RS_RH_RK'] = calculate_angle(data[['RightShoulderX', 'RightShoulderY', 'RightShoulderZ']].values,
                                               data[['RightHipX', 'RightHipY', 'RightHipZ']].values,
                                               data[['RightKneeX', 'RightKneeY', 'RightKneeZ']].values)

    angles['Angle_LH_LK_LA'] = calculate_angle(data[['LeftHipX', 'LeftHipY', 'LeftHipZ']].values,
                                               data[['LeftKneeX', 'LeftKneeY', 'LeftKneeZ']].values,
                                               data[['LeftAnkleX', 'LeftAnkleY', 'LeftAnkleZ']].values)

    angles['Angle_RH_RK_RA'] = calculate_angle(data[['RightHipX', 'RightHipY', 'RightHipZ']].values,
                                               data[['RightKneeX', 'RightKneeY', 'RightKneeZ']].values,
                                               data[['RightAnkleX', 'RightAnkleY', 'RightAnkleZ']].values)

    return angles

# 타겟 객체 정보를 제외한 데이터 삭제 함수
def remove_unrelated_columns(data, target_object):
        # 기본 유지할 열 목록
    columns_to_keep = [
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
        'Angle_LH_LK_LA', 'Angle_RH_RK_RA'
    ]

    # 모든 객체의 관련 열 추가 (distance, angle, head angle)
    related_columns = ['Distance', 'Angle', 'HeadAngle']
    for obj in object_names:
        for col in related_columns:
            target_col = f'{obj}_{col}'
            if target_col in data.columns:
                columns_to_keep.append(target_col)

    # 모든 객체의 위치 및 회전 정보 추가
    for obj in object_names:
        pos_cols = [f'{obj}_Position_X', f'{obj}_Position_Y', f'{obj}_Position_Z']
        rot_cols = [f'{obj}_Rotation_X', f'{obj}_Rotation_Y', f'{obj}_Rotation_Z']
        columns_to_keep.extend(pos_cols)
        columns_to_keep.extend(rot_cols)

    # 중복된 열 제거 후 데이터프레임 업데이트
    columns_to_keep = list(dict.fromkeys(columns_to_keep))
    data = data[columns_to_keep]

    return data

# 예시로 사용할 객체 이름 리스트
object_names = ['BlueBowl', 'WhiteBowl', 'RedBowl', 'BronzeBottle', 'WhiteBottle', 'CeladonBottle', 'BlueCup', 'WhiteCup', 'RedCup']

# 데이터 파일을 처리하는 함수
def process_files(file_list, columns_to_remove, body_pose_cols):
    all_objects = ['RedBowl', 'WhiteBowl', 'BlueBowl', 'BronzeBottle', 'WhiteBottle', 'CeladonBottle', 'BlueCup', 'WhiteCup', 'RedCup']
    base_columns_to_keep = [
        'Timestamp', 'GrabEventOccurred', 

        'BlueBowl_Position_X', 'BlueBowl_Position_Y', 'BlueBowl_Position_Z', 'BlueBowl_Rotation_X', 'BlueBowl_Rotation_Y', 'BlueBowl_Rotation_Z',
        'BlueCup_Position_X', 'BlueCup_Position_Y', 'BlueCup_Position_Z', 'BlueCup_Rotation_X', 'BlueCup_Rotation_Y', 'BlueCup_Rotation_Z',
        'BronzeBottle_Position_X', 'BronzeBottle_Position_Y', 'BronzeBottle_Position_Z', 'BronzeBottle_Rotation_X', 'BronzeBottle_Rotation_Y', 'BronzeBottle_Rotation_Z',
        'CeladonBottle_Position_X', 'CeladonBottle_Position_Y', 'CeladonBottle_Position_Z', 'CeladonBottle_Rotation_X', 'CeladonBottle_Rotation_Y', 'CeladonBottle_Rotation_Z',
        'RedBowl_Position_X', 'RedBowl_Position_Y', 'RedBowl_Position_Z', 'RedBowl_Rotation_X', 'RedBowl_Rotation_Y', 'RedBowl_Rotation_Z',
        'RedCup_Position_X', 'RedCup_Position_Y', 'RedCup_Position_Z', 'RedCup_Rotation_X', 'RedCup_Rotation_Y', 'RedCup_Rotation_Z', 
        'WhiteBottle_Position_X', 'WhiteBottle_Position_Y', 'WhiteBottle_Position_Z', 'WhiteBottle_Rotation_X', 'WhiteBottle_Rotation_Y', 'WhiteBottle_Rotation_Z',
        'WhiteBowl_Position_X', 'WhiteBowl_Position_Y', 'WhiteBowl_Position_Z', 'WhiteBowl_Rotation_X', 'WhiteBowl_Rotation_Y', 'WhiteBowl_Rotation_Z', 
        'WhiteCup_Position_X', 'WhiteCup_Position_Y', 'WhiteCup_Position_Z', 'WhiteCup_Rotation_X', 'WhiteCup_Rotation_Y', 'WhiteCup_Rotation_Z', 

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
        'Angle_LH_LK_LA', 'Angle_RH_RK_RA'
    ]
    related_columns = ['Distance', 'Angle', 'HeadAngle']
    for obj in all_objects:
        for col in related_columns:
            base_columns_to_keep.append(f'{obj}_{col}')

    for file in file_list:
        print(f"\n파일 처리 중: {file}")

        # 데이터 로드
        data = pd.read_csv(file)

        # 파일 이름에서 타겟 객체 추출
        target_object = extract_target_object(os.path.basename(file))
        if target_object is None:
            print(f"파일에서 타겟 객체를 찾을 수 없습니다: {file}")
            continue

        print(f"타겟 객체: {target_object}")

        # 열 이름 끝의 공백 제거
        data.columns = data.columns.str.strip()
        
        # 지정된 열 제거 및 값 수정
        data = clean_data(data, columns_to_remove)
        
        # 모든 객체에 대한 head angle 계산
        data = calculate_head_angles(data, all_objects, file)

        # Grab 이벤트 설정
        data = set_grab_events(data, object_to_number)

        # 지정된 body pose 값들의 delta 값 계산
        data = calculate_body_pose_deltas(data, body_pose_cols)

        # Body Joint Angles 계산
        angles = calculate_body_angles(data)
        data = pd.concat([data, angles], axis=1)

        # 타겟 객체 정보를 제외한 다른 객체 데이터 삭제
        data = remove_unrelated_columns(data, all_objects)

        # 지정된 열만 유지 (존재하지 않는 열은 무시)
        columns_to_keep_final = [col for col in base_columns_to_keep if col in data.columns]
        data = data[columns_to_keep_final]

        # 처리된 데이터를 새로운 CSV 파일로 저장
        output_dir = 'processed_data_withPosRot'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(file).replace('_merged.csv', '_processed.csv'))
        data.to_csv(output_file, index=False)
        print(f"{file}의 처리된 데이터가 {output_file}에 저장되었습니다.")

def main():
    # 특정 patient 번호와 object name만 전처리할 수 있도록 설정
    patient_numbers = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # 원하는 patient 번호를 리스트에 추가

    # 전처리된 파일 경로 패턴
    file_list = []
    for number in patient_numbers:
        file_pattern = f'../merged_data/merged_data/patient{number}/patient{number}_*_*_merged.csv'
        file_list.extend(glob.glob(file_pattern))

    columns_to_remove = [
        'AttentionRateToBlueBowl', 'AttentionRateToWhiteBowl', 'AttentionRateToRedBowl',
        'AttentionRateToBronzeBottle', 'AttentionRateToWhiteBottle', 'AttentionRateToCeladonBottle',
        'AttentionRateToBlueCup', 'AttentionRateToWhiteCup', 'AttentionRateToRedCup',
        'GazeCount', 'LeftIntCount', 'RightIntCount'
    ]

    # delta 값을 계산할 body pose 열 이름 리스트
    body_pose_cols = ['LeftShoulderX', 'LeftShoulderY', 'LeftShoulderZ', 'RightShoulderX','RightShoulderY', 'RightShoulderZ', 
                      'LeftElbowX', 'LeftElbowY', 'LeftElbowZ', 'RightElbowX', 'RightElbowY', 'RightElbowZ', 
                      'LeftWristX', 'LeftWristY', 'LeftWristZ', 'RightWristX', 'RightWristY', 'RightWristZ', 
                      'LeftHipX', 'LeftHipY', 'LeftHipZ', 'RightHipX', 'RightHipY', 'RightHipZ', 
                      'LeftKneeX', 'LeftKneeY', 'LeftKneeZ', 'RightKneeX', 'RightKneeY', 'RightKneeZ',
                      'LeftAnkleX', 'LeftAnkleY', 'LeftAnkleZ', 'RightAnkleX', 'RightAnkleY','RightAnkleZ']

    # 배치 단위로 파일 처리
    batch_size = 9
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i + batch_size]
        process_files(batch_files, columns_to_remove, body_pose_cols)

if __name__ == "__main__":
    main()