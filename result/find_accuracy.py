import os
import sys
import numpy as np

# TinyHAR.py 파일의 경로를 추가합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from sklearn.metrics import f1_score
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import initial_test.Final_TinyHAR as TinyHAR
import conditions

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def calculate_f1_score(predictions, labels):
    if predictions is not None and labels is not None:
        return f1_score(labels, predictions)
    else:
        return 0

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing', leave=False):
            inputs, labels = inputs.to(device).float(), labels.to(device).float()  # Convert inputs and labels to float32 and move to device
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze(1) == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    f1 = calculate_f1_score(all_predictions, all_labels)
    return accuracy, f1

def load_and_evaluate_model(folder_pattern, target_object, time_steps, batch_size, group_name, test_patients, train_patients, window_size, step_size, augment_ratio, model_path):    
    _, test_loader, input_dim = conditions.get_dataloader(
        folder_pattern, target_object, time_steps, batch_size, group_name, test_patients, train_patients, window_size, step_size, augment_ratio
    )

    # 입력 형식을 (Batch, Features, Length, Channel)로 설정합니다.
    input_shape = (batch_size, input_dim, window_size, 1)
    print(f"Input shape for model: {input_shape}")
    
    model = TinyHAR.TinyHAR_Model(input_shape=input_shape, number_class=1, filter_num=64).to(device)
    model.load_state_dict(torch.load(model_path))

    accuracy, f1 = evaluate_model(model, test_loader)
    return accuracy, f1

def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]

def main():
    folder_pattern = 'D:/Python/ARGradProject/initial_test/processed_data_withPosRot/*_processed.csv'
    time_steps_list = [1, 2, 3]
    target_objects = ['RedBowl', 'WhiteBowl', 'BlueBowl', 'BronzeBottle', 'WhiteBottle', 'CeladonBottle', 'BlueCup', 'WhiteCup', 'RedCup']
    group_names = ["all", "head_eye", "bodypose_deltas_angles"]
    result_dir = 'D:/Python/ARGradProject/initial_test/final_data'

    all_patients = list(range(5, 23))
    random.seed(42)
    random.shuffle(all_patients)
    split_idx = int(0.9 * len(all_patients))
    train_patients = all_patients[:split_idx]
    test_patients = all_patients[split_idx:]

    for time_steps in time_steps_list:
        accuracy_results = {group_name: [] for group_name in group_names}

        for target_object in target_objects:
            for group_name in group_names:
                model_path = os.path.join(result_dir, f"result_{time_steps}_best_model_{target_object}_{group_name}.pth")
                if os.path.exists(model_path):
                    accuracy, f1 = load_and_evaluate_model(
                        folder_pattern, target_object, time_steps, 128, group_name,
                        test_patients, train_patients, 90, 5, 0.1, model_path
                    )
                    accuracy_results[group_name].append(accuracy)
                    print(f"Evaluated {model_path} -> Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}")

        print(f"Results for time_steps {time_steps}:")
        for group_name in group_names:
            accuracies = accuracy_results[group_name]
            if accuracies:
                accuracies_no_outliers = remove_outliers(accuracies)
                mean_accuracy = np.mean(accuracies_no_outliers)
                std_accuracy = np.std(accuracies_no_outliers)
                print(f"Group Name: {group_name}")
                print(f"Mean Accuracy (without outliers): {mean_accuracy:.2f}%")
                print(f"Standard Deviation (without outliers): {std_accuracy:.2f}%")
                print("----")

if __name__ == "__main__":
    main()