import torch
import os
import scipy.stats as stats
from itertools import combinations

# 파일 경로 리스트
file_paths = [
    "result_all_1_test_data_fold0_all_objects.pth",
    "result_all_1_test_data_fold1_all_objects.pth",
    "result_all_1_test_data_fold2_all_objects.pth",
    "result_all_1_test_data_fold3_all_objects.pth",
    "result_all_2_test_data_fold0_all_objects.pth",
    "result_all_2_test_data_fold1_all_objects.pth",
    "result_all_2_test_data_fold2_all_objects.pth",
    "result_all_2_test_data_fold3_all_objects.pth",
    "result_all_3_test_data_fold0_all_objects.pth",
    "result_all_3_test_data_fold1_all_objects.pth",
    "result_all_3_test_data_fold2_all_objects.pth",
    "result_all_3_test_data_fold3_all_objects.pth",
    "result_bodypose_deltas_angles_1_test_data_fold0_all_objects.pth",
    "result_bodypose_deltas_angles_1_test_data_fold1_all_objects.pth",
    "result_bodypose_deltas_angles_1_test_data_fold2_all_objects.pth",
    "result_bodypose_deltas_angles_1_test_data_fold3_all_objects.pth",
    "result_bodypose_deltas_angles_2_test_data_fold0_all_objects.pth",
    "result_bodypose_deltas_angles_2_test_data_fold1_all_objects.pth",
    "result_bodypose_deltas_angles_2_test_data_fold2_all_objects.pth",
    "result_bodypose_deltas_angles_2_test_data_fold3_all_objects.pth",
    "result_bodypose_deltas_angles_3_test_data_fold0_all_objects.pth",
    "result_bodypose_deltas_angles_3_test_data_fold1_all_objects.pth",
    "result_bodypose_deltas_angles_3_test_data_fold2_all_objects.pth",
    "result_bodypose_deltas_angles_3_test_data_fold3_all_objects.pth",
    "result_head_eye_1_test_data_fold0_all_objects.pth",
    "result_head_eye_1_test_data_fold1_all_objects.pth",
    "result_head_eye_1_test_data_fold2_all_objects.pth",
    "result_head_eye_1_test_data_fold3_all_objects.pth",
    "result_head_eye_2_test_data_fold0_all_objects.pth",
    "result_head_eye_2_test_data_fold1_all_objects.pth",
    "result_head_eye_2_test_data_fold2_all_objects.pth",
    "result_head_eye_2_test_data_fold3_all_objects.pth",
    "result_head_eye_3_test_data_fold0_all_objects.pth",
    "result_head_eye_3_test_data_fold1_all_objects.pth",
    "result_head_eye_3_test_data_fold2_all_objects.pth",
    "result_head_eye_3_test_data_fold3_all_objects.pth",
]

# 결과 저장 딕셔너리
results = {
    'all_1sec': [],
    'all_2sec': [],
    'all_3sec': [],
    'bodypose_1sec': [],
    'bodypose_2sec': [],
    'bodypose_3sec': [],
    'head_eye_1sec': [],
    'head_eye_2sec': [],
    'head_eye_3sec': []
}

# 파일 로드 및 마지막 fold 정확도 추출
for file_path in file_paths:
    if os.path.exists(file_path):
        data = torch.load(file_path)
        test_accuracies = data.get('test_accuracies', None)
        if test_accuracies:
            # 마지막 fold의 정확도만 추출
            last_fold_accuracy = test_accuracies[-1]
            result = {'file': file_path, 'last_fold_accuracy': last_fold_accuracy}
            if 'result_all_1' in file_path:
                results['all_1sec'].append(result)
            elif 'result_all_2' in file_path:
                results['all_2sec'].append(result)
            elif 'result_all_3' in file_path:
                results['all_3sec'].append(result)
            elif 'result_bodypose_deltas_angles_1' in file_path:
                results['bodypose_1sec'].append(result)
            elif 'result_bodypose_deltas_angles_2' in file_path:
                results['bodypose_2sec'].append(result)
            elif 'result_bodypose_deltas_angles_3' in file_path:
                results['bodypose_3sec'].append(result)
            elif 'result_head_eye_1' in file_path:
                results['head_eye_1sec'].append(result)
            elif 'result_head_eye_2' in file_path:
                results['head_eye_2sec'].append(result)
            elif 'result_head_eye_3' in file_path:
                results['head_eye_3sec'].append(result)

# 각 범주별 마지막 fold 정확도 출력
for category in results:
    print(f"\nCategory: {category}")
    for res in results[category]:
        print(f"File: {res['file']} - Last Fold Accuracy: {res['last_fold_accuracy']}")

# 범주별 정확도 리스트 생성
accuracies = {category: [res['last_fold_accuracy'] for res in results[category]] for category in results}

# 모든 범주 조합에 대해 t-test 수행
categories = list(accuracies.keys())
for (cat1, cat2) in combinations(categories, 2):
    acc1 = accuracies[cat1]
    acc2 = accuracies[cat2]
    t_stat, p_value = stats.ttest_ind(acc1, acc2)
    print(f"\nComparison: {cat1} vs {cat2}")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")
    if p_value < 0.05:
        print("The difference in accuracies is statistically significant (p < 0.05).")
    else:
        print("The difference in accuracies is not statistically significant (p >= 0.05).")