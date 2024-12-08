import os
import numpy as np


def read_pssm_file(file_path):
    PSSM_matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            if len(line.split()) > 42:
                PSSM_matrix.append(line.split()[2:22])
    return np.array(PSSM_matrix)


def pse_pssm(PSSM_matrix, alpha=1):
    PSSM_norm = normalizePSSM(PSSM_matrix)
    L = PSSM_norm.shape[0]  # 序列长度
    avg_pssm = np.mean(PSSM_norm, axis=0)  # 计算均值
    diff_pssm = np.zeros(20)

    for i in range(L - alpha):
        diff_pssm += (PSSM_norm[i] - PSSM_norm[i + alpha]) ** 2
    diff_pssm /= (L - alpha)

    pse_pssm_vector = np.hstack((avg_pssm, diff_pssm))
    return pse_pssm_vector


def normalizePSSM(PSSM_matrix):
    PSSM = PSSM_matrix[:, :20].astype(float)
    mean_matrix = np.mean(PSSM, axis=1, keepdims=True)
    std_matrix = np.std(PSSM, axis=1, keepdims=True)

    std_matrix[std_matrix == 0] = 1

    PSSM_norm = (PSSM - mean_matrix) / std_matrix
    return PSSM_norm


def numeric_sort(file_name):
    """
    从文件名中提取数字并进行排序
    """
    return int(os.path.splitext(file_name)[0])



def process_pssm_directory(input_dir, output_file):
    feature_vectors = []

    files = sorted(os.listdir(input_dir), key=numeric_sort)

    for file_name in files:
        file_path = os.path.join(input_dir, file_name)

        if os.path.isfile(file_path):
            print(f"Processing file: {file_path}")
            PSSM_matrix = read_pssm_file(file_path)
            pse_pssm_vector = pse_pssm(PSSM_matrix)
            feature_vectors.append(pse_pssm_vector)

    feature_matrix = np.array(feature_vectors)
    np.savetxt(output_file, feature_matrix, delimiter=',', fmt="%.8f")
    print(f"Features saved to {output_file}")


def main():
    process_pssm_directory(
        input_dir="../result/PeNGaRoo_train_P/pssm_profile_uniref50/",
        output_file="../features/pse-pssm/PeNGaRoo_train_P.txt"
    )

    process_pssm_directory(
        input_dir="../result/PeNGaRoo_train_N/pssm_profile_uniref50/",
        output_file="../features/pse-pssm/PeNGaRoo_train_N.txt"
    )

    process_pssm_directory(
        input_dir="../result/PeNGaRoo_independent_test_P/pssm_profile_uniref50/",
        output_file="../features/pse-pssm/PeNGaRoo_independent_test_P.txt"
    )

    process_pssm_directory(
        input_dir="../result/PeNGaRoo_independent_test_N/pssm_profile_uniref50/",
        output_file="../features/pse-pssm/PeNGaRoo_independent_test_N.txt"
    )

if __name__ == "__main__":
    main()
