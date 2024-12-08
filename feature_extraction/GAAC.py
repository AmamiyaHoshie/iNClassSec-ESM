import os
import re
from collections import Counter


def read_fasta(file_path):
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith('>'):
                sequence += line
    return sequence


def calculate_gaac(sequence):
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'positivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    sequence = re.sub('-', '', sequence.upper())
    sequence_length = len(sequence)
    if sequence_length == 0:
        return None

    count = Counter(sequence)
    gaac_vector = []
    for key in group:
        group_count = sum([count[aa] for aa in group[key]])
        gaac_vector.append(group_count / sequence_length)
    return gaac_vector


def numeric_sort(file_name):
    return int(file_name)  # 文件名即为数字，直接转换为整数


def process_all_directories_for_gaac(base_directories):
    for directory in base_directories:
        directory_path = os.path.join('../result', directory, 'sequence')

        folder_name = os.path.basename(directory)

        output_file = os.path.join('../features', 'GAAC', f'{folder_name}.txt')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        gaac_vectors = []

        if not os.path.exists(directory_path):
            continue

        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        if not files:
            continue

        files_sorted = sorted(files, key=numeric_sort)

        for file in files_sorted:
            file_path = os.path.join(directory_path, file)
            print(f"Processing file: {file_path}")

            sequence = read_fasta(file_path)

            if not sequence:
                continue

            gaac_vector = calculate_gaac(sequence)
            if gaac_vector is not None:
                gaac_vectors.append(gaac_vector)

        if not gaac_vectors:
            continue

        with open(output_file, 'w') as out_file:
            for gaac_vector in gaac_vectors:
                out_file.write(','.join(map(str, gaac_vector)) + '\n')


base_directories = [
    'PeNGaRoo_independent_test_N',
    'PeNGaRoo_independent_test_P',
    'PeNGaRoo_train_N',
    'PeNGaRoo_train_P'
]

process_all_directories_for_gaac(base_directories)
