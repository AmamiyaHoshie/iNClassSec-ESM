import os
import re


def read_fasta(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    return sequence


def calculate_cksaap(sequence, gap=3, aa_order='ACDEFGHIKLMNPQRSTVWY'):
    sequence = re.sub('-', '', sequence)
    sequence_length = len(sequence)

    if sequence_length < gap + 2:
        return None

    aa_pairs = [aa1 + aa2 for aa1 in aa_order for aa2 in aa_order]

    cksaap_vector = []

    for g in range(gap + 1):
        pair_count = {pair: 0 for pair in aa_pairs}
        sum_pairs = 0

        for i in range(sequence_length - g - 1):
            if sequence[i] in aa_order and sequence[i + g + 1] in aa_order:
                pair = sequence[i] + sequence[i + g + 1]
                pair_count[pair] += 1
                sum_pairs += 1

        cksaap_vector.extend([pair_count[pair] / sum_pairs if sum_pairs > 0 else 0 for pair in aa_pairs])

    if all(value == 0 for value in cksaap_vector):
        return None

    return cksaap_vector


def numeric_sort(file_name):
    return int(file_name)


def process_all_directories(base_directories, gap=3):
    for directory in base_directories:

        directory_path = os.path.join('../result', directory, 'sequence')

        folder_name = os.path.basename(directory)

        output_file = f'features/CKSAAP/{folder_name}.txt'

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        all_cksaap_vectors = []
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        files_sorted = sorted(files, key=numeric_sort)

        for file in files_sorted:
            file_path = os.path.join(directory_path, file)

            print(f"Processing file: {file_path}")

            sequence = read_fasta(file_path)
            cksaap_vector = calculate_cksaap(sequence, gap)
            if cksaap_vector is not None:
                all_cksaap_vectors.append(cksaap_vector)

        with open(output_file, 'w') as out_file:
            for cksaap_vector in all_cksaap_vectors:
                out_file.write(','.join(map(str, cksaap_vector)) + '\n')


base_directories = [
    'PeNGaRoo_independent_test_N',
    'PeNGaRoo_independent_test_P',
    'PeNGaRoo_train_N',
    'PeNGaRoo_train_P'
]

process_all_directories(base_directories, gap=3)
