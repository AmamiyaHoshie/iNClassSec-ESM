import os
from collections import Counter
import re


def read_fasta(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    return sequence


def calculate_aac(sequence, aa_order='ACDEFGHIKLMNPQRSTVWY'):
    sequence = re.sub('-', '', sequence)
    sequence_length = len(sequence)

    if sequence_length == 0:
        return None

    aa_count = Counter(sequence)
    aac_vector = [aa_count[aa] / sequence_length for aa in aa_order]

    if all(value == 0 for value in aac_vector):
        return None

    return aac_vector


def numeric_sort(file_name):

    return int(file_name)


def process_all_files(directory_path):

    folder_name = directory_path.split('/')[1]

    output_file = f'features/AAC/{folder_name}.txt'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    all_aac_vectors = []
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    files_sorted = sorted(files, key=numeric_sort)

    for file in files_sorted:
        file_path = os.path.join(directory_path, file)

        print(f"Processing file: {file_path}")

        sequence = read_fasta(file_path)
        aac_vector = calculate_aac(sequence)

        if aac_vector is not None:
            all_aac_vectors.append(aac_vector)

    with open(output_file, 'w') as out_file:
        for aac_vector in all_aac_vectors:
            out_file.write(','.join(map(str, aac_vector)) + '\n')


base_directories = [
    'result/PeNGaRoo_independent_test_N/sequence/',
    'result/PeNGaRoo_independent_test_P/sequence/',
    'result/PeNGaRoo_train_N/sequence/',
    'result/PeNGaRoo_train_P/sequence/'
]

for directory_path in base_directories:
    process_all_files(directory_path)
