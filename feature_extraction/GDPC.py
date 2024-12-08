import os
import re


def read_fasta(file_path):
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith('>'):
                sequence += line
    return sequence


def calculate_gdpc(sequence):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    group_keys = list(group.keys())
    dipeptides = [g1 + '.' + g2 for g1 in group_keys for g2 in group_keys]

    index = {}
    for key in group_keys:
        for aa in group[key]:
            index[aa] = key

    sequence = re.sub('-', '', sequence.upper())
    sequence_length = len(sequence)
    if sequence_length < 2:
        return None

    myDict = {dipeptide: 0 for dipeptide in dipeptides}
    total = 0

    for j in range(sequence_length - 1):
        aa1 = sequence[j]
        aa2 = sequence[j + 1]
        if aa1 in index and aa2 in index:
            dipeptide = index[aa1] + '.' + index[aa2]
            myDict[dipeptide] += 1
            total += 1
        else:
            continue

    if total == 0:
        gdpc_vector = [0] * len(dipeptides)
    else:
        gdpc_vector = [myDict[dipeptide] / total for dipeptide in dipeptides]

    return gdpc_vector


def numeric_sort(file_name):
    return int(file_name)


def process_all_directories_for_gdpc(base_directories):
    for directory in base_directories:
        directory_path = os.path.join('../result', directory, 'sequence')

        folder_name = os.path.basename(directory)

        output_file = os.path.join('../features', 'GDPC', f'{folder_name}.txt')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        gdpc_vectors = []

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

            gdpc_vector = calculate_gdpc(sequence)
            if gdpc_vector is not None:
                gdpc_vectors.append(gdpc_vector)

        if not gdpc_vectors:
            continue

        with open(output_file, 'w') as out_file:
            for gdpc_vector in gdpc_vectors:
                out_file.write(','.join(map(str, gdpc_vector)) + '\n')


base_directories = [
    'PeNGaRoo_independent_test_N',
    'PeNGaRoo_independent_test_P',
    'PeNGaRoo_train_N',
    'PeNGaRoo_train_P'
]

process_all_directories_for_gdpc(base_directories)
