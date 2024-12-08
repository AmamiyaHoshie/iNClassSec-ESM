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


def calculate_ctriad(sequence):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())  # ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7']

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.' + f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]

    sequence = re.sub('-', '', sequence.upper())

    sequence_length = len(sequence)
    if sequence_length < 3:
        return None

    myDict = {}
    for f in features:
        myDict[f] = 0

    for i in range(sequence_length - 2):
        aa1 = sequence[i]
        aa2 = sequence[i + 1]
        aa3 = sequence[i + 2]
        if aa1 not in AADict or aa2 not in AADict or aa3 not in AADict:
            continue
        key = AADict[aa1] + '.' + AADict[aa2] + '.' + AADict[aa3]
        myDict[key] += 1

    total = sum(myDict.values())
    if total == 0:
        return [0] * len(features)

    ctriad_vector = [myDict[f] / total for f in features]

    return ctriad_vector


def numeric_sort(file_name):
    return int(file_name)


def process_all_directories_for_ctriad(base_directories):
    for directory in base_directories:
        directory_path = os.path.join('../result', directory, 'sequence')

        folder_name = os.path.basename(directory)

        output_file = os.path.join('../features', 'CTriad', f'{folder_name}.txt')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        ctriad_vectors = []

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

            ctriad_vector = calculate_ctriad(sequence)
            if ctriad_vector is not None:
                ctriad_vectors.append(ctriad_vector)

        if not ctriad_vectors:
            continue

        with open(output_file, 'w') as out_file:
            for ctriad_vector in ctriad_vectors:
                out_file.write(','.join(map(str, ctriad_vector)) + '\n')


base_directories = [
    'PeNGaRoo_independent_test_N',
    'PeNGaRoo_independent_test_P',
    'PeNGaRoo_train_N',
    'PeNGaRoo_train_P'
]

process_all_directories_for_ctriad(base_directories)
