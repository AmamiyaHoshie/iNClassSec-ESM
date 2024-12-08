import os
import re
import math


def read_fasta(file_path):
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith('>'):
                sequence += line
    return sequence


def count_distribution(aa_set, sequence):
    positions = [i+1 for i, aa in enumerate(sequence) if aa in aa_set]
    number = len(positions)
    if number == 0:
        return [0, 0, 0, 0, 0]
    cutoff_nums = [1, math.ceil(0.25 * number), math.ceil(0.5 * number), math.ceil(0.75 * number), number]
    distribution = []
    for cutoff in cutoff_nums:
        index = positions[cutoff - 1]
        distribution.append(index / len(sequence) * 100)
    return distribution


def calculate_ctdd(sequence):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
    }

    properties = [
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101', 'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101',
        'hydrophobicity_FASG890101', 'normwaalsvolume', 'polarity', 'polarizability', 'charge',
        'secondarystruct', 'solventaccess'
    ]

    sequence = re.sub('-', '', sequence.upper())
    sequence_length = len(sequence)

    if sequence_length == 0:
        return None

    ctdd_vector = []

    for prop in properties:
        ctdd_vector.extend(count_distribution(group1[prop], sequence))
        ctdd_vector.extend(count_distribution(group2[prop], sequence))
        ctdd_vector.extend(count_distribution(group3[prop], sequence))

    return ctdd_vector


def numeric_sort(file_name):

    return int(file_name)


def process_all_directories_for_ctdd(base_directories):
    for directory in base_directories:
        directory_path = os.path.join('../result', directory, 'sequence')

        folder_name = os.path.basename(directory)

        output_file = os.path.join('../features', 'CTDD', f'{folder_name}.txt')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        ctdd_vectors = []

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

            ctdd_vector = calculate_ctdd(sequence)
            if ctdd_vector is not None:
                ctdd_vectors.append(ctdd_vector)

        if not ctdd_vectors:
            continue

        with open(output_file, 'w') as out_file:
            for ctdd_vector in ctdd_vectors:
                out_file.write(','.join(map(str, ctdd_vector)) + '\n')


base_directories = [
    'PeNGaRoo_independent_test_N',
    'PeNGaRoo_independent_test_P',
    'PeNGaRoo_train_N',
    'PeNGaRoo_train_P'
]

process_all_directories_for_ctdd(base_directories)
