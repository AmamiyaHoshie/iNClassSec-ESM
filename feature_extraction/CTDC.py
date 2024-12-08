import os
import re

def read_fasta(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    return sequence


def calculate_ctdc(sequence):
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

    properties = (
        'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
        'hydrophobicity_PONP930101', 'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101',
        'hydrophobicity_FASG890101', 'normwaalsvolume', 'polarity', 'polarizability', 'charge',
        'secondarystruct', 'solventaccess'
    )

    sequence = re.sub('-', '', sequence)
    sequence_length = len(sequence)

    if sequence_length == 0:
        return None

    ctdc_vector = []

    for prop in properties:
        c1 = sum(sequence.count(aa) for aa in group1[prop]) / sequence_length
        c2 = sum(sequence.count(aa) for aa in group2[prop]) / sequence_length
        c3 = sum(sequence.count(aa) for aa in group3[prop]) / sequence_length
        ctdc_vector.extend([c1, c2, c3])

    return ctdc_vector


def numeric_sort(file_name):

    return int(file_name)


def process_all_directories_for_ctdc(base_directories):
    for directory in base_directories:

        directory_path = os.path.join('../result', directory, 'sequence')

        folder_name = os.path.basename(directory)

        output_file = f'features/CTDC/{folder_name}.txt'

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        all_ctdc_vectors = []
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        files_sorted = sorted(files, key=numeric_sort)

        for file in files_sorted:
            file_path = os.path.join(directory_path, file)

            print(f"Processing file: {file_path}")

            sequence = read_fasta(file_path)
            ctdc_vector = calculate_ctdc(sequence)
            if ctdc_vector is not None:
                all_ctdc_vectors.append(ctdc_vector)

        with open(output_file, 'w') as out_file:
            for ctdc_vector in all_ctdc_vectors:
                out_file.write(','.join(map(str, ctdc_vector)) + '\n')



base_directories = [
    'PeNGaRoo_independent_test_N',
    'PeNGaRoo_independent_test_P',
    'PeNGaRoo_train_N',
    'PeNGaRoo_train_P'
]

process_all_directories_for_ctdc(base_directories)
