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


def calculate_ctdt(sequence):
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

    sequence = re.sub('-', '', sequence)
    sequence = sequence.upper()
    sequence_length = len(sequence)

    if sequence_length < 2:
        return None

    ctdt_vector = []

    for prop in properties:
        g1 = group1[prop]
        g2 = group2[prop]
        g3 = group3[prop]

        c1221 = 0
        c1331 = 0
        c2332 = 0

        aa_pairs = [sequence[i:i+2] for i in range(sequence_length - 1)]

        for pair in aa_pairs:
            aa1, aa2 = pair[0], pair[1]

            if (aa1 in g1 and aa2 in g2) or (aa1 in g2 and aa2 in g1):
                c1221 += 1
            elif (aa1 in g1 and aa2 in g3) or (aa1 in g3 and aa2 in g1):
                c1331 += 1
            elif (aa1 in g2 and aa2 in g3) or (aa1 in g3 and aa2 in g2):
                c2332 += 1

        total_pairs = len(aa_pairs)

        freq_c1221 = c1221 / total_pairs if total_pairs > 0 else 0
        freq_c1331 = c1331 / total_pairs if total_pairs > 0 else 0
        freq_c2332 = c2332 / total_pairs if total_pairs > 0 else 0

        ctdt_vector.extend([freq_c1221, freq_c1331, freq_c2332])

    return ctdt_vector


def numeric_sort(file_name):
    return int(file_name)


def process_all_directories_for_ctdt(base_directories):
    for directory in base_directories:
        directory_path = os.path.join('../result', directory, 'sequence')

        if not os.path.exists(directory_path):
            continue

        folder_name = os.path.basename(directory)

        output_file = f'features/CTDT/{folder_name}.txt'

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        all_ctdt_vectors = []

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

            ctdt_vector = calculate_ctdt(sequence)
            if ctdt_vector is not None:
                all_ctdt_vectors.append(ctdt_vector)

        if not all_ctdt_vectors:
            continue

        with open(output_file, 'w') as out_file:
            for ctdt_vector in all_ctdt_vectors:
                out_file.write(','.join(map(str, ctdt_vector)) + '\n')


base_directories = [
    'PeNGaRoo_independent_test_N',
    'PeNGaRoo_independent_test_P',
    'PeNGaRoo_train_N',
    'PeNGaRoo_train_P'
]

process_all_directories_for_ctdt(base_directories)
