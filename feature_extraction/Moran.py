import os
import re
import sys
import platform
import numpy as np

def read_fasta(file_path):
    sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith('>'):
                sequence += line
    return sequence

def load_AAidx(file_path):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    with open(file_path) as f:
        records = f.readlines()
    print(f'Total AAindex read: {len(records) - 1}')
    myDict = {}
    props = []
    for i in records[1:]:
        array = i.rstrip().split('\t')
        if 'NA' in array:
            continue
        myDict[array[0]] = array[1:]
        props.append(array[0])
    print(f'Total valid AAindex: {len(props)}')
    AAidx = []
    for i in props:
        AAidx.append([float(x) for x in myDict[i]])

    AAidx = np.array(AAidx)
    # 标准化
    propMean = np.mean(AAidx, axis=1)
    propStd = np.std(AAidx, axis=1)
    AAidx = (AAidx - propMean[:, None]) / propStd[:, None]

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    return props, AAidx, index

def calculate_moran(sequence, nlag, props, AAidx, index):
    sequence = re.sub('-', '', sequence.upper())
    N = len(sequence)
    if N < nlag + 1:
        return None

    code = []
    for prop in range(len(props)):
        xmean = np.mean([AAidx[prop][index.get(aa, 0)] for aa in sequence])
        fenmu = np.sum([(AAidx[prop][index.get(aa, 0)] - xmean) ** 2 for aa in sequence]) / N
        for n in range(1, nlag + 1):
            if N > nlag:
                fenzi = np.sum([(AAidx[prop][index.get(sequence[j], 0)] - xmean) *
                                (AAidx[prop][index.get(sequence[j + n], 0)] - xmean)
                                for j in range(N - n)]) / (N - n)
                rn = fenzi / fenmu if fenmu != 0 else 0
            else:
                rn = 0
            code.append(rn)
    return code


def numeric_sort(file_name):
    return int(file_name)

def process_all_directories_for_moran(base_directories, nlag=2):
    AAidx_file = os.path.join('../features', 'AAidx.txt')
    if not os.path.exists(AAidx_file):
        return

    props, AAidx, index = load_AAidx(AAidx_file)

    for directory in base_directories:
        directory_path = os.path.join('../result', directory, 'sequence')

        folder_name = os.path.basename(directory)

        output_file = os.path.join('../features', 'Moran', f'{folder_name}.txt')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        moran_vectors = []

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

            moran_vector = calculate_moran(sequence, nlag, props, AAidx, index)
            if moran_vector is not None:
                moran_vectors.append(moran_vector)

        if not moran_vectors:
            continue

        with open(output_file, 'w') as out_file:
            for moran_vector in moran_vectors:
                out_file.write(','.join(map(str, moran_vector)) + '\n')


base_directories = [
    'PeNGaRoo_independent_test_N',
    'PeNGaRoo_independent_test_P',
    'PeNGaRoo_train_N',
    'PeNGaRoo_train_P'
]

process_all_directories_for_moran(base_directories)
