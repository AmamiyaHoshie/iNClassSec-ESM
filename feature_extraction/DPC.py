import os
import re


def read_fasta(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    return sequence


def calculate_dpc(sequence, aa_order='ACDEFGHIKLMNPQRSTVWY'):
    sequence = re.sub('-', '', sequence)

    # Skip if sequence length is less than 2
    if len(sequence) < 2:
        return None

    # All possible dipeptides
    dipeptides = [aa1 + aa2 for aa1 in aa_order for aa2 in aa_order]

    # Initialize counts for each dipeptide
    dipeptide_count = {dipeptide: 0 for dipeptide in dipeptides}

    # Count dipeptides in the sequence
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i] + sequence[i + 1]
        if dipeptide in dipeptide_count:
            dipeptide_count[dipeptide] += 1

    # Normalize counts to get the DPC feature vector
    total_dipeptides = sum(dipeptide_count.values())
    dpc_vector = [dipeptide_count[dipeptide] / total_dipeptides if total_dipeptides > 0 else 0 for dipeptide in
                  dipeptides]

    # Check if the DPC vector is all zeros and skip if true
    if all(value == 0 for value in dpc_vector):
        return None

    return dpc_vector

# 自定义排序函数，按文件名中的数字升序排序
def numeric_sort(file_name):
    """
    直接将文件名转换为整数，用于按数字排序
    """
    return int(file_name)  # 文件名即为数字，直接转换为整数

def process_all_directories_for_dpc(base_directories):
    for directory in base_directories:
        # Define the path to the 'sequence' directory
        directory_path = os.path.join('../result', directory, 'sequence')

        # Get the folder name to use in output file names
        folder_name = os.path.basename(directory)

        # Define output file path
        output_file = f'features/DPC/{folder_name}.txt'

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        all_dpc_vectors = []
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        # 按文件名（数字）进行升序排序
        files_sorted = sorted(files, key=numeric_sort)

        for file in files_sorted:
            file_path = os.path.join(directory_path, file)

            # 输出正在处理的文件名
            print(f"Processing file: {file_path}")

            sequence = read_fasta(file_path)
            dpc_vector = calculate_dpc(sequence)
            # Skip sequences with an invalid or all-zero DPC vector
            if dpc_vector is not None:
                all_dpc_vectors.append(dpc_vector)

        # Write DPC vectors to the output file
        with open(output_file, 'w') as out_file:
            for dpc_vector in all_dpc_vectors:
                out_file.write(','.join(map(str, dpc_vector)) + '\n')


# List of base directories to process
base_directories = [
    'PeNGaRoo_independent_test_N',
    'PeNGaRoo_independent_test_P',
    'PeNGaRoo_train_N',
    'PeNGaRoo_train_P'
]

# Process all directories for DPC
process_all_directories_for_dpc(base_directories)
