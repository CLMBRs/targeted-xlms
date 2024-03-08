import os
import sys

from pathlib import Path

vocab_size_shorthand = {
    '16k': 16386,
    '32k': 32770,
    '64k': 65538,
    '128k': 131074,
    'full': 'NA'
}

log_file_prefix = sys.argv[1]
output_file_stem = sys.argv[2]

log_file_prefix = Path(log_file_prefix)
log_file_prefix_stem = log_file_prefix.stem
directory = log_file_prefix.parent

input_files = [
    filename for filename in os.listdir(directory) if filename.startswith(log_file_prefix_stem) and filename.endswith('.out')
]

output_csv_lines = []
output_file = os.path.join(directory, output_file_stem + '.csv')

for input_file in input_files:
    # Some of the relevant info for these eval logs is encoded only in the file names, so we extract
    # that here. Pretty hacky, sorry
    input_file_path = Path(os.path.join(directory, input_file))
    file_stem = input_file_path.stem
    file_info = file_stem.split('_')

    eval_task = eval_setting = vocab_alpha = lapt_alpha = vocab_size = lapt_steps = None

    eval_task = file_info[1]
    eval_setting = file_info[2]
    if file_info[3].endswith('ots'):
        vocab_alpha = 'NA'
        lapt_alpha = 'NA'
        vocab_size = 'NA'
        lapt_steps = 0
    else:
        if file_info[4].startswith('voc'):
            vocab_alpha = file_info[4][3:]
            vocab_alpha = vocab_alpha[0] + '.' + vocab_alpha[1:]
            del file_info[4]
        elif file_info[5] == 'full-voc':
            vocab_alpha = 'NA'
            vocab_size = 0
        else:
            vocab_alpha = 0.2
        lapt_alpha = file_info[4][5:]
        lapt_alpha = lapt_alpha[0] + '.' + lapt_alpha[1:]
        if vocab_size != 0:
            vocab_size = vocab_size_shorthand[file_info[5][3:]]
        lapt_steps = file_info[-1].replace('k', '000') if len(file_info) > 6 else '100000'

    log_lines = [line.strip() for line in open(input_file_path, 'r') if line != '']

    language = random_seed = metric = None
    
    # This block actually extracts information from the content of the log file
    for line in log_lines:
        if line.startswith('Beginning'):
            language = line.split(' ')[-1]
        if line.startswith('random'):
            random_seed = line.split(' ')[-1]
        if line.startswith('accuracy'):
            metric = line.split(' ')[-1]
            output_csv_lines.append(
                f'{eval_task}, {eval_setting}, {vocab_alpha}, {lapt_alpha}, {vocab_size}, {lapt_steps}, {language}, {random_seed}, {metric}'
            )

with open(output_file, 'w') as fout:
    print('task, setting, vocab_alpha, lapt_alpha, vocab_size, lapt_steps, language, seed, accuracy', file=fout)

with open(output_file, 'a') as fout:
    for line in output_csv_lines:
        print(line, file=fout)
