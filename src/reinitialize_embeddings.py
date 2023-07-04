import argparse
import copy
from collections import defaultdict

import torch
from transformers import AutoTokenizer, XLMRobertaTokenizer, AutoModelForMaskedLM

from utils import _xlmr_special_tokens


def hex2dec(hex_str):
    """Convert Unicode hexadecimal string to base-10 int."""
    return int(hex_str, 16)


def get_ord2script(scriptfile):
    """Return dictionary (key: Unicode decimal, val: script of corresponding
    character according to Unicode documentation)"""
    with open(scriptfile, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
    ord2script = dict()
    for line in lines:
        if line[0] != '#':
            items = line.split()
            if len(items) > 0:
                script = items[2]
                encoding = items[0]
                if '..' in encoding:
                    start_stop = encoding.split('..')
                    start = hex2dec(start_stop[0])
                    stop = hex2dec(start_stop[1])
                    for dec_encoding in range(start, stop + 1):
                        ord2script[dec_encoding] = script
                else:
                    dec_encoding = hex2dec(encoding)
                    ord2script[dec_encoding] = script

    return ord2script


def top_script(token, ord2script):
    """Return most-used script within token (str), using ord2script (dict)
    to retrieve the script of each char in token."""
    script_counts = defaultdict(lambda: 0)

    for character in token:
        try:
            script = ord2script[ord(character)]
        except KeyError:
            script = 'UNK'
        script_counts[script] += 1

    return max(script_counts, key=lambda x: script_counts[x])


def get_script_to_ids(vocab: dict[str, int], ord_to_script: dict[int, str]) -> dict[str, list[int]]:
    # get script for each token in XLM-R's vocab
    script_to_ids = defaultdict(list)
    for token, index in vocab.items():
        if token in _xlmr_special_tokens:
            script_to_ids['xlmr_special'].append(index)
        # leave out the preceding whitespace when identifying token script
        token_text = token[1:] if token[0] == '▁' and len(token) > 1 else token
        # identify top script for the token based on characters and Unicode mapping
        script = top_script(token_text, ord_to_script)
        script_to_ids[script].append(index)
    return script_to_ids


def initialize_by_category_means(
    categories: list[str],
    means: torch.Tensor,
    stdevs: torch.Tensor,
    category_to_indices: dict[str, int],
    matrix: torch.Tensor,
    categories_to_omit: list[str] = ['xlmr_special']
) -> torch.Tensor:
    # only initialize the categories that are in both the old and new data
    category_intersection = set(categories).intersection(set(category_to_indices.keys()))
    for category in category_intersection:
        if category in categories_to_omit:
            continue
        category_index = categories.index(category)
        category_distribution = torch.distributions.Normal(
            means[category_index], stdevs[category_index]
        )
        for index in category_to_indices[category]:
            matrix[index] = category_distribution.sample()

    return matrix


def reinitialize_by_script(
    old_vocab, old_embeddings, new_vocab, new_embeddings, unicode_table_path
):
    # get dictionary to map Unicode decimal to script
    ord_to_script = get_ord2script(unicode_table_path)

    old_script_to_ids = get_script_to_ids(old_vocab, ord_to_script)
    new_script_to_ids = get_script_to_ids(new_vocab, ord_to_script)

    all_old_scripts = list(old_script_to_ids.keys())

    # get mean and standard deviation of embeddings for each script
    old_script_stdevs = []
    old_script_means = []
    for script in all_old_scripts:
        script_embed_list = [old_embeddings[x] for x in old_script_to_ids[script]]
        script_embeddings = torch.stack(script_embed_list, dim=0)
        std_and_mean = torch.std_mean(script_embeddings, dim=0)
        old_script_stdevs.append(std_and_mean[0])
        old_script_means.append(std_and_mean[1])
    old_script_stdevs = torch.stack(old_script_stdevs, dim=0)
    old_script_means = torch.stack(old_script_means, dim=0)

    new_embeddings = initialize_by_category_means(
        all_old_scripts, old_script_means, old_script_stdevs, new_script_to_ids, new_embeddings
    )
    return new_embeddings

def reinitialize_by_identity(old_vocab, old_embeddings, new_vocab, new_embeddings, tokens_to_ignore):
    identical_tokens = []
    for token, new_index in new_vocab.items():
        if token in old_vocab and token not in tokens_to_ignore:
            new_embeddings[new_index] = old_embeddings[old_vocab[token]]
            identical_tokens.append(token)

    return new_embeddings, identical_tokens


if __name__ == '__main__':
    # read training configurations from YAML file
    parser = argparse.ArgumentParser(
        description="Initialize new embeddings for an XLM-R based model"
    )
    parser.add_argument('--old_model_path', type=str, required=True)
    parser.add_argument('--old_tokenizer_path', type=str, default=None)
    parser.add_argument('--new_vocab_file', type=str, required=True)
    parser.add_argument('--embedding_output_path', type=str, required=True)
    parser.add_argument('--reinit_by_script', action='store_true')
    parser.add_argument('--unicode_block_table', type=str, default="tools/unicode_scripts_for_embeddings_exploration.txt")
    parser.add_argument('--reinit_by_identity', action='store_true')
    args = parser.parse_args()

    # load pretrained model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(args.old_model_path)
    embedding_size = model.config.hidden_size

    # get the vocab and embeddings from base XLM-R model
    if not args.old_tokenizer_path:
        args.old_tokenizer_path = args.old_model_path
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_tokenizer_path)
    old_vocab = old_tokenizer.get_vocab()
    old_embeddings = copy.deepcopy(model.get_input_embeddings().weight).detach()
    del model

    # read in the new tokenizer, initialize new embeddings
    new_tokenizer = XLMRobertaTokenizer(vocab_file=args.new_vocab_file)
    new_vocab = new_tokenizer.get_vocab()
    new_vocab_size = new_tokenizer.vocab_size
    new_embeddings = torch.nn.Embedding(new_vocab_size, embedding_size).weight.detach()

    # set the embeddings for special tokens to be identical to XLM-R (invariant for all
    # embedding reinitialization techniques, since these tokens are likely critical for the model
    # to function properly)
    for special_token in _xlmr_special_tokens:
        old_token_index = old_vocab[special_token]
        new_token_index = new_vocab[special_token]
        new_embeddings[new_token_index] = old_embeddings[old_token_index]

    if args.reinit_by_script:
        new_embeddings = reinitialize_by_script(
            old_vocab, old_embeddings, new_vocab, new_embeddings, args.unicode_block_table
        )

    if args.reinit_by_identity:
        new_embeddings, identical_tokens = reinitialize_by_identity(old_vocab, old_embeddings, new_vocab, new_embeddings, tokens_to_ignore=_xlmr_special_tokens)

    torch.save(new_embeddings, args.embedding_output_path)