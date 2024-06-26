import argparse
import copy
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from tqdm import tqdm
from seqeval.metrics import f1_score

from utils import (
    MODEL_CLASSES, _consolidate_features, _label_spaces, _lang_choices, _model_names, _task_choices,
    load_hf_model, load_ud_splits, load_ner_splits
)

_loader_functions_by_task = {
    'pos': load_ud_splits,
    'ner': load_ner_splits,
    'uas': load_ud_splits
}


class Tagger(torch.nn.Module):
    """
    Tagger class for conducting classification with a pre-trained model. Takes in the pre-trained
    model as the `encoder` argument on initialization. The tagger/classification-head itself
    consists of a single linear layer + softmax
    """
    def __init__(self, encoder, output_dim):
        super(Tagger, self).__init__()

        self.encoder = copy.deepcopy(encoder)
        input_dim = self.encoder.encoder.layer[-1].output.dense.out_features

        self.proj = torch.nn.Linear(input_dim, output_dim)

    def forward(self, input_ids, alignments):
        # run through encoder
        output = self.encoder(input_ids).last_hidden_state

        # get single rep for each word
        feats = []
        for i in range(output.shape[0]):
            feats.extend(_consolidate_features(output[i], alignments[i]))
        output = torch.cat(feats, dim=0)

        output = self.proj(output)
        output = F.softmax(output, dim=-1)
        return output
    
class DependencyTagger(torch.nn.Module):
    """
    Dependency Tagger class for conducting classification with a pre-trained model. Takes in the
    pre-trained model as the `encoder` argument on initialization. The encoder argument is then
    represented as one head and one dependent using a Linear layer. The tagger/classification-head
    itself consists of one weight and one bias layer based on cite:`Dozat and Manning (2017)`
    """
    def __init__(self, encoder, output_dim):
        super(DependencyTagger, self).__init__()

        self.encoder = copy.deepcopy(encoder)
        input_dim = self.encoder.encoder.layer[-1].output.dense.out_features # (seq len)
        self.head_layer = torch.nn.Linear(input_dim, output_dim)
        self.dep_layer = torch.nn.Linear(input_dim, output_dim)
        self.linear_weight = torch.nn.Linear(output_dim, output_dim, bias = False)
        self.linear_bias = torch.nn.Linear(output_dim, 1, bias = False)

    def forward(self, input_ids, alignments):
        # run through encoder
        embed = self.encoder(input_ids).last_hidden_state
        # get single rep for each word
        feats = []
        for i in range(embed.shape[0]):
            feat = torch.stack(_consolidate_features(embed[i], alignments[i]),dim=1) #(1,seq len,hidden_dim)
            feats.append(feat) # list of (1, seq len, hidden_dim)
        
        outputs = []
        for feat in feats: 
            feat = feat.squeeze(dim=0) # (seq len, hidden_dim)
            head_embed = self.head_layer(feat) # (seq len,hidden_dim) & (hidden_dim, output)->(seq,output)
            dep_embed = self.dep_layer(feat) # (seq len,output_dim)

            partial_weight_mul = self.linear_weight(head_embed) # (seq len, output)
            weight = torch.matmul(dep_embed,partial_weight_mul.transpose(0,1)) # (seq len, seq len)
            bias = self.linear_bias(head_embed) # (seq len, 1)
            biaffine_score = weight + bias # (seq len,seq len)
            output = F.softmax(biaffine_score, dim=-1) #(seq len,seq len)
            outputs.append(output)
        return outputs

def whitespace_to_sentencepiece(tokenizer, dataset, label_space, max_seq_length=512, layer_id=-1):
    """
    Pre-process the text data from a UD dataset using a huggingface tokenizer, keeping track of the
    mapping between word and subword tokenizations. Break up sentences in the dataset that are
    longer than 511 tokens
    """
    processed_dataset = []

    num_subwords = 0.
    num_words = 0.
    for sentence, labels in tqdm(dataset):
        # tokenize manually word by word because huggingface can't get word
        # alignments to track correctly
        # https://github.com/huggingface/transformers/issues/9637
        tokens = []
        if tokenizer.cls_token_id != None:
            tokens = [tokenizer.cls_token_id]
        alignment = []
        label_subset = []
        alignment_id = 1  #offset for cls token
        for word, label in zip(sentence, labels):
            word_ids = tokenizer.encode(' ' + word, add_special_tokens=False)
            if len(word_ids) < 1:
                continue
            if len(tokens) + len(word_ids) > (max_seq_length - 1):
                # add example to dataset
                if tokenizer.cls_token_id != None:
                    tokens += [tokenizer.sep_token_id]
                else:
                    tokens += [tokenizer.eos_token_id]

                input_ids = torch.LongTensor(tokens)
                labels = [
                    torch.LongTensor([label_space.index(l)])
                    if l in label_space else torch.LongTensor([-100]) for l in label_subset
                ]
                processed_dataset.append((input_ids, alignment, labels))

                # reset
                tokens = []
                if tokenizer.cls_token_id != None:
                    tokens = [tokenizer.cls_token_id]
                alignment = []
                label_subset = []
                alignment_id = 1  #offset for cls token

            tokens.extend(word_ids)
            word_alignments = [x for x in range(alignment_id, alignment_id + len(word_ids))]
            alignment_id = alignment_id + len(word_ids)
            alignment.append(word_alignments)
            label_subset.append(label)

            num_subwords += len(word_ids)
            num_words += 1

        if tokenizer.cls_token_id != None:
            tokens += [tokenizer.sep_token_id]
        else:
            tokens += [tokenizer.eos_token_id]

        input_ids = torch.LongTensor(tokens)
        # process labels into tensor
        # labels = [torch.LongTensor([label_space.index(l)]) for l in labels]
        # filtering out "unlabeled" examples with "_"
        labels = [
            torch.LongTensor([label_space.index(l)])
            if l in label_space else torch.LongTensor([-100]) for l in label_subset
        ]
        processed_dataset.append((input_ids, alignment, labels))

    # 1.183 for en train
    # print('Avg. number of subword pieces per word = {}'.format(num_subwords / num_words))
    return processed_dataset


def _pad_to_len(seq, length, pad_idx):
    s_len = seq.shape[-1]
    if s_len < length:
        pad_tensor = torch.LongTensor([pad_idx] * (length - s_len))
        seq = torch.cat([seq, pad_tensor], dim=-1)
    return seq


def batchify(data, pad_idx):
    input_ids, alignments, labels = zip(*data)

    max_length = max([x.shape[-1] for x in input_ids])
    input_ids = torch.stack([_pad_to_len(x, max_length, pad_idx) for x in input_ids], dim=0)

    flat_labels = []
    for l in labels:
        flat_labels.extend(l)
    labels = torch.cat(flat_labels, dim=0)

    return input_ids, alignments, labels


def train_model(
    model,
    data,
    valid_data,
    criterion,
    optimizer,
    pad_idx,
    eval_metric,
    bsz=1,
    gradient_accumulation=1,
    max_grad_norm=1.0,
    epochs=1,
    max_train_examples=math.inf,
    eval_every=2,
    patience=0,
    model_dir='./eval_checkpoints',
    task='pos'
):
    model.train()
    model.to('cuda:0')

    best_model_path = os.path.join(model_dir, "best_model.pt")
    latest_model_path = os.path.join(model_dir, "latest_model.pt")
    checkpoint_control_path = os.path.join(model_dir, "checkpoint_control.yml")

    max_valid_acc = 0
    patience_step = 0
    accumulation_counter = 0

    total_examples = min(len(data), max_train_examples)
    if total_examples < len(data):
        random.shuffle(data)
        data = data[:total_examples]

    checkpoint_control_exists = os.path.isfile(checkpoint_control_path)
    if checkpoint_control_exists:
        with open(checkpoint_control_path, 'r') as fin:
            control_dict = yaml.load(fin, Loader=yaml.Loader)
            resume_from_epoch = control_dict['resume_from_epoch']
        if resume_from_epoch and not control_dict['training_complete']:
            model.load_state_dict(torch.load(latest_model_path))
            print(f"Resuming training from epoch {resume_from_epoch}", file=sys.stderr)
    else:
        resume_from_epoch = None

    # training epochs
    for epoch_id in tqdm(range(0, epochs), desc='training loop', total=epochs, unit='epoch'):
        random.shuffle(data)
        # fast forward to the latest checkpointed epoch, if any
        if resume_from_epoch and ((epoch_id + 1) <= resume_from_epoch):
            continue

        # TODO: make this and eval work with new data format (and add feature
        # handling to forward()) go over training data
        for i in range(0, total_examples, bsz):
            if i + bsz > len(data):
                batch_data = data[i:]
            else:
                batch_data = data[i:i + bsz]
            input_ids, alignments, labels = batchify(batch_data, pad_idx)

            input_ids = input_ids.to('cuda:0')
            labels = labels.to('cuda:0')

            output = model.forward(input_ids, alignments)
            if task == 'ner':
                # This is hard-coding the number of classes as well as the fact
                # that the O class has index 0. Needs to be changed if these
                # assumptions don't hold
                total_labels = labels.numel()
                num_BI_labels = torch.count_nonzero(labels)
                num_O_labels = total_labels - num_BI_labels
                if num_BI_labels > 0 and num_O_labels > num_BI_labels:
                    O_lambda = num_BI_labels / num_O_labels
                else:
                    O_lambda = 1.0
                class_weights = torch.ones(7, device='cuda:0')
                class_weights[0] = O_lambda
                loss = torch.nn.functional.cross_entropy(output, labels, weight=class_weights)
            elif task == "uas":
                # Initialize list of losses and weights for each example of loss (seq len)
                losses = []
                weights = []
                # offset to map output examples to labels
                offset=0 

                # Iterate over the list of tensors and calculate the loss for each pair
                for i in range(len(output)):
                    # output[i] dim: (seq_len, seq_len)
                    # Assuming output[i] and labels[offset:offset+example_len] are the tensors for 
                    # the i-th example
                    example_len = output[i].size(dim=-1)
                    loss = torch.nn.functional.cross_entropy(
                        output[i], labels[offset:offset+example_len], reduction="sum"
                    )

                    #track batch metrics
                    losses.append(loss)
                    weights.append(example_len)
                    offset += example_len

                # Calculate the weighted average loss
                weight_sum = sum(weights)
                weights = [w/weight_sum for w in weights]
                # average loss of all elements in a batch
                loss = sum([w*l for w,l in zip(weights, losses)]) / len(losses) 
                
            else: # for task: pos
                loss = criterion(output, labels)
            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss detected: epoch {epoch_id + 1}")
            loss.backward()
            accumulation_counter += 1
            if accumulation_counter >= gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                accumulation_counter = 0

        if ((epoch_id + 1) % eval_every) == 0:
            valid_acc = evaluate_model(model, valid_data, pad_idx, bsz=bsz, metric=eval_metric)
            model.train()
            torch.save(model.state_dict(), latest_model_path)
            with open(checkpoint_control_path, 'w') as fout:
                print("training_complete: False", file=fout)
                print("num_epochs_to_convergence: null", file=fout)
                print(f"resume_from_epoch: {epoch_id + 1}", file=fout)

            # use train loss to see if we need to stop
            if max_valid_acc > valid_acc:
                if patience_step == patience:
                    break
                else:
                    patience_step += 1
            else:
                max_valid_acc = valid_acc
                # checkpoint model
                torch.save(model.state_dict(), best_model_path)
                patience_step = 0

    # output a marker that training is completed, as well as the number of epochs to convergence;
    # this is so trials need not be repeated if a job is preempted
    num_epochs_to_convergence = (epoch_id + 1 - (patience_step * eval_every))
    with open(checkpoint_control_path, 'w') as fout:
        print("training_complete: True", file=fout)
        print(f"num_epochs_to_convergence: {num_epochs_to_convergence}", file=fout)
        print("resume_from_epoch: null", file=fout)

    # Remove the "latest model", if it exists (only keep best)
    try:
        os.remove(latest_model_path)
    except:
        pass

    # return model, number of training epochs for best ckpt
    return model, num_epochs_to_convergence


def evaluate_model(model, data, pad_idx, bsz=1, metric='acc'):
    model.eval()
    model.to('cuda:0')

    if metric == 'acc':
        num_correct = 0.
        total_num = 0.

        for i in range(0, len(data), bsz):
            if i + bsz > len(data):
                batch_data = data[i:]
            else:
                batch_data = data[i:i + bsz]
            input_ids, alignments, labels = batchify(batch_data, pad_idx)

            input_ids = input_ids.to('cuda:0')
            labels = labels.to('cuda:0')

            with torch.no_grad():
                output = model.forward(input_ids, alignments)
            _, preds = torch.topk(output, k=1, dim=-1)

            batch_correct = torch.eq(labels.squeeze(), preds.squeeze()).sum().item()
            num_correct += batch_correct
            total_num += labels.shape[-1]

        return num_correct / total_num

    elif metric == 'ner_f1':
        int2tag = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
        true = []
        predicted = []

        for i in range(0, len(data), bsz):
            if i + bsz > len(data):
                batch_data = data[i:]
            else:
                batch_data = data[i:i + bsz]
            input_ids, alignments, labels = batchify(batch_data, pad_idx)

            input_ids = input_ids.to('cuda:0')
            labels = labels.to('cuda:0')

            with torch.no_grad():
                output = model.forward(input_ids, alignments)
            _, preds = torch.topk(output, k=1, dim=-1)

            true.append([int2tag[x] for x in labels])
            predicted.append([int2tag[x] for x in preds])

        return f1_score(true, predicted)
        
        #To add:LAS
    elif metric == 'uas':
        num_correct_arcs = 0
        total_arcs = 0
        ignore_tokens = 0

        for i in range(0, len(data), bsz):
            if i + bsz > len(data):
                batch_data = data[i:]
            else:
                batch_data = data[i:i + bsz]
            input_ids, alignments, labels = batchify(batch_data, pad_idx)
    
            input_ids = input_ids.to('cuda:0')
    
            with torch.no_grad():
                output = model.forward(input_ids, alignments)
    
            # Extract predicted arcs
            predictions = []
            for op in output: #op shape (seq_len, seq_len)
                _, preds = torch.max(op, dim=-1) 
                preds = preds.cpu().numpy() # shape (seq_len)
                predictions.append(preds)
            # storing all predictions in a 1-d numpy array
            predictions = np.concatenate(predictions)
    
            # Calculate correct arcs
            for i, pred in enumerate(predictions):
                if labels[i] == -100:
                    ignore_tokens += 1
                if pred == labels[i]:
                    num_correct_arcs += 1
            total_arcs += labels.shape[0]
        total_arcs = total_arcs - ignore_tokens
        print("Correct tokens:", num_correct_arcs, "Total tokens:", total_arcs, file=sys.stderr)
        print("Total UAS score:", str(num_correct_arcs / total_arcs), file=sys.stderr)
        return num_correct_arcs / total_arcs
            
    

    else:
        raise Exception(f'Evaluation metric not recognized: {metric}')


def majority_label_baseline(text_data, label_set):
    """
    Return the majority-label baseline for a text dataset of sentences and their respective
    word-wise labels. Majority-label baseline is the accuracy achieved if a model predicts only the
    most common label
    """
    num_labels = len(label_set)
    label_counts = []
    words = {}
    for _ in range(num_labels):
        label_counts.append(0.)
    for sent, labels in text_data:
        for w, l in zip(sent, labels):
            if l == '_':
                continue
            label_counts[label_set.index(l)] += 1
            if w not in words:
                words[w] = [0. for _ in range(num_labels)]
            words[w][label_set.index(l)] += 1

    return (max(label_counts) / sum(label_counts) * 100, words)


def per_word_majority_baseline(text_data, word_label_count, label_set):
    """
    Return the per-word majority-label baseline for a text dataset of sentences and their respective
    word-wise labels. Per-word majority-label baseline is the accuracy achieved if a model predicts
    the most common label for each token
    """
    words = word_label_count
    for w in words:
        words[w] = label_set[np.argmax(words[w])]
    per_word_maj_correct = 0.
    per_word_maj_total = 0.
    for sent, labels in text_data:
        for w, l in zip(sent, labels):
            if w in words and words[w] == l:
                per_word_maj_correct += 1
            per_word_maj_total += 1

    return per_word_maj_correct / per_word_maj_total * 100


def mean_stdev(values):
    """
    Return the mean and standard deviation of the input list of values
    """
    mean = sum(values) / len(values)
    var = sum([(v - mean)**2 for v in values]) / len(values)
    std_dev = math.sqrt(var)
    return mean, std_dev


# task expects loose .conllu files for train an valid for the given language in
# the data_path dir
def finetune_classification(
    args,
    task,
    model,
    tokenizer,
    data_path,
    ckpt_path,
    lang='en',
    max_epochs=50,
    max_train_examples=math.inf
):
    """
    Conduct fine-tuning and evaluation for a token classification task. Fine-tune the model on four
    different random seeds per training set. If `args.zero_shot_transfer` is true and
    `args.transfer_source` is specified, conduct zero-shot transfer. Zero-shot transfer assumes that
    only test set(s) are available for the language(s) being evaluated. If doing zero-shot,
    fine-tune the model on the training/dev sets available for `args.transfer_source`, then loop
    over the specified language test sets at eval time
    """
    task_labels = _label_spaces[task]
    metric_name = 'f1' if args.eval_metric == 'ner_f1' else 'accuracy'
    is_zero_shot = getattr(args, 'zero_shot_transfer', False)
    has_transfer_source = getattr(args, 'transfer_source', False)
    do_zero_shot = is_zero_shot and has_transfer_source

    split_loader_function = _loader_functions_by_task[task]

    # If doing zero-shot transfer, load the train and dev sets for `args.transfer_source` and load
    # the test set for each of the languages in `args.langs`. Pre-process the data
    if do_zero_shot:
        train_valid_data = split_loader_function(
            data_path, args.transfer_source, splits=['train', 'dev'], task=task
        )
        train_text_data = train_valid_data['train']
        valid_text_data = train_valid_data['dev']
        test_text_data = {
            lg: split_loader_function(data_path, lg, splits=['test'], task=task)
            for lg in args.langs
        }

        train_data = whitespace_to_sentencepiece(
            tokenizer, train_text_data, task_labels, max_seq_length=args.max_seq_length
        )
        valid_data = whitespace_to_sentencepiece(
            tokenizer, valid_text_data, task_labels, max_seq_length=args.max_seq_length
        )
        test_data = {
            lg:
                whitespace_to_sentencepiece(
                    tokenizer, data, task_labels, max_seq_length=args.max_seq_length
                )
            for lg, data in test_text_data.items()
        }

        scores = defaultdict(list)
    # If not doing zero-shot transfer, pre-process the train, dev, and test sets for the language
    # in question. Also get and log the majority label baselines
    else:
        print("########")
        print(f"Beginning {task.upper()} evaluation for {lang}")
        # load train and eval data for probing on given langauge
        data_splits = split_loader_function(data_path, lang, task=task)
        train_text_data, valid_text_data, test_text_data = [
            data_splits[x] for x in ['train', 'dev', 'test']
        ]

        # Get majority sense baseline
        majority_baseline, words = majority_label_baseline(train_text_data, task_labels)
        print(f"majority label baseline: {round(majority_baseline, 2)}")

        per_word_baseline = per_word_majority_baseline(test_text_data, words, task_labels)
        print(f"per word majority baseline: {round(per_word_baseline, 2)}")

        # preprocessing can take in a hidden layer id if we want to probe inside model
        train_data = whitespace_to_sentencepiece(
            tokenizer, train_text_data, task_labels, max_seq_length=args.max_seq_length
        )
        valid_data = whitespace_to_sentencepiece(
            tokenizer, valid_text_data, task_labels, max_seq_length=args.max_seq_length
        )
        test_data = whitespace_to_sentencepiece(
            tokenizer, test_text_data, task_labels, max_seq_length=args.max_seq_length
        )

        scores = []

    random_seeds = [1, 2, 3, 4]
    epochs = []
    for rand_x in random_seeds:
        # set random seeds for model init, data shuffling
        torch.cuda.manual_seed(rand_x)
        torch.cuda.manual_seed_all(rand_x)
        np.random.seed(rand_x)
        random.seed(rand_x)

        # look for a control file to determine if a trial has already been done, e.g. by a
        # submitted job that was pre-empted before completing all trials
        lang_name = args.transfer_source if do_zero_shot else lang
        model_dir = os.path.join(ckpt_path, lang_name, str(rand_x))
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_control_path = os.path.join(model_dir, "checkpoint_control.yml")
        checkpoint_control_exists = os.path.isfile(checkpoint_control_path)
        if checkpoint_control_exists:
            with open(checkpoint_control_path, 'r') as fin:
                control_dict = yaml.load(fin, Loader=yaml.Loader)
                training_complete = control_dict['training_complete']
        else:
            control_dict = None
            training_complete = False

        # create probe model
        if task=="uas":
            tagger = DependencyTagger(model, len(task_labels))
        else:
            tagger = Tagger(model, len(task_labels))
        tagger_bsz = args.batch_size

        # only do training if we can't retrieve the existing checkpoint
        if not training_complete:
            # load criterion and optimizer
            tagger_lr = args.learning_rate
            gradient_accumulation = getattr(args, 'gradient_accumulation', 1)

            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            optimizer = torch.optim.Adam(tagger.parameters(), lr=tagger_lr)

            # train classification model
            tagger, num_epochs = train_model(
                tagger,
                train_data,
                valid_data,
                criterion,
                optimizer,
                tokenizer.pad_token_id,
                args.eval_metric,
                bsz=tagger_bsz,
                gradient_accumulation=gradient_accumulation,
                max_grad_norm=args.max_grad_norm,
                epochs=max_epochs,
                max_train_examples=max_train_examples,
                patience=args.patience,
                model_dir=model_dir,
                task=task
            )
        else:
            print(
                f"{lang_name} random seed {rand_x} previously trained; reading checkpoint",
                file=sys.stderr
            )
            num_epochs = control_dict['num_epochs_to_convergence']

        epochs.append(num_epochs * 1.0)

        # load best checkpoint for model state
        best_model_path = os.path.join(model_dir, "best_model.pt")
        tagger.load_state_dict(torch.load(best_model_path))

        print(f"random seed: {rand_x}")
        print(f"num epochs: {num_epochs}")

        # evaluate classificaiton model
        if do_zero_shot:
            for lg in args.langs:
                score = evaluate_model(
                    tagger,
                    test_data[lg],
                    tokenizer.pad_token_id,
                    bsz=tagger_bsz,
                    metric=args.eval_metric
                )
                score = score * 100
                scores[lg].append(score)
                print(f"{lg} {metric_name}: {round(score, 2)}")
            print("----")
        else:
            score = evaluate_model(
                tagger, test_data, tokenizer.pad_token_id, bsz=tagger_bsz, metric=args.eval_metric
            )
            score = score * 100
            scores.append(score)
            print(f"{metric_name}: {round(score, 2)}")
            print("----")

        # reinitalize the model for each trial if using randomly initialized encoder
        if args.random_weights:
            model, tokenizer = load_hf_model(
                args.model_class,
                args.model_name,
                task=args.task,
                random_weights=args.random_weights
            )
            model.cuda()
            model.eval()

            if do_zero_shot:
                train_data = whitespace_to_sentencepiece(
                    tokenizer, train_text_data, task_labels, max_seq_length=args.max_seq_length
                )
                valid_data = whitespace_to_sentencepiece(
                    tokenizer, valid_text_data, task_labels, max_seq_length=args.max_seq_length
                )
                test_data = {
                    lg:
                        whitespace_to_sentencepiece(
                            tokenizer, data, task_labels, max_seq_length=args.max_seq_length
                        )
                    for lg, data in test_text_data.items()
                }
            else:
                train_data = whitespace_to_sentencepiece(
                    tokenizer, train_text_data, task_labels, max_seq_length=args.max_seq_length
                )
                valid_data = whitespace_to_sentencepiece(
                    tokenizer, valid_text_data, task_labels, max_seq_length=args.max_seq_length
                )
                test_data = whitespace_to_sentencepiece(
                    tokenizer, test_text_data, task_labels, max_seq_length=args.max_seq_length
                )

    print("all trials finished")
    mean_epochs, epochs_stdev = mean_stdev(epochs)
    print(f"mean epochs: {round(mean_epochs, 2)}")

    if do_zero_shot:
        for lg in args.langs:
            mean_score, score_stdev = mean_stdev(scores[lg])
            print(f"{lg} mean {metric_name}: {round(mean_score, 2)}")
            print(f"{lg} standard deviation: {round(score_stdev, 2)}")
    else:
        mean_score, score_stdev = mean_stdev(scores)
        print(f"mean {metric_name}: {round(mean_score, 2)}")
        print(f"standard deviation: {round(score_stdev, 2)}")


def set_up_classification(args):
    """
    For each language being evaluated, set up POS tagging task by loading a huggingface
    model/tokenizer then calling the `pos` function
    """
    # Loop over languages to eval
    for lang in args.langs:
        # load huggingface model, tokenizer
        model, tokenizer = load_hf_model(
            args.model_class,
            args.model_name,
            task=args.task,
            random_weights=args.random_weights,
            tokenizer_path=args.tokenizer_path
        )
        model.cuda()
        model.eval()
        # Do pos task
        finetune_classification(
            args,
            args.task,
            model,
            tokenizer,
            args.dataset_path,
            args.checkpoint_path,
            lang=lang,
            max_epochs=args.epochs,
            max_train_examples=args.max_train_examples
        )


def set_up_classification_zero_shot(args):
    """
    Set up POS tagging task by loading a huggingface model/tokenizer then calling the `pos`
    function. Unlike `set_up_pos`, the loop over evaluation languages occurs inside the `pos`
    function in the zero-shot case, since the model is fine-tuned only on `transfer-source`
    """
    # load huggingface model, tokenizer
    model, tokenizer = load_hf_model(
        args.model_class,
        args.model_name,
        task=args.task,
        random_weights=args.random_weights,
        tokenizer_path=args.tokenizer_path
    )
    model.cuda()
    model.eval()
    # Do pos task
    finetune_classification(
        args,
        args.task,
        model,
        tokenizer,
        args.dataset_path,
        args.checkpoint_path,
        max_epochs=args.epochs,
        max_train_examples=args.max_train_examples
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # allow languages from any eval task in args
    all_choices = []
    for _, v in _lang_choices.items():
        all_choices.extend(v)

    # required parameters
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config_dict = vars(args)
    with open(args.config, 'r') as config_file:
        config_dict.update(yaml.load(config_file, Loader=yaml.Loader))

    print(f"Config file: {args.config}")

    os.makedirs(args.checkpoint_path, exist_ok=True)

    # set defaults for optional parameters
    # set max_train_examples to infinity it is is null or absent in config
    if not getattr(args, 'max_train_examples', False):
        args.max_train_examples = math.inf
    args.tokenizer_path = getattr(args, 'tokenizer_path', None)
    args.max_seq_length = getattr(args, 'max_seq_length', 512)
    args.learning_rate = getattr(args, 'learning_rate', float('5.0e-6'))
    args.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)

    #args.eval_metric = 'ner_f1' if args.task == 'ner' else 'acc'
    #account for task = uas
    if args.task == 'ner':
        args.eval_metric = 'ner_f1'
    elif args.task == 'uas':
        args.eval_metric = 'uas'
    else:
        args.eval_metric = 'acc'


    # ensure that given lang matches given task
    #for lang in args.langs:
    #    assert lang in _lang_choices[args.task]

    print(f"Pytorch version: {torch.__version__}")
    print(f"Pytorch CUDA version: {torch.version.cuda}")
    print(f"GPUs available: {torch.cuda.is_available()}")

    # set random seeds
    torch.manual_seed(args.random_seed)
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.zero_shot_transfer and args.transfer_source:
        print(
            f"Doing zero-shot transfer from source {args.transfer_source} to languages {', '.join(args.langs)}"
        )
        set_up_classification_zero_shot(args)
    else:
        set_up_classification(args)


#EOF
