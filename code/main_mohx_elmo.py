from util import get_num_lines, get_pos2idx_idx2pos, index_sequence, get_vocab, embed_indexed_sequence, \
    get_word2idx_idx2word, get_embedding_matrix, write_predictions, get_performance_VUAverb_val, process_paraphrase, \
    paraphrase_embed, sequence_embed, elmo_paraphrase_embed
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate
from model import TextEncoder, ParaEncoder, LinearFC, ModelSet, AttenLoss
from attention_model import AttentionModel

import torch
# import  torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from nltk.corpus import wordnet as wn

import csv
import h5py
import pickle
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Modified based on Gao Ge https://github.com/gao-g/metaphor-in-context

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
print(torch.cuda.is_available())
using_GPU = False



def get_correct_def(syns, sent):
    corres_def = ''
    max_ratio = 0
    for syn in syns:
        examples = syn.examples()

        for exam in examples:
            exam = exam.split()
            count = 0
            for word in exam:
                if word in sent:
                    count += 1
            ratio = count / len(exam)
            if ratio > max_ratio:
                corres_def = syn.definition()
                max_ratio = ratio
    return corres_def

"""
1. Data pre-processing
"""

'''
1.3 MOH-X
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a index: int: idx of the focus verb
    a label: int 1 or 0
'''
raw_mohx = []
paraphrase_index_dict = {}

with open('../data/MOH-X/MOH-X_formatted_svo_cleaned.csv') as f:
    # arg1  	arg2	verb	sentence	verb_idx	label
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        sentence = line[3]
        label_seq = [0] * len(sentence.split())
        pos_seq = [0] * len(label_seq)
        verb_idx = int(line[4])
        verb_label = int(line[5])
        label_seq[verb_idx] = verb_label
        sent = sentence.split()
        pos_seq[verb_idx] = 1  # idx2pos = {0: 'words that are not focus verbs', 1: 'focus verb'}
        target_word = sent[verb_idx]
        syns = wn.synsets(target_word)
        verb_set = [syn for syn in syns if '.v.' in syn.name()]
        definition_set = [process_paraphrase(syn.definition()) for syn in syns]
        corres_def = get_correct_def(syns, sent)
        corres_def = process_paraphrase(corres_def)
        corr_idx = definition_set.index(corres_def)


        raw_mohx.append([sentence.strip(), label_seq, pos_seq, corr_idx, definition_set])

with open('../elmo/mohx_file.txt') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
        line = line[:-1]
        paraphrase_index_dict[line] = str(count)
        count += 1


print('MOH-X dataset division: ', len(raw_mohx))

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
random.shuffle(raw_mohx)
"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_mohx)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# # glove_embeddings a nn.Embeddings
# glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
# set elmos_mohx=None to exclude elmo vectors. Also need to change the embedding_dim in later model initialization
elmos_mohx = h5py.File('../elmo/MOH-X_cleaned.hdf5', 'r')
elmos_paraphrases = h5py.File('../elmo/mohx_elmo_embeddings.hdf5', 'r')

bert_mohx = None
glove_mohx = None
suffix_embeddings = None
#suffix_embeddings = nn.Embedding(15, 50)

'''
2. 2
embed the datasets
'''

# temp1 = embed_indexed_sequence(raw_mohx[0][0], raw_mohx[0][2], word2idx, glove_mohx, elmos_mohx, bert_mohx, suffix_embeddings)
# temp2 = paraphrase_embed(raw_mohx[0][0], word2idx, glove_embeddings)

# second argument is the post sequence, which we don't need
embedded_mohx = [[embed_indexed_sequence(example[0], example[2], word2idx,
                                      glove_mohx, elmos_mohx, bert_mohx, suffix_embeddings),
                       example[2], example[1], example[3],
                       [elmo_paraphrase_embed(d, paraphrase_index_dict, elmos_paraphrases) for d in example[4]]]
                      for example in raw_mohx]
# embedded_mohx = [[paraphrase_embed(example[0], word2idx, glove_embeddings),
#                        example[2], example[1], example[3],
#                        [paraphrase_embed(d, word2idx, glove_embeddings) for d in example[4]]]
#                       for example in raw_mohx]

#100 times 10-fold cross validation
#for valid in range(100):

'''
2. 3 10-fold cross validation
'''
# separate the embedded_sentences and labels into 2 list, in order to pass into the TextDataset as argument
sentences = [example[0] for example in embedded_mohx]
poss = [example[1] for example in embedded_mohx]
labels = [example[2] for example in embedded_mohx]
paraphrases = [example[3] for example in embedded_mohx]
para_sets = [example[4] for example in embedded_mohx]
# ten_folds is a list of 10 tuples, each tuple is (list_of_embedded_sentences, list_of_corresponding_labels)
ten_folds = []
fold_size = int(647 / 10)
for i in range(10):
    ten_folds.append((sentences[i * fold_size:(i + 1) * fold_size],
                      poss[i * fold_size:(i + 1) * fold_size],
                      labels[i * fold_size:(i + 1) * fold_size],
                      paraphrases[i * fold_size: (i + 1) * fold_size],
                      para_sets[i * fold_size: (i + 1) * fold_size]))

idx2pos = {0: 'words that are not focus verbs', 1: 'focus verb'}

optimal_f1s = []
optimal_ps = []
optimal_rs = []
optimal_accs = []
optimal_attn_acc = []
predictions_all = []
for i in range(10):
    '''
    2. 3
    set up Dataloader for batching
    '''
    training_sentences = []
    training_labels = []
    training_poss = []
    training_paraphrase = []
    training_para_sets = []
    for j in range(10):
        if j != i:
            training_sentences.extend(ten_folds[j][0])
            training_poss.extend(ten_folds[j][1])
            training_labels.extend(ten_folds[j][2])
            training_paraphrase.extend(ten_folds[j][3])
            training_para_sets.extend(ten_folds[j][4])
    training_dataset_mohx = TextDataset(training_sentences, training_poss, training_labels, \
                                        training_paraphrase, training_para_sets)
    val_dataset_mohx = TextDataset(ten_folds[i][0], ten_folds[i][1], ten_folds[i][2], ten_folds[i][3], ten_folds[i][4])

    # Data-related hyperparameters
    batch_size = 2
    # Set up a DataLoader for the training, validation, and test dataset
    train_dataloader_mohx = DataLoader(dataset=training_dataset_mohx, batch_size=batch_size, shuffle=True,
                                       collate_fn=TextDataset.collate_fn)
    val_dataloader_mohx = DataLoader(dataset=val_dataset_mohx, batch_size=batch_size, shuffle=False,
                                     collate_fn=TextDataset.collate_fn)

    """
    3. Model training
    """
    '''
    3. 1 
    set up model, loss criterion, optimizer
    '''
    # Instantiate the model

    Exp_model = TextEncoder(embedding_dim=1024, hidden_size=256,
                                    num_layers=1, bidir=True,
                                    dropout1=0.5)
    Query_model = TextEncoder(embedding_dim=1024, hidden_size=256,
                                    num_layers=1, bidir=True,
                                    dropout1=0.5)
    Attn_model = AttentionModel(para_encoder_input_dim=512, query_dim=512, output_dim=256)
    para_encoder_attn_model = AttentionModel(para_encoder_input_dim=512, query_dim=512, output_dim=512)

    para_encoder = ParaEncoder(input_dim=1024, hidden_size=256,
                         num_layers=1, attn_model=para_encoder_attn_model, bidir=True,
                         dropout1=0.5)
    linearfc = LinearFC(num_classes=2, encoded_embedding_dim=512, context_dim=512, dropout1=0.2)
    # Move the model to the GPU if available
    models = ModelSet(Exp_model=Exp_model, Attn_model=Attn_model, para_encoder=para_encoder, linearfc=linearfc, Query_model=Query_model)
    if using_GPU:
        models = models.cuda()
    # Set up criterion for calculating loss
    loss_criterion = nn.NLLLoss()
    loss_criterion_2 = nn.NLLLoss()
    attn_loss_criterion = AttenLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    rnn_optimizer = optim.Adam(models.parameters(), lr=0.001)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = 60

    '''
    3. 2
    train model
    '''
    train_loss = []
    val_loss = []
    performance_matrix = None
    val_f1 = []
    val_p = []
    val_r = []
    val_acc = []
    val_attn = []
    train_f1 = []
    # A counter for the number of gradient updates
    num_iter = 0
    model_index = 0
    comparable = []
    for epoch in range(num_epochs):
        print("Starting epoch {}".format(epoch + 1))
        for (example_pos, example_text, example_lengths, labels, para_idx, paraset, paraset_attn, paraset_len_mask) in train_dataloader_mohx:

            with torch.no_grad():
                example_text = Variable(example_text)
                paraset = Variable(paraset)
                example_lengths = Variable(example_lengths)
                labels = Variable(labels)
                para_idx = Variable(para_idx)
                paraset_attn = Variable(paraset_attn)
                paraset_len_mask = Variable(paraset_len_mask)
            if using_GPU:
                example_text = example_text.cuda()
                example_lengths = example_lengths.cuda()
                labels = labels.cuda()
                para_idx = para_idx.cuda()
                paraset = paraset.cuda()
                paraset_attn = paraset_attn.cuda()
                paraset_len_mask = paraset_len_mask.cuda()

            pos_idx = [idx for pos in example_pos for idx, a in enumerate(pos) if pos[idx] == 1]
            target_label = [labels[batch][pos_idx[batch]] for batch in range(labels.size()[0])]
            predicted, attn_weight = models(example_pos, example_text, example_lengths,
                                            pos_idx, paraset, paraset_attn, paraset_len_mask)


            attn_loss = attn_loss_criterion(attn_weight, para_idx, labels)

            label_mask = torch.Tensor(target_label)
            #attn_loss = attn_loss * label_mask
            attn_loss = torch.mean(attn_loss)


            # # predicted shape: (batch_size, seq_len, 2)
            # predicted = RNNseq_model(example_text, example_lengths)
            # a = labels.view(-1)
            # b = predicted.view(-1, 3)
            target_label = torch.stack(target_label)
            batch_loss = loss_criterion(predicted.view(-1, 2), target_label.view(-1))

            batch_loss = attn_loss

            rnn_optimizer.zero_grad()

            batch_loss.backward(retain_graph=True)
            rnn_optimizer.step()
            num_iter += 1
            tmp = [x.grad for x in rnn_optimizer.param_groups[0]['params']]

            # Calculate validation and training set loss and accuracy every 50 gradient updates
            if num_iter % 100 == 0:
                avg_eval_loss, performance_matrix, attn_acc  = evaluate(idx2pos, val_dataloader_mohx, models,
                                                             loss_criterion, attn_loss_criterion, using_GPU)
                val_loss.append(avg_eval_loss)
                val_p.append(performance_matrix[0])
                val_r.append(performance_matrix[1])
                val_f1.append(performance_matrix[2])
                val_acc.append(performance_matrix[3])
                val_attn.append(attn_acc)
                print("Iteration {}. Validation Loss {}.".format(num_iter, avg_eval_loss))

            if num_iter % 100 == 0:
                avg_eval_loss, performance_matrix, attn_acc = evaluate(idx2pos, train_dataloader_mohx, models,
                                                                       loss_criterion, attn_loss_criterion, using_GPU)
                train_loss.append(avg_eval_loss)
                train_f1.append(performance_matrix[2])
                print("Iteration {}. Training Loss {}. Atten {}.\n".format(num_iter, avg_eval_loss, attn_acc))

    x = np.linspace(0, 15000, 150)
    l1, = plt.plot(x, val_loss[:150], label='linear line')
    l2, = plt.plot(x, train_loss[:150], color='red', linewidth=1.0, linestyle='--', label='square line')

    plt.legend(handles=[l1, l2], labels=['val_loss', 'train_loss'], loc='best')
    print("Training done for fold {}".format(i))

    print('val_f1: ', val_f1)

    idx = 0

    optimal_f1s.append(np.nanmax(val_f1))
    idx = val_f1.index(optimal_f1s[-1])
    optimal_ps.append(val_p[idx])
    optimal_rs.append(val_r[idx])
    optimal_accs.append(val_acc[idx])
    optimal_attn_acc.append(val_attn[idx])

print('F1 on MOH-X by 10-fold = ', optimal_f1s)
print('Precision on MOH-X = ', np.nanmean(np.array(optimal_ps)))
print('Recall on MOH-X = ', np.nanmean(np.array(optimal_rs)))
print('F1 on MOH-X = ', np.nanmean(np.array(optimal_f1s)))
print('Accuracy on MOH-X = ', np.nanmean(np.array(optimal_accs)))
print('Paraphrase accuracy on MOH-X = ', np.nanmean(np.array(optimal_attn_acc)))
