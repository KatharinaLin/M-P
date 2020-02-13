import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch
from attention_model import AttentionModel




class TextEncoder(nn.Module):
    '''
    This class is used as an encoder of the input text.
    # embedding_dim: The input dimension of the word embeddings
    # hidden_size: The size of the BiLSTM hidden state.
    # num_layers: Number of layers to use in BiLSTM
    # bidir: boolean of whether to use bidirectional or not
    # dropout1: dropout on input to BiLSTM
    '''

    def __init__(self, embedding_dim, hidden_size, num_layers, bidir=True,
                 dropout1=0.2):

        super(TextEncoder, self).__init__()

        self.rnn = nn.LSTM(input_size=embedding_dim , hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bidirectional=bidir)

        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)


    def forward(self, inputs):

        embedded_input = self.dropout_on_input_to_LSTM(inputs)

        output, _ = self.rnn(embedded_input)

        return output


class ParaEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, attn_model, bidir=True,
                 dropout1=0.2):
        '''
        This module is used as the encoder of the paraphrase.
        :param embedding_dim: The dimension of the input tensor
        :param hidden_size: The size of the BiLSTM hidden state
        :param num_layers: The number of layers in the BiLSTM
        :param attn_model: The model to compute the attention weight of every paraphrases
        :param bidir: boolean of whether to use bidirectional or not
        :param dropout1: dropout on input to BiLSTM
        '''

        super(ParaEncoder, self).__init__()

        self.encoder = nn.LSTM(input_size=input_dim , hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bidirectional=bidir)

        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)

        self.attn_model = attn_model


    def forward(self, inputs, target_query_state, paraset_len_mask):
        '''
        :param inputs: The ELMo embeddings of paraphrases
        :param target_query_state: The latent representation of the target word
        :param paraset_len_mask: List for the length of each paraphrase
        :return: The representation of all the paraphrases
        '''

        embedded_input = self.dropout_on_input_to_LSTM(inputs)
        batch_size = embedded_input.size()[0]
        embedded_input_size = embedded_input.size()

        # embedded_input: [batch_size, para_set_size, max_len_for_para, dimension]
        # output: [batch_size * para_set_size, max_len_for_para, dimension]
        output, (a, h) = self.encoder(embedded_input.view(embedded_input_size[0]*embedded_input_size[1],
                                                          embedded_input_size[2], embedded_input_size[3]))

        # get the tensor of mask for paraphrases from the paraset_len_mask
        len_arr_mask = []
        for batch in range(batch_size):
            for i in range(embedded_input.size()[1]):
                len_arr = torch.LongTensor(paraset_len_mask[:, i])
                t_mask = torch.ones(embedded_input_size[2], dtype=torch.int64)
                t_mask[len_arr[batch]:] = 0
                len_arr_mask.append(t_mask)
        len_arr_mask = torch.stack(len_arr_mask)


        # Adjust the encoded representation of the target word according to the size of paraphrases' representation
        target_query_state_repeated = target_query_state.repeat(1, embedded_input_size[1])
        target_query_state_repeated = target_query_state_repeated.view(embedded_input_size[0]*embedded_input_size[1],
                                                                       target_query_state.size()[1])

        # get the attention weight
        word_attn_weight_for_para = self.attn_model(output, target_query_state_repeated, len_arr_mask)

        word_attn_weight_for_para = word_attn_weight_for_para.unsqueeze(-1)
        para_context_vector = word_attn_weight_for_para * output
        para_context_vector = torch.sum(para_context_vector, [1])

        #output_sum = torch.sum(output, [1])
        #final_output = output_sum.view(embedded_input_size[0], embedded_input_size[1], output_sum.size()[1])

        final_output = para_context_vector.view(embedded_input_size[0], embedded_input_size[1], para_context_vector.size()[1])
        return final_output

class LinearFC(nn.Module):
    def __init__(self, num_classes, encoded_embedding_dim, context_dim, dropout1=0.2):
        '''
        :param num_classes: The number of classes in the classification problem.
        :param encoded_embedding_dim: The dimension of the encoded word
        :param context_dim: the dimension of the context vector of the paraphrase
        :param dropout1: dropout on input to FC
        '''

        super(LinearFC, self).__init__()


        self.output_to_label = nn.Linear(encoded_embedding_dim + context_dim, num_classes)

        self.dropout_on_input_to_FC = nn.Dropout(dropout1)


    def forward(self, encoded_state, context_vector):

        inputs = torch.cat((encoded_state, context_vector),dim=1)

        embedded_input = self.dropout_on_input_to_FC(inputs)

        output = self.output_to_label(embedded_input)

        normalized_output = torch.log_softmax(output, dim=-1)

        return normalized_output


class ModelSet(nn.Module):
    def __init__(self, Exp_model, Attn_model, para_encoder, linearfc, Query_model):

        super(ModelSet, self).__init__()

        self.Exp_model = Exp_model

        self.Query_model = Query_model

        self.para_encoder = para_encoder

        self.attn_model = Attn_model

        self.linearfc = linearfc


    def encode_text(self, inputs):

        return self.Exp_model(inputs)

    def encode_text_q(self, inputs):

        return self.Query_model(inputs)

    def encode_paraphrase(self, inputs):

        return self.para_encoder(inputs)

    def compute_attn(self, inputs):

        return self.attn_model(inputs)

    def fct_transmation(self, inputs):

        return self.linearfc(inputs)

    def forward(self, example_pos, example_text, example_lengths, pos_idx, paraset, paraset_attn, paraset_len_mask):

        encoder_state = self.Exp_model(example_text)

        indices = torch.LongTensor(pos_idx)
        indices = indices.unsqueeze(1)


        target_encoder_state = torch.gather(encoder_state, 1, indices[..., None].expand(*indices.shape, encoder_state.shape[-1]))
        target_encoder_state = target_encoder_state.squeeze(1)

        query_state = self.Query_model(example_text)

        target_query_state_tmp = torch.gather(query_state, 1,
                                            indices[..., None].expand(*indices.shape, query_state.shape[-1]))
        target_query_state = target_query_state_tmp.squeeze(1)


        para_encoded = self.para_encoder(paraset, target_query_state, paraset_len_mask)

        # attn_weight = self.attn_model(para_encoded, target_encoder_state, paraset_attn)
        attn_weight = self.attn_model(para_encoded, target_query_state, paraset_attn)


        attn_weight = attn_weight.unsqueeze(-1)

        context_vector = attn_weight * para_encoded

        context_vector = torch.sum(context_vector, [1])

        predicted = self.linearfc(target_encoder_state, context_vector)

        return predicted, attn_weight



class AttenLoss(nn.Module):
    def __init__(self):
        super(AttenLoss, self).__init__()

    def forward(self, attn_weight, para_idx, labels):
        '''
        :param attn_weight: the attention weight of all the paraphrases
        :param para_idx: the corresponding index of the target word
        :param labels: the labels of metaphorical or not
        :return: the loss of selecting the corresponding paraphrases
        '''

        batch_size = labels.size()[0]
        class_num = attn_weight[0].size()[0]

        # get the one-hot representation of the correct paraphrase
        para_idx_stack = torch.unsqueeze(para_idx, 1)
        one_hot = torch.zeros(batch_size, class_num).scatter_(1, para_idx_stack, 1)
        one_hot = torch.unsqueeze(one_hot, -1)

        # compute the loss
        #loss = - one_hot * torch.log(attn_weight+1e-12) - (1 - one_hot) * torch.log(1 - attn_weight+1e-12)
        loss = - one_hot * torch.log(attn_weight+1e-12)
        return loss






