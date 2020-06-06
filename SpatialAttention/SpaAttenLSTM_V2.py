import os

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class SpaAttenLSTM_V2(nn.Module):
    def __init__(self, pretrained=False, model_dir=''):
        super(SpaAttenLSTM_V2, self).__init__()
        self.atten_score = nn.Linear(512, 1)
        self.softmax_atten_score = nn.Softmax(dim=-2)
        self.score = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(input_size=512, num_layers=1, hidden_size=128, batch_first=True)
        self.__init_weight()
        if pretrained:
            self.__load_pretrained_weights(model_dir=model_dir)

    def forward(self, x, batch_x_len):
        # batch size X frame num X feature num X width X height
        batch_size = x.size()[0]
        frame_num = x.size()[1]
        feature_num = x.size()[-1]
        # batch size X frame num X pixel X feature num
        x = x.view(batch_size, frame_num, 196, feature_num)
        # batch size X frame num X pixel X 1
        atten_score = self.atten_score(x)
        norm_atten_score = self.softmax_atten_score(atten_score)
        # batch size X frame num X feature num
        weighted_x = torch.matmul(x.permute(0, 1, 3, 2), norm_atten_score).squeeze()
        # LSTM Module
        pack_batch = rnn_utils.pack_padded_sequence(weighted_x, batch_x_len, batch_first=True)
        outs, (last_hs, last_cs) = self.lstm(pack_batch)
        outs_pad, outs_len = rnn_utils.pad_packed_sequence(outs, batch_first=True)
        # Classification Module
        score_list = []
        for i in range(outs_pad.size()[0]):
            cur_len = outs_len[i]
            cur_data = outs_pad[i][:cur_len]
            cur_score = self.score(cur_data)
            cur_norm_score = self.sigmoid(cur_score)
            max_score = cur_norm_score.max(dim=0)[0]
            score_list.append(max_score)
        score_list = torch.stack(score_list).cuda()
        return score_list

    def feat_base_predict(self, x, batch_x_len):
        # batch size X frame num X feature num X width X height
        batch_size = x.size()[0]
        frame_num = x.size()[1]
        feature_num = x.size()[-1]
        # batch size X frame num X pixel X feature num
        x = x.view(batch_size, frame_num, 196, feature_num)
        # batch size X frame num X pixel X 1
        atten_score = self.atten_score(x)
        norm_atten_score = self.softmax_atten_score(atten_score)
        # batch size X frame num X feature num
        weighted_x = torch.matmul(x.permute(0, 1, 3, 2), norm_atten_score).squeeze(-1)
        # LSTM Module
        pack_batch = rnn_utils.pack_padded_sequence(weighted_x, batch_x_len, batch_first=True)
        outs, (last_hs, last_cs) = self.lstm(pack_batch)
        outs_pad, outs_len = rnn_utils.pad_packed_sequence(outs, batch_first=True)
        # Classification Module
        score_list = []
        for i in range(outs_pad.size()[0]):
            cur_len = outs_len[i]
            cur_data = outs_pad[i][:cur_len]
            cur_score = self.score(cur_data)
            cur_norm_score = self.sigmoid(cur_score)
            score_list.append(cur_norm_score)
        score_list = torch.stack(score_list).cuda()
        return score_list

    def __load_pretrained_weights(self, model_dir):
        """Initialize network with pre-trained weights"""
        p_dict = torch.load(model_dir, map_location=torch.device('cpu'))
        s_dict = self.state_dict()
        for name in s_dict.keys():
            s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        nn.init.xavier_normal_(self.atten_score.weight)
        nn.init.xavier_normal_(self.score.weight)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    pass
