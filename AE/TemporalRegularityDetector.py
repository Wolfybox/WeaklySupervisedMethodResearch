import torch
import torch.nn as nn


class TemporalRegularityDetector(nn.Module):
    def __init__(self, input_dim=1024, pretrained=False, model_dir=''):
        super(TemporalRegularityDetector, self).__init__()

        self.encode1 = nn.Linear(input_dim, 2000)
        self.relu_en1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.6)
        self.encode2 = nn.Linear(2000, 1000)
        self.relu_en2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.6)
        self.encode3 = nn.Linear(1000, 500)
        self.relu_en3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.6)
        self.encode4 = nn.Linear(500, 30)
        self.relu_en4 = nn.ReLU(inplace=True)
        self.dropout4 = nn.Dropout(0.6)

        self.decode1 = nn.Linear(30, 500)
        self.relu_de1 = nn.ReLU(inplace=True)
        self.dropout5 = nn.Dropout(0.6)
        self.decode2 = nn.Linear(500, 1000)
        self.relu_de2 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(0.6)
        self.decode3 = nn.Linear(1000, 2000)
        self.relu_de3 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(0.6)
        self.decode4 = nn.Linear(2000, input_dim)

        if pretrained:
            self.__load_pretrained_weights(model_dir)
        else:
            self.__init_weight()

    def forward(self, features):
        features = self.relu_en1(self.encode1(features))
        features = self.dropout1(features)
        features = self.relu_en2(self.encode2(features))
        features = self.dropout2(features)
        features = self.relu_en3(self.encode3(features))
        features = self.dropout3(features)
        features = self.relu_en4(self.encode4(features))
        features = self.dropout4(features)
        features = self.relu_de1(self.decode1(features))
        features = self.dropout5(features)
        features = self.relu_de2(self.decode2(features))
        features = self.dropout6(features)
        features = self.relu_de3(self.decode3(features))
        features = self.dropout7(features)
        features = self.decode4(features)
        return features

    def __load_pretrained_weights(self, model_dir):
        """Initialize network with pre-trained weights"""
        p_dict = torch.load(model_dir, map_location=torch.device('cpu'))
        s_dict = self.state_dict()
        for name in s_dict.keys():
            s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        nn.init.xavier_normal_(self.encode1.weight)
        nn.init.xavier_normal_(self.encode2.weight)
        nn.init.xavier_normal_(self.encode3.weight)
        nn.init.xavier_normal_(self.encode4.weight)

        nn.init.xavier_normal_(self.decode1.weight)
        nn.init.xavier_normal_(self.decode2.weight)
        nn.init.xavier_normal_(self.decode3.weight)
        nn.init.xavier_normal_(self.decode4.weight)
