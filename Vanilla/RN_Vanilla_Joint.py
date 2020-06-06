import torch
import torch.nn as nn


class RNVanillaUnit(nn.Module):
    def __init__(self, input_dim=4096):
        super(RNVanillaUnit, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.dropout1 = nn.Dropout(p=0.6)
        self.relu1 = nn.ReLU()
        self.__init_weight()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def __init_weight(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


class RNVanillaJoint(nn.Module):
    """
    The Ranking Regression network.
    """

    def __init__(self, pretrained=False, model_dir='', input_dim=4096):
        super(RNVanillaJoint, self).__init__()
        self.rgb = RNVanillaUnit(input_dim=input_dim)
        self.flow = RNVanillaUnit(input_dim=input_dim)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        if pretrained:
            self.__load_pretrained_weights(model_dir=model_dir)

    def forward(self, x):
        rgb_feat = x[:, :, :1024]
        flow_feat = x[:, :, 1024:]
        rgb_out = self.rgb(rgb_feat)
        flow_out = self.flow(flow_feat)
        # merged = rgb_out + flow_out
        cat = torch.cat([rgb_out, flow_out], dim=-1)
        scores = self.sigmoid(self.fc2(self.fc1(cat)))
        return scores

    def predict(self, x):
        return self.forward(x)

    def __init_weight(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def __load_pretrained_weights(self, model_dir):
        """Initialize network with pre-trained weights"""
        p_dict = torch.load(model_dir, map_location=torch.device('cpu'))
        s_dict = self.state_dict()
        for name in s_dict.keys():
            s_dict[name] = p_dict[name]
        self.load_state_dict(s_dict)


if __name__ == "__main__":
    device = torch.device('cuda:5')
    pass
