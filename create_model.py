import torch
import torch.nn as nn
from utils.hydrolayerGR4J import PRNNLayer
from utils.embed import TimeFeatureEmbedding


def create_model(input_dim, hs_cnn, hs_rnn=8, es=1, model_type='hybrid-J', frozen_module_type='torchlstm',
                 ks=25, device=torch.device('cpu')):
    """
    Paramters
        input_dim: dim of input features
        hs_cnn: hidden size for the first Conv1D layer
        hs_rnn: hidden size for LSTM of standard DL model or frozen module of hybrid-Z model
        es: embedding size of timestamp
    """
    if model_type == 'hybrid-J':
        model = hybrid_J_model(input_dim=input_dim, hs_cnn=hs_cnn, es=es, kernel_size=ks, device=device)

    return model

#hybrid_J_model
class hybrid_J_model(nn.Module):
    def __init__(self, input_dim, hs_cnn, es=1, kernel_size=9, device=torch.device('cpu')):
        super(hybrid_J_model, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.embedding = TimeFeatureEmbedding(es, freq='day')
        self.PRNNLayer = PRNNLayer(device=device)
        self.ScaleLayer1 = nn.BatchNorm1d(input_dim - 1 + es)  # + 1(streamflow) - 2(month and date)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(input_dim - 1 + es, hs_cnn, kernel_size=self.kernel_size,
                               padding=self.kernel_size - 1)
        self.ScaleLayer2 = nn.BatchNorm1d(hs_cnn)
        self.conv2 = nn.Conv1d(hs_cnn, 1, kernel_size=1, padding='same')

    def forward(self, x, hidden):
        # PRNN layer
        embeded = self.embedding(x[:, :, -2:])
        x_input = torch.cat((x[:, :, :-2], embeded), dim=-1)
        hidden = self.init_hidden(x_input.size(0)) if hidden is None else hidden
        hydro_output, hidden = self.PRNNLayer(x_input, hidden)
        # 1d CNN layers
        new_input = torch.concat((x_input, hydro_output), dim=-1).permute(0, 2, 1)  # (N,L,C) to (N,C,L)
        x = self.ScaleLayer1(new_input)
        x = self.conv1(x)
        x = x[:, :, :-(self.kernel_size - 1)]
        x = self.relu(self.ScaleLayer2(x))
        x = self.conv2(x)
        x = self.relu(x)
        return x.permute(0, 2, 1), hidden

    def init_hidden(self, bsz):
        # initial hydrology model params
        weight = next(self.parameters()).data
        return weight.new_zeros(bsz, 1).to(self.device), weight.new_zeros(bsz, 1).to(self.device)




class physical_model(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(physical_model, self).__init__()
        self.device = device
        self.PRNNLayer = PRNNLayer(device=device)

    def forward(self, x_input, hidden):
        hidden = self.init_hidden(x_input.size(0)) if hidden is None else hidden
        hydro_output, hidden = self.PRNNLayer(x_input, hidden)
        return hydro_output, hidden

    def init_hidden(self, bsz):
        # 初始隐藏状态的设置，这里只是简单返回零张量
        return torch.zeros(bsz, 1).to(self.device), torch.zeros(bsz, 1).to(self.device)


