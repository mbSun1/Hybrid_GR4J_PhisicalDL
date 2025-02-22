import torch.nn as nn
import torch

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq='day', cycle_time=True,dropout=0.5):
        super(TimeFeatureEmbedding, self).__init__()
        self.freq = freq
        self.PI = 3.1415
        self.cycle_time = cycle_time
        if cycle_time:
            # self.sin_t = nn.Parameter(torch.empty(1))
            # self.cos_t = nn.Parameter(torch.empty(1))
            # nn.init.uniform_(self.sin_t)
            # nn.init.uniform_(self.cos_t)
            freq_map = {'day': 2, 'month': 2}
            self.scales = {'month': 13, 'day': 32}
        else:
            freq_map = {'day': 2, 'month': 1}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def cycle(self, time):
        month, day = time[:, :, 1:2], time[:, :, -1:]
        if self.freq == 'month':
            if self.cycle_time:
                t_input = month / 13 * self.PI * 2
                t_sin = (1 + torch.sin(t_input)) / 2
                t_cos = (1 + torch.cos(t_input)) / 2
                # t_sin = (1 + torch.sin(t_input * self.sin_t)) / 2
                # t_cos = (1 + torch.cos(t_input * self.cos_t)) / 2
                return torch.cat([t_sin, t_cos], -1)
            else:
                return month
        elif self.freq == 'day':
            if self.cycle_time:
                t_input = (month + day / 32) / 13 * self.PI * 2
                t_sin = (1 + torch.sin(t_input)) / 2
                t_cos = (1 + torch.cos(t_input)) / 2
                return torch.cat([t_sin, t_cos], -1)
            else:
                return torch.cat([month, day], -1)

    def forward(self, x):
        """ x: [N, L, 2], means month and day """
        x = self.dropout(self.embed(self.cycle(x)))  # embed month and day
        return x