import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMModel(nn.Module):
    def __init__(self, config):
        super(BiLSTMModel, self).__init__()
        '''
        baseline hyper parameters:
        hidden_size: 32
        num_layers: 3
        dropout_p: 0.5
        '''
        self.n_input_channel = len(config['dynamic_labels']) 
        self.n_output = len(config['static_labels'])
        self.hidden_size = config['bilstm_hidden_size']
        self.num_layers = config['bilstm_num_layers']
        dropout_p = config['bilstm_dropout_p']
        self.dropout = nn.Dropout(p=dropout_p)

        self.sequence_length = config['target_padding_length']
        self.lstm1 = nn.LSTM(input_size=self.n_input_channel, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        
        self.fc1 = nn.Linear(self.hidden_size * 2, 1600)  # 2 for bidirectional
        self.fc2 = nn.Linear(1600, 1)

    def forward(self, x):
        out, (h, c) = self.lstm1(x)
        out = self.dropout(out[:, -1, :])  # Use the output of the last time step
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    

