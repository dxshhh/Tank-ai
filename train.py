import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os, sys
import random
from torch.optim import lr_scheduler

class DQ_nashN(nn.Module):

    def __init__(self):
        super(DQ_nashN, self).__init__()
        self.conv1 = nn.Conv2d(9, 16, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.head = nn.Linear(3600, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

lr = 2e-3
batch_size = 128
epoch_size = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = DQ_nashN().to(device)
optimizer = optim.Adam(policy.parameters(), weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
model_name = 'current_policy.model'

for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def train(state_batch, reward_batch):
    state_batch = torch.from_numpy(state_batch).float().to(device)
    reward_batch = torch.from_numpy(reward_batch).float().to(device)
    optimizer.zero_grad()
    value = policy(state_batch)
    loss = F.mse_loss(value.view(-1), reward_batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss)


if __name__ == '__main__':

    gdata = pickle.load(open('matches.data', 'rb'))
    print('load data')
    sys.stdout.flush()

    if os.path.exists(model_name):
        policy.load_state_dict(torch.load(model_name))
        print('load model')
        sys.stdout.flush()

    random.shuffle(gdata)

    for epoch in range(epoch_size):
        print('epoch:', epoch)
        sys.stdout.flush()

        loss_sum = 0
        count_batch = 0
        for i in range(0, len(gdata), batch_size):
            ir = min(i + batch_size, len(gdata))
            mini_batch = gdata[i:ir]
            state_batch = np.array([data[0] for data in mini_batch])
            reward_batch = np.array([data[1] for data in mini_batch])
            loss = train(state_batch, reward_batch)

            loss_sum += loss
            count_batch += 1
            if count_batch == 1000:
                print('loss:', loss_sum / 1000)
                sys.stdout.flush()
                count_batch = 0
                loss_sum = 0

        torch.save(policy.state_dict(), model_name)