import json
import sys
import random
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import os
import time
from torch.multiprocessing import Process, Queue, set_start_method
from MatrixGame import matrixGame

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

MaxProcessNum = 20

MaxTurn = 100

GPU_list = [2, 3, 4, 5, 6]

Beta = 0.95
lr = 0.002
train_fre = 3
batch_size = 64
batch_rate = 3
data_buffer = []


model_name = 'current_policy.model'

FIELD_HEIGHT = 9
FIELD_WIDTH = 9
SIDE_COUNT = 2
TANK_PER_SIDE = 2

dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]


class FieldItemType():
    Nil = 0
    Brick = 1
    Steel = 2
    Base = 3
    Tank = 4

class Action():
    Invalid = -2
    Stay = -1
    Up = 0
    Right = 1
    Down = 2
    Left = 3
    UpShoot = 4
    RightShoot = 5
    DownShoot = 6
    LeftShoot = 7

class WhoWins():
    NotFinished = -2
    Draw = -1
    Blue = 0
    Red = 1

class FieldObject:
    def __init__(self, x: int, y: int, itemType: int):
        self.x = x
        self.y = y
        self.itemType = itemType
        self.destroyed = False

    def __repr__(self):
        return str(self.itemType)

class Base(FieldObject):
    def __init__(self, side: int):
        super().__init__(4, side * 8, FieldItemType.Base)
        self.side = side


class Tank(FieldObject):
    def __init__(self, side: int, tankID: int):
        super().__init__(6 if side ^ tankID else 2, side * 8, FieldItemType.Tank)
        self.side = side
        self.tankID = tankID


class TankField:
    def __init__(self, policy, device, selfcopy = None):
        self.policy = policy
        self.device = device
        self.fieldContent = [
            [[] for x in range(FIELD_WIDTH)] for y in range(FIELD_HEIGHT)
        ]
        if selfcopy != None:
            self.tanks = copy.deepcopy(selfcopy.tanks)
            self.bases = copy.deepcopy(selfcopy.bases)
            for i in range(9):
                for j in range(9):
                    for t in selfcopy.fieldContent[i][j]:
                        if type(t) != Tank and type(t) != Base:
                            self.insertFieldItem(copy.deepcopy(t))
            for tanks in self.tanks:
                for tank in tanks:
                    self.insertFieldItem(tank)
            for base in self.bases:
                self.insertFieldItem(base)
            self.lastActions = copy.deepcopy(selfcopy.lastActions)
            self.actions = copy.deepcopy(selfcopy.actions)
            self.currentTurn = selfcopy.currentTurn
            return

        self.tanks = [[Tank(s, t) for t in range(TANK_PER_SIDE)] for s in range(SIDE_COUNT)]
        self.bases = [Base(s) for s in range(SIDE_COUNT)]
        self.lastActions = [[Action.Invalid for t in range(TANK_PER_SIDE)] for s in range(SIDE_COUNT)]
        self.actions = [[Action.Invalid for t in range(TANK_PER_SIDE)] for s in range(SIDE_COUNT)]
        self.currentTurn = 1

        for tanks in self.tanks:
            for tank in tanks:
                self.insertFieldItem(tank)
        for base in self.bases:
            self.insertFieldItem(base)
        self.insertFieldItem(FieldObject(4, 1, FieldItemType.Steel))
        self.insertFieldItem(FieldObject(4, 7, FieldItemType.Steel))
        self.data = []

    def random_brick(self):
        for i in range(9):
            self.insertFieldItem(FieldObject(i, 4, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(4, 2, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(4, 3, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(4, 5, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(4, 6, FieldItemType.Brick))

        self.insertFieldItem(FieldObject(3, 0, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(3, 1, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(5, 0, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(5, 1, FieldItemType.Brick))

        self.insertFieldItem(FieldObject(5, 7, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(3, 7, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(3, 8, FieldItemType.Brick))
        self.insertFieldItem(FieldObject(5, 8, FieldItemType.Brick))

        for i in range(3):
            for j in range(9):
                if len(self.fieldContent[i][j]) == 0:
                    if len(self.fieldContent[8-i][8-j]) != 0:
                        print("error",i,j)
                    pan = random.randint(0, 1)
                    if pan == 1:
                        self.insertFieldItem(FieldObject(j, i, FieldItemType.Brick))
                        self.insertFieldItem(FieldObject(8-j, 8-i, FieldItemType.Brick))

    def print_map(self):
        print(self.currentTurn)
        for i in range(9):
            for j in range(9):
                if self.fieldContent[i][j]:
                    print(self.fieldContent[i][j][0], end = "")
                else:
                    print(0, end = "")
            print("")
        print("")

    def insertFieldItem(self, item: FieldObject):
        self.fieldContent[item.y][item.x].append(item)
        item.destroyed = False

    def removeFieldItem(self, item: FieldObject):
        self.fieldContent[item.y][item.x].remove(item)
        item.destroyed = True

    def fromBinary(self, bricks: List[int]):
        for i in range(3):
            mask = 1
            for y in range(i * 3, i * 3 + 3):
                for x in range(FIELD_WIDTH):
                    if bricks[i] & mask:
                        self.insertFieldItem(FieldObject(x, y, FieldItemType.Brick))
                    mask = mask << 1

    def actionValid(self, side: int, tank: int, action: int) -> bool:
        if action >= Action.UpShoot and self.lastActions[side][tank] >= Action.UpShoot:
            return False
        if action == Action.Stay or action >= Action.UpShoot:
            return True
        x = self.tanks[side][tank].x + dx[action]
        y = self.tanks[side][tank].y + dy[action]
        return self.inRange(x, y) and not self.fieldContent[y][x]

    def allValid(self) -> bool:
        for tanks in self.tanks:
            for tank in tanks:
                if not tank.destroyed and not self.actionValid(tank.side, tank.tankID,
                                                               self.actions[tank.side][tank.tankID]):
                    return False
        return True

    def inRange(self, x: int, y: int) -> bool:
        return x >= 0 and x < FIELD_WIDTH and y >= 0 and y < FIELD_HEIGHT

    def setActions(self, side: int, actions: List[int]) -> bool:
        if self.actionValid(side, 0, actions[0]) and self.actionValid(side, 1, actions[1]):
            self.actions[side] = actions
            return True
        return False

    def doActions(self) -> bool:
        if not self.allValid():
            return False

        self.lastActions = self.actions.copy()

        for tanks in self.tanks:
            for tank in tanks:
                action = self.actions[tank.side][tank.tankID]
                if not tank.destroyed and action >= Action.Up and action < Action.UpShoot:
                    self.removeFieldItem(tank)
                    tank.x = tank.x + dx[action]
                    tank.y = tank.y + dy[action]
                    self.insertFieldItem(tank)

        itemsToBeDestroyed = []

        for tanks in self.tanks:
            for tank in tanks:
                action = self.actions[tank.side][tank.tankID]
                if not tank.destroyed and action >= Action.UpShoot:
                    x = tank.x
                    y = tank.y
                    action = action % 4
                    multipleTankWithMe = len(self.fieldContent[y][x]) > 1
                    while True:
                        x = x + dx[action]
                        y = y + dy[action]
                        if not self.inRange(x, y):
                            break
                        collides = self.fieldContent[y][x]
                        if collides:
                            if not multipleTankWithMe and len(collides) == 1 and collides[0].itemType == FieldItemType.Tank:
                                oppAction = self.actions[collides[0].side][collides[0].tankID]
                                if oppAction >= Action.UpShoot and action == (oppAction + 2) % 4:
                                    break
                            itemsToBeDestroyed += collides
                            break

        for item in itemsToBeDestroyed:
            if item.itemType != FieldItemType.Steel and not item.destroyed:
                self.removeFieldItem(item)

        self.currentTurn = self.currentTurn + 1
        self.actions = [[Action.Invalid for t in range(TANK_PER_SIDE)] for s in range(SIDE_COUNT)]

    def sideLose(self, side: int) -> bool:
        return (self.tanks[side][0].destroyed and self.tanks[side][1].destroyed) or self.bases[side].destroyed

    def whoWins(self) -> int:
        fail = [self.sideLose(s) for s in range(SIDE_COUNT)]
        if fail[0] == fail[1]:
            return WhoWins.Draw if fail[0] or self.currentTurn > MaxTurn else WhoWins.NotFinished
        if fail[0]:
            return WhoWins.Red
        return WhoWins.Blue

    def pick_randomly(self, pro):
        rand = []
        ans = []
        for i in range(2):
            rand.append(random.random())
        for i in range(2):
            if random.random() < 0.03:
                ans.append(random.randrange(len(pro[i])))
                continue
            tmp = 0
            for j in range(len(pro[i])):
                tmp += pro[i][j]
                if rand[i] <= tmp + 1e-3:
                    ans.append(j)
                    break
        return ans

    def Stateaction_to_input(self, action_1, action_2):
        tmpfield = TankField(self.policy, self.device, self)
        tmpfield.setActions(0, action_1)
        tmpfield.setActions(1, action_2)
        tmpfield.doActions()
        ans = tmpfield.State_to_state()
        return ans, self.whoWins()

    def State_to_state(self):
        state_map = np.zeros((9, 9, 9))
        for i in range(2):
            for j in range(2):
                if not self.tanks[i][j].destroyed:
                    t = 0 if self.lastActions[i][j] >= Action.UpShoot else 1
                    state_map[i*4+j*2+t][self.tanks[i][j].y][self.tanks[i][j].x] = 1

        state_map[8][1][4] = 10
        state_map[8][7][4] = 10
        for i in range(9):
            for j in range(9):
                if len(self.fieldContent[i][j]) == 1 and self.fieldContent[i][j][0].itemType == FieldItemType.Brick:
                    state_map[8][i][j] = 1

        return state_map


    def pick_action(self):
        available_actions = [[], []]
        valid_actions = [[[], []], [[], []]]
        for side in range(2):
            for tank in range(2):
                valid_actions[side][tank] = [action for action in range(Action.Stay, Action.LeftShoot + 1)
                                             if self.actionValid(side, tank, action)]
            for act0 in valid_actions[side][0]:
                for act1 in valid_actions[side][1]:
                    available_actions[side].append([act0, act1])
        Q = []
        for i in range(len(available_actions[0])):
            Q.append([])
            for j in range(len(available_actions[1])):
                state, whowins = self.Stateaction_to_input(available_actions[0][i], available_actions[1][j])
                if whowins == WhoWins.Draw:
                    value = 0
                elif whowins == WhoWins.Blue:
                    value = 1
                elif whowins == WhoWins.Red:
                    value = -1
                else:
                    value = self.policy(torch.stack([torch.from_numpy(state)]).float().to(self.device))
                Q[i].append(float(value))
        pro = [None, None]
        v, pro[0], pro[1] = matrixGame(Q)
        #print('value=',v)
        action_num = self.pick_randomly(pro)
        for side in range(2):
            self.setActions(side, available_actions[side][action_num[side]])
        self.data.append([self.State_to_state(), v])


    def data_pro(self, whowins):
        del self.data[0]
        r = 0.95
        R = 0
        if whowins == WhoWins.Blue:
            R = 1
        elif whowins == WhoWins.Red:
            R = -1
        else:
            mxy = max([(self.tanks[0][i].y if not self.tanks[0][i].destroyed else 0) for i in range(2)])
            mny = min([(self.tanks[1][i].y if not self.tanks[1][i].destroyed else 8) for i in range(2)])
            fR = [8-mxy, mny]
            if fR[0] < fR[1]:
                R = 0.3
            elif fR[0] > fR[1]:
                R = -0.3
        for i in range(len(self.data)-1,-1,-1):
            self.data[i].append(R)
            R = R*r

def self_play(policy, device):
    field = TankField(policy, device)
    field.random_brick()
    while True:
        #print(field.currentTurn)
        #field.print_map()
        if field.whoWins() != WhoWins.NotFinished:
            break
        field.pick_action()
        #print('actions=',field.actions)
        field.doActions()

    field.data_pro(field.whoWins())
    return field.data

def PlayProcess(id, q, fin, policy):
    device = torch.device("cuda:%d" % GPU_list[id % len(GPU_list)])
    print('play process(%d) started on device %s' % (id, str(device)))
    policy.to(device)
    sys.stdout.flush()
    while True:
        data = self_play(policy, device)
        for d in data:
            q.put(d)
        time.sleep(1)
        sz = q.qsize()
        sys.stdout.flush()
        if sz >= batch_size * batch_rate:
            break
    fin.put(id)
    print('process(%d) exit, size %d' % (id, q.qsize()))
    sys.stdout.flush()

def self_multiplay(policy):
    q = Queue()
    finq = []
    procs = []
    policy.train(False)
    for i in range(MaxProcessNum):
        fin = Queue()
        t = Process(target=PlayProcess, args=(i, q, fin, policy))
        t.start()
        procs.append(t)
        finq.append(fin)
    for i in range(MaxProcessNum):
        id = finq[i].get()
        print("finish process(%d)" % id)
        sys.stdout.flush()
    try:
        while not q.empty():
            data_buffer.append(q.get(timeout=1))
    except TimeoutError:
        pass
    print('finish Queue get')
    sys.stdout.flush()
    for i in range(len(procs)):
        p = procs[i]
        p.join(timeout=10)
        if p.is_alive():
            print('forcing process(%d) to terminate' % i)
            sys.stdout.flush()
            p.terminate()
    print('finish join')
    sys.stdout.flush()


if __name__ == '__main__':

    set_start_method('spawn')

    device_cpu = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DQ_nashN().to(device_cpu)
    optimizer = optim.Adam(policy.parameters(), weight_decay=1e-4)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if os.path.exists(model_name):
        policy.load_state_dict(torch.load(model_name))
        print('load model')
        sys.stdout.flush()
    policy.share_memory()

    def train(state_batch, value_batch, reward_batch):
        state_batch = torch.from_numpy(state_batch).float().to(device)
        value_batch = torch.from_numpy(value_batch).float().to(device)
        reward_batch = torch.from_numpy(reward_batch).float().to(device)
        optimizer.zero_grad()
        value = policy(state_batch)
        loss = F.mse_loss(value.view(-1), reward_batch + Beta * value_batch)
        loss.backward()
        optimizer.step()
        return float(loss)

    while True:
        self_multiplay(policy)
        print('data_buffer size:', len(data_buffer))
        sys.stdout.flush()
        loss_sum = 0
        count_batch = 0
        policy.to(device)
        policy.train(True)
        for ep in range(train_fre):
            random.shuffle(data_buffer)
            bs = batch_size
            for i in range(0, len(data_buffer), bs):
                ir = min(i + bs, len(data_buffer))
                mini_batch = data_buffer[i:ir]
                state_batch = np.array([data[0] for data in mini_batch])
                value_batch = np.array([data[1] for data in mini_batch])
                reward_batch = np.array([data[2] for data in mini_batch])
                loss = train(state_batch, value_batch, reward_batch)
                loss_sum += loss
                count_batch += 1

        print('loss:', loss_sum / count_batch)
        sys.stdout.flush()
        data_buffer = []
        policy.to(device_cpu)
        torch.save(policy.state_dict(), model_name)