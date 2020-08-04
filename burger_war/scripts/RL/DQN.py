'''
DQN implementation with pytorch
'''

import random
import numpy as np
from collections import namedtuple

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


'''
Transitionの各要素のtype
state <float>: tensor 2D, size=torch.Size([1, num_states]), e.g. tensor([[0.1, 0.3, 0.4, 0.5]])
action <int64(long)>: tensor 2D, size=torch.Size([1, 1]), e.g. tensor([[2]])
next_state <float>: same as state above
reward <float>: tensor 2D, size=torch.Size([1, 1]), e.g. tensor([[1.0]])
'''
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)

class Memory:

    def __init__(self, capacity):
        self.capacity = capacity  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)


class Brain:

    def __init__(self, num_states, num_actions, memory_cap):

        self.num_actions = num_actions
        self.memory = Memory(memory_cap)  # 経験を記憶するメモリオブジェクトを生成

        # ニューラルネットワークを構築
        self.model = nn.Sequential(
            nn.Linear(num_states, 72),
            nn.ReLU(),
            nn.Linear(72, 72),
            nn.ReLU(),
            nn.Linear(72, num_actions)
        )
        
        print(self.model)
        
        # 最適化手法の設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)


    def replay(self, batch_size):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        # ------------------------------
        # 1. メモリサイズの確認
        # ------------------------------

        # メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < batch_size:
            return

        # ------------------------------
        # 2. ミニバッチの作成
        # ------------------------------

        # メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(batch_size) # [Transition(state=tensor(...), action=tensor(...), ...), Transition(), ...]

        # 各変数をミニバッチに対応する形に変形
        batch = Transition(*zip(*transitions)) # Transition(state=(tensor(...), tensor(...), ...), action=(...), ...)

        # 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        state_batch = torch.cat(batch.state) # tensor([[0.1, 0.2, ....], [...], ...]), size=torch.Size([batch_size, num_states])
        action_batch = torch.cat(batch.action) # tensor([[1], [2], ....]), size=torch.Size([batch_size, 1])
        reward_batch = torch.cat(batch.reward) # tensor([[1.0], [0.0], ....]), size=torch.Size([batch_size, 1])
        
        # 最後のステップではnext_statesは存在しない(None)ので、最後のステップがバッチに含まれている場合にはそれを取り除く
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # ------------------------------
        # 3. 教師データとなるQ(s,a)の値を求める
        # state_action_valuesは、推論するモデルに対する出力値 -> 推論なので変化
        # expected_state_action_valuesは、1step前のモデルと次の状態(next_states)を元に計算した教師データ -> 教師データなので固定
        # ------------------------------

        # ネットワークを推論モードに切り替える
        self.model.eval()
        
        # ネットワークが出力したQ(s,a)を求める
        # self.model(state_batch)がすべての行動のQ値, size=torch.Size([batch_size, num_actions])
        # gatherを使って、action_batchの値(0 ~ num_actions-1)をもとに取った行動のQ値を取り出す
        state_action_values = self.model(state_batch).gather(1, action_batch) # size=torch.Size([batch_size, 1])

        # next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        # 現状のモデルをもとに次の状態で最大のQ値を求める
        # max(1)は列方向の最大値の[値, index]を返す
        # Q値(index=0)をdetachで取り出す(detachによりこの部分を勾配計算の対象外にする)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        # 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # ------------------------------
        # 4. パラメータの更新
        # ------------------------------

        # ネットワークを訓練モードに切り替える
        self.model.train()

        # 損失関数を計算する（smooth_l1_lossはHuberloss）
        # expected_state_action_valuesはsizeが[minbatch]になっているので、unsqueezeで[minibatch x 1]へ
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # パラメータを更新する
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # パラメータを更新

    def decide_action(self, state, episode):
        '''現在の状態に応じて、行動を決定する'''
        # eps-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)は[torch.LongTensor of size 1]をsize 1x1に変換します

        else:
            # 0,1の行動をランダムに返す
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 0,1の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class Agent:
    def __init__(self, num_states, num_actions, memory_cap):
        '''課題の状態と行動の数を設定する'''
        self.brain = Brain(num_states, num_actions, memory_cap)  # エージェントが行動を決定するための頭脳を生成

    def update_q_function(self):
        '''Q関数を更新する'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''行動を決定する'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        self.brain.memory.push(state, action, state_next, reward)
