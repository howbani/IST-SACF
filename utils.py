# This piece of code was copied & modified from the following source:
#
#    Title: CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning
#    Author: Laskin, Michael and Srinivas, Aravind and Abbeel, Pieter
#    Date: 2020
#    Availability: https://github.com/MishaLaskin/curl

import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
import psutil
from skimage.util.shape import view_as_windows

class eval_mode(object):  #上下文管理器 好处：临时关闭训练特性，同时不影响模型在离开这段代码后的正常训练
    def __init__(self, *models):  #支持可变参数，把要切换模式的多个模型一次性接收
        self.models = models
 
    def __enter__(self):  #定义进入上下文时的钩子 __enter__
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):  #定义退出上下文时的钩子 __exit__
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):  #让目标网络的参数缓慢跟随主网络的参数（定义函数，传入主网络net、目标网络target_net、平滑系数tau）
    for param, target_param in zip(net.parameters(), target_net.parameters()):  #同步遍历两个网络的参数对（要求两者结构一致）
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data  #用加权和更新目标参数，实现指数滑动平均
        )


def set_seed_everywhere(seed):  #把多个常用随机数发生器统一“播种”，让实验尽可能复现
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module): #用来快速、低成本地判断“模型状态是否变化”
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):  #一个简单的“创建目录”工具函数，用于训练脚本批量创建实验输出目录
    # dir_path = os.path.join(os.path.abspath(os.getcwd()), dir_path)
    try:
        os.mkdir(dir_path)
    except OSError:
        print('Unable to create directory ' + dir_path)
    return dir_path


def preprocess_obs(obs, bits=5):  #图像观测预处理函数，把像素做“降比特”和“去量化 + 归一化”
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits  #计算每通道的桶数
    assert obs.dtype == torch.float32  #断言输入
    if bits < 8:  #若小于 8 位，则执行“降比特”
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins  #将像素缩放到 [0,1) 左右的范围：数值稳定性和收敛速度
    obs = obs + torch.rand_like(obs) / bins  #加入均匀噪声，这一步称为“去量化”，把离散桶边界打散为连续值，降低离散化带来的伪影
    obs = obs - 0.5   #将分布中心平移到 0 附近
    return obs


class ReplayBuffer(Dataset): #经验回放缓冲区  按需随机采样批次用于训练  支持两类采样：标准训练采样和增强采样（CURL的对比学习采样）
    """Buffer to store environment transitions."""  #这个 ReplayBuffer 类是用来存储环境转换的，负责初始化一个固定容量的环形缓冲
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, augmentor, transform=None):  #action_shape 动作空间形状
        self.capacity = capacity  #最大存储条目数，达到容量后采用环形覆盖
        self.batch_size = batch_size
        self.device = device      #将采样出的张量放在 cpu 或 cuda 上
        self.augmentor = augmentor
        self.transform = transform   #transform ：可选的额外变换，供 __getitem__ 按需应用

        # The proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8  #本体感知特征一维（np.float32），像素观测二维（np.uint8）
        
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype) #预分配五个数组（使用 np.empty 预分配固定容量的环形缓冲，避免动态扩容的开销）
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32) #预分配奖励数组self.rewards ，形状(capacity, 1)，类型float32（每步一个标量奖励）
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        # Check if replay buffer size exceeds available memory  要检查缓冲区大小是否超过可用内存
        total_bytes = (self.obses.nbytes + self.next_obses.nbytes + self.actions.nbytes + self.rewards.nbytes + self.not_dones.nbytes)  #求各数组的字节数并求和得到 total_bytes ，表示整个缓冲区的理论占用
        total_memory = psutil.virtual_memory().total #读取系统“总内存”字节数 total_memory（物理内存容量）
        total_available_memory = psutil.virtual_memory().available  #读取系统“当前可用内存”字节数（可立即分配的内存）
        print('-'*50)
        if total_bytes > 1024**3:  #以 GB 为单位打印：缓冲区大小、系统总内存、系统当前可用内存
            print('Replay buffer size: %.2f GB' % (total_bytes / 1024 / 1024 / 1024))
            print('Total memory: %.2f GB' % (total_memory / 1024 / 1024 / 1024))
            print('Total available memory: %.2f GB' % (total_available_memory / 1024 / 1024 / 1024))
        else:  #以 MB 为单位打印：缓冲区大小、系统总内存、系统当前可用内存
            print('Replay buffer size: %.2f MB' % (total_bytes / 1024 / 1024))
            print('Total memory: %.2f MB' % (total_memory / 1024 / 1024))
            print('Total available memory: %.2f MB' % (total_available_memory / 1024 / 1024))
        print('-'*50)
        if total_bytes > total_available_memory: #若缓冲区总字节数大于当前可用内存，抛出 ValueError 异常   让程序在初始化阶段就停止，避免后续采样/训练过程中发生 OOM
            raise ValueError('Replay buffer size exceeds available memory')  #OOM：Out Of Memory 的缩写，表示“内存不足”错误


        self.idx = 0  #通过 idx 和取模实现环形覆盖，旧数据会被新数据按顺序覆盖
        self.last_save = 0  #最近一次保存到磁盘的起始位置，用于只保存“新增的数据片段”
        self.full = False  #full 切换采样范围，写满后能从整个缓冲区均匀采样

    def add(self, obs, action, reward, next_obs, done): #“写入”一条完整的转移到缓冲区当前索引位置，并维护环形索引
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):#从经验回放缓冲区随机采样转成PyTorch张量，返回给训练过程使用
        
        # Sample a random batch of transitions from the replay buffer  随机生成批次索引 idxs
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)

        # Convert to Pytorch tensors on the device  将采到的观测转为张量并移动到self.device同时 .float() 转为float32以便后续连续运算
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones  #返回采样到的五元组（张量形式）

    def sample_cpc(self):#定义了ReplayBuffer的采样函数sample_cpc，用于为CURL的对比学习构造训练批次
        #从回放缓冲区随机抽取一批图像观测，按增强器生成“锚样本”和“正样本”，并同时返回强化学习所需的 (obs, action, reward, next_obs, not_done)
        # Sample a random batch of transitions from the replay buffer
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=self.batch_size)  #从回放缓冲区随机抽取批次索引idxs，抽样是“有放回”的

        # 当前图像增强都走张量路径，不再区分随机裁剪的 NumPy 分支。
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        pos = obses.detach().clone()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        pos = torch.as_tensor(pos, device=self.device).float()

        # Apply augmentations to the targets
        obses = self.augmentor.training_augmentation(obses)
        next_obses = self.augmentor.training_augmentation(next_obses)
        pos = self.augmentor.training_augmentation(pos)

        # Store CPC kwargs in a dict  说明下面要把 CPC相关的参数收集到一个字典中
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos, time_anchor=None, time_pos=None) #

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def save(self, save_dir):  #增量保存（把“自上次保存以来新增的数据片段”写到磁盘，避免重复保存全量数据）
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx  #增量保存的“起点推进到当前写入位置”，为下一次增量保存做准备
        torch.save(payload, path)

    def load(self, save_dir):  #回放缓冲区的“增量加载”方法
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):  #ReplayBuffer 作为 PyTorch 数据集接口的取样方法
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity 

class FrameStack(gym.Wrapper):  #用于强化学习环境的 Frame Stacking（帧堆叠）包装器  为卷积编码器提供最近几步的视觉上下文，而不用引入 RNN
    def __init__(self, env, k):  #初始化方法，接收原始环境 env 和要堆叠的帧数 k
        gym.Wrapper.__init__(self, env)
        self._k = k   #堆叠的帧数
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape  #原始环境的观察空间形状
        self.observation_space = gym.spaces.Box( #新的观察空间：通道数变为原来的k倍
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps
        self.curl_driving = False

    def reset(self):  #重置环境，返回初始状态
        obs = self.env.reset()
        self.curl_driving = self.env.curl_driving
        for _ in range(self._k):  #用初始观察填充所有k帧
            self._frames.append(obs)
        return self._get_obs()  # 返回堆叠后的观察

    def step(self, action):  #步进方法
        obs, reward, done, info = self.env.step(action)
        self.env.curl_driving = self.curl_driving
        self._frames.append(obs)  
        return self._get_obs(), reward, done, info

    def _get_obs(self):  # 观察获取方法 _get_obs
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)



