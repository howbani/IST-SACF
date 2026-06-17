# This piece of code was copied & modified from the following source:
#
#    Title: CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning
#    Author: Laskin, Michael and Srinivas, Aravind and Abbeel, Pieter
#    Date: 2020
#    Availability: https://github.com/MishaLaskin/curl
#基于卷积神经网络的图像编码器，主要用于将输入的图像观测（如游戏画面）编码为低维的特征向量
import torch
import torch.nn as nn


def tie_weights(src, trg):  #两个神经网络层之间的权重共享  src:源层(即要被共享的参数所在的层)，trg:目标层
    assert type(src) == type(trg) #断言两者类型一致（如都为nn.Conv2d或都为nn.Linear），避免将不兼容的参数绑定
    trg.weight = src.weight
    trg.bias = src.bias

# Calculate output dimensions for different input sizes: http://layer-calc.com/,  针对不同输入尺寸计算输出维度：http://layer-calc.com/
# or just use the commented print statement on line 86 of this file               或者直接使用本文件第86行处被注释的print语句

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

# for 76 x 135 inputs
OUT_DIM_RECT_76_135 = {4: [31, 61],}

# for 90 x 160 inputs  #本项目默认相机尺寸即 90×160，卷积层数为 4，最终特征图尺寸 38×73
OUT_DIM_RECT_90_160 = {4: [38, 73],}


class CNNEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""  #对像素观测进行卷积编码的编码器
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, output_logits=False): #初始化编码器架构
        super().__init__()
        assert len(obs_shape) == 3   #确保输入是 (C, H, W) 三维像素张量

        if obs_shape[1:] == (84, 84):  #据输入分辨率与层数选择卷积后特征图的空间尺寸
            out_dim = OUT_DIM[num_layers]
        elif obs_shape[1:] == (64, 64):
            out_dim = OUT_DIM_64[num_layers]
        elif obs_shape[1:] == (76, 135) and num_layers == 4:
            out_dim = OUT_DIM_RECT_76_135[num_layers]
        elif obs_shape[1:] == (90, 160) and num_layers == 4:
            out_dim = OUT_DIM_RECT_90_160[num_layers]
        else:
            raise NotImplementedError("Encoder does not support input shape")

        self.obs_shape = obs_shape   #保存基本配置：输入形状、输出特征维度、卷积层数
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # Build convolutional layers  构建第一层卷积
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=obs_shape[0], 
                                              out_channels=num_filters, 
                                              kernel_size=3, 
                                              stride=2)])  #第一层卷积核大小3x3，（stride）步长2，输出通道数32  第一层使用 stride=2 做下采样，快速减小空间尺寸

        for _ in range(num_layers - 1):  #追加剩余num_layers-1个卷积层
            self.convs.append(nn.Conv2d(in_channels=num_filters, 
                                        out_channels=num_filters, 
                                        kernel_size=3, 
                                        stride=1))  #继续提取局部视觉特征但不再下采样

        # Build linear layers    构建全连接层
        self.fc = nn.Linear(num_filters * out_dim[0] * out_dim[1], self.feature_dim)  #将卷积输出的特征图展平为长度映射到feature_dim
        self.ln = nn.LayerNorm(self.feature_dim)  #构建层归一化

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd): #重参数化：从均值和对数标准差中采样，确保可微性（让AI能看清楚随机性的来源，从而精确调整配方参数）
        std = torch.exp(logstd)           #将对数标准差转换为实际标准差
        eps = torch.randn_like(std)       #从标准正态分布采样随机噪声
        return mu + eps * std             #重参数化：得到来自 N(mu, std^2) 的样本【让AI能看清楚随机性的来源，从而精确调整配方参数】

    def forward_conv(self, obs):  #卷积部分的前向传播（视觉提取特征） 将输入的原始图像像素通过多层卷积网络，最终转换为一维的特征向量
        obs = obs / 255.           #图像预处理：像素值归一化：从[0,255]缩放到[0,1]
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))   # 第一层卷积（stride=2，进行下采样）
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):   #后续卷积层：深层特征提取
            conv = torch.relu(self.convs[i](conv))
            # print(f'conv{i+1} shape: {conv.shape}')
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)   #将特征图展平为一维向量（展平只是重新排列维度，并未丢弃数值），以便送入后面的全连接层fc
        return h                          #线性层要求二维输入(batch, in_features)

    def forward(self, obs, detach=False):  #完整前向传播，包括卷积和全连接层
        h = self.forward_conv(obs)   #将输入的图像数据通过卷积网络提取特征，再通过全连接层和归一化处理，最终输出一个固定维度的特征向量
        #forward_conv()  调用卷积部分提取图像特征77行
        if detach:
            h = h.detach()  #detach=True：切断梯度，卷积层参数不更新；detach=False：正常梯度传播，所有层都更新

        h_fc = self.fc(h)   # 全连接层：降维到目标特征维度
        self.outputs['fc'] = h_fc  ## 记录全连接层输出

        h_norm = self.ln(h_fc)   # 层归一化：稳定训练过程
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm  
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out  #输出tanh压缩后的值，范围[-1,1]，防止特征值过大，提高数值稳定性

        return out   #输出：50维特征向量（特征值反映了输入图像在不同特征上的抽象表示强度）

    def copy_conv_weights_from(self, source):#卷积层（通用特征提取器）参数共享（当前编码器的所有卷积层与另一个源编码器的对应卷积层共享相同的权重参数）
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):  #训练日志记录
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

if __name__ == "__main__":

    # Get torch device  设备选择(主程序中的辅助函数)
    def get_device():  
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  #相当于 Mac 平台的“GPU 设备”
        else:
            device = torch.device("cpu")
        return device

    # Define the encoder
    frame_stack = 3   #帧堆叠数量（连续3帧作为输入）
    n_channels = 3 * frame_stack #总通道数（RGB × 3帧）
    height = 90  #
    width = 160    #
    obs_shape = (n_channels, height, width)   #观测形状（通道，高，宽）
    feature_dim = 50        #输出特征维度
    model = CNNEncoder(obs_shape=obs_shape, feature_dim=feature_dim, num_layers=4, num_filters=32, output_logits=False)  #创建CNNEncoder实例，配置4层卷积，32个滤波器
    device = get_device()                                                                   #output_logits=False：使用tanh激活输出
    model.to(device)  #将模型参数转移到选定的设备（GPU/CPU）

    # Print the model summary
    batch_size = 256  #模拟训练时的批次大小
    input_size = (batch_size, n_channels, height, width)  #测试输入张量形状
    try:  #模型测试和摘要，执行模型测试并输出详细信息
        import torchinfo
        dummy_input = torch.randn(input_size).to(device)  #创建测试输入
        output = model(dummy_input)       # 执行完整前向传播
        print(output.shape)   #输出形状验证
        torchinfo.summary(  # 生成模型摘要
            model, 
            input_size=input_size,
            device=device,
        )
    except:
        print("torchinfo is not installed. Skipping summary.")  #异常处理分支