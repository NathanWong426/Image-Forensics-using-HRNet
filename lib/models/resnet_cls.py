import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    # 输入数据的通道数，输出数据的通道数
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# 用于resnet18,resnet34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 使存放数据的顺序与取出数据的顺序一致
        # 比如:存，a,b,c,3,2,1,取：a,b,c,3,2,1
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        # inplace=True，进行覆盖运算
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        # nn.Sequential是一个有序容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.relu(out)

        return out


# 用于resnet50,resnet101,resnet152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # 加速收敛，提高泛化能力，用于防止过拟合
        # 使得每一层神经网络的输入保持相同的分布
        m['bn1'] = nn.BatchNorm2d(planes)
        # 使小于0的值为0，大于等于0的保持不变
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1 = nn.Sequential(m)
        # 这里的block是一个类
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))

        self.group2 = nn.Sequential(
            OrderedDict([
                # 输入参数为：in_feature,out_feature
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )
        # 用于初始化网络中的每个module
        # nn.modules()返回网络中的所有modules
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # blocks代表模块要重复进行的操作次数
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 如果步长为1，使输入输出的通道数一致
        # 输入通道!=输出通道*4，输入通道为64
        # 也就是说，只要调用make_layer这个函数，downsample必执行，但执行发生在block操作之后
        # 每个重复的卷积块的首次操作，都要在旁路连接上进行下采样操作
        # 那么，该卷积块剩下的几次操作，就不再进行下采样操作，即在旁路连接上不进行下采样操作
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        # list数组，用*转化，将layers拆成一个个元素
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # 把四维张量变为2维张量后，才能作为FC的输入
        x = x.view(x.size(0), -1)
        x = self.group2(x)

        return x


def resnet18(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet18'], model_root)
        model_root = 'model/resnet18-5c106cde.pth'
        # 加载预训练好的模型参数
        model_data = torch.load(model_root)
        # 将模型参数加载到net中
        model.load_state_dict(model_data)
    return model


def resnet34(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet34'], model_root)
        model_root = 'model/resnet34-333f7ec4.pth'
        model_data = torch.load(model_root)
        model.load_state_dict(model_data)
    return model


def resnet50(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet50'], model_root)
        model_root = 'model/resnet50-19c8e357.pth'
        model_data = torch.load(model_root)
        model.load_state_dict(model_data)
    return model


def resnet101(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet101'], model_root)
        model_root = 'model/resnet101-5d3b4d8f.pth'
        model_data = torch.load(model_root)
        model.load_state_dict(model_data)
    return model


def resnet152(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet152'], model_root)
        model_root = 'model/resnet152-b121ed2d.pth'
        model_data = torch.load(model_root)
        model.load_state_dict(model_data)
    return model


def resnet18(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet18'], model_root)
        model_root = 'model/resnet18-5c106cde.pth'
        # 加载预训练好的模型参数
        model_data = torch.load(model_root)
        # 将模型参数加载到net中
        model.load_state_dict(model_data)
    return model


def resnet34(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet34'], model_root)
        model_root = 'model/resnet34-333f7ec4.pth'
        model_data = torch.load(model_root)
        model.load_state_dict(model_data)
    return model


def resnet50(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet50'], model_root)
        model_root = 'model/resnet50-19c8e357.pth'
        model_data = torch.load(model_root)
        model.load_state_dict(model_data)
    return model


def resnet101(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet101'], model_root)
        model_root = 'model/resnet101-5d3b4d8f.pth'
        model_data = torch.load(model_root)
        model.load_state_dict(model_data)
    return model


def resnet152(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        # misc.load_state_dict(model, model_urls['resnet152'], model_root)
        model_root = 'model/resnet152-b121ed2d.pth'
        model_data = torch.load(model_root)
        model.load_state_dict(model_data)
    return model

