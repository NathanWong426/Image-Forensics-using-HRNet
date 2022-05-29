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

