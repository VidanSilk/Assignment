# get architecture
def get_arch(opt):
    arch = opt.arch
    type = opt.input_type + '_' + opt.target_type

    print('You choose ' + arch + '...')
    if arch == 'ED_FNN':
        from models.ED_FNN import ED_FNN
        model = ED_FNN(win_len=opt.win_len, hop_len=opt.hop_len, fft_len=opt.fft_len,
                       processing_type=type, norm=opt.norm_layer, act=opt.act)
    elif arch == 'ED_CNN':
        from models.ED_CNN import ED_CNN
        model = ED_CNN(win_len=opt.win_len, hop_len=opt.hop_len, fft_len=opt.fft_len,
                       processing_type=type, norm=opt.norm_layer, act=opt.act, hidd_ch=opt.hidd_ch,
                       kernel_size=opt.kernel_size, dilation=opt.dilation)
        
    # recent speech enhancement model

    elif arch == 'NUNet-TLS':
        if opt.input_type != 'mag':
            raise Exception("You should choose input type \'mag\'")
        from models.ref.NUNet_TLS import NUNet_TLS
        model = NUNet_TLS()

    else:
        raise Exception("Arch error!")

    return model


# get trainer and validator (train method)
def get_train_mode(opt):
    input_type = opt.input_type
    loss_type = opt.loss_type

    print('You choose ' + loss_type + '...')
    if loss_type == 'base':  # single loss function
        if input_type == 'mag':
            from .trainer import base_mag_train
            from .trainer import base_mag_valid
            trainer = base_mag_train
            validator = base_mag_valid
        elif (input_type == 'complex_operation') or (input_type == 'complex_channel'):
            from .trainer import base_complex_train
            from .trainer import base_complex_valid
            trainer = base_complex_train
            validator = base_complex_valid
        else:
            raise Exception("Input type error! Please check the option")
    elif loss_type == 'joint':  # multiple(joint) loss function
        if input_type == 'mag':
            from .trainer import joint_mag_train
            from .trainer import joint_mag_valid
            trainer = joint_mag_train
            validator = joint_mag_valid
        elif (input_type == 'complex_operation') or (input_type == 'complex_channel'):
            from .trainer import joint_complex_train
            from .trainer import joint_complex_valid
            trainer = joint_complex_train
            validator = joint_complex_valid
        else:
            raise Exception("Input type error! Please check the option")
    else:
        raise Exception("Loss type error!")

    return trainer, validator


def get_loss(opt):
    from torch.nn import L1Loss
    from torch.nn.functional import mse_loss
    loss_oper = opt.loss_oper

    print('You choose loss operation with ' + loss_oper + '...')
    if loss_oper == 'l1':
        loss_calculator = L1Loss()
    elif loss_oper == 'l2':
        loss_calculator = mse_loss
    else:
        raise Exception("Arch error!")

    return loss_calculator


def get_normalization_layer(ch, dim=1, norm='bn'):
    if norm == 'bn':  # batch norm
        if dim == 1:
            from torch.nn import BatchNorm1d
            return BatchNorm1d(ch)
        elif dim == 2:
            from torch.nn import BatchNorm2d
            return BatchNorm2d(ch)
        else:
            raise Exception("Dimension error! Please check the option")
    elif norm == 'in':  # instance norm
        from torch.nn import GroupNorm
        return GroupNorm(ch, ch)
    elif norm == 'gn': # group norm
        from torch.nn import GroupNorm
        groups = 2
        return GroupNorm(groups, ch)
    elif norm == 'ln': # layer norm
        from torch.nn import GroupNorm
        return GroupNorm(1, ch)
    else:
        raise Exception("Normalization error! Please check the option")


def get_activation_layer(act='relu'):
    if act == 'sigmoid':
        from torch.nn import Sigmoid
        return Sigmoid()
    elif act == 'relu':
        from torch.nn import ReLU
        return ReLU()
    elif act == 'leakyrelu':
        from torch.nn import LeakyReLU
        return LeakyReLU()
    elif act == 'tanh':
        from torch.nn import Tanh
        return Tanh()
    elif act == 'prelu':
        from torch.nn import PReLU
        return PReLU()
    else:
        raise Exception("Activation error! Please check the option")
