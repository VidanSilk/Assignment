from utils import ConvSTFT, ConviSTFT
from .baseBlocks import *


class EnhancementStrategy(nn.Module):
    def __init__(self, win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size=(3, 2), dilation=(1, 1)):
        super().__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.fft_half_len = fft_len // 2
        self.norm = norm
        self.act = act
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.hidd_ch = hidd_ch

"""
입력된 magnitude 스펙트로그램을 통해 오디오 신호를 매핑하는 전략입니다.
encoder와 decoder 네트워크를 설정하여 입력을 잘 압축하고 다시 복원합니다.
"""
class MagMapping(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation):
        super().__init__(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_ch)):
            if idx == 0:
                self.encoder.append(
                    convLayer(1, hidd_ch[idx], kernel_size=kernel_size,
                              dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    convLayer(hidd_ch[idx - 1], hidd_ch[idx], kernel_size=kernel_size,
                              dilation=dilation, norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_ch), 0, -1):
            if idx == 1:
                self.decoder.append(
                    upConvLayer(hidd_ch[idx - 1], 1, kernel_size=kernel_size,
                                dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    upConvLayer(hidd_ch[idx - 1], hidd_ch[idx - 2], kernel_size=kernel_size,
                                dilation=dilation, norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='real')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='real')

    def forward(self, mag):
        hx = mag.unqueeze(1)[:, :, 1:]

        out = hx
        for idx, layer in enumerate(self.encoder):
            out = layer(out)

        for idx, layer in enumerate(self.decoder):
            out = layer(out)

        mag_out = functional.pad(out, [0, 0, 1, 0])
        return mag_out

"""
magnitude 기반의 마스킹을 통해 입력된 스펙트로그램을 개선합니다.
encoder와 decoder를 통해 입력을 압축 및 복원하며, mask를 적용하여 입력 스펙트로그램을 개선합니다.
"""
class MagMasking(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation):
        super().__init__(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_ch)):
            if idx == 0:
                self.encoder.append(
                    convLayer(1, hidd_ch[idx], kernel_size=kernel_size,
                              dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    convLayer(hidd_ch[idx - 1], hidd_ch[idx], kernel_size=kernel_size,
                              dilation=dilation, norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_ch), 0, -1):
            if idx == 1:
                self.decoder.append(
                    upConvLayer(hidd_ch[idx - 1], 1, kernel_size=kernel_size,
                                dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    upConvLayer(hidd_ch[idx - 1], hidd_ch[idx - 2], kernel_size=kernel_size,
                                dilation=dilation, norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='real')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='real')

    def forward(self, mag):
        hx = mag.unsqueeze(1)[:, :, 1:]

        out = hx
        for idx, layer in enumerate(self.encoder):
            out = layer(out)

        for idx, layer in enumerate(self.decoder):
            out = layer(out)

        mask_mag_out = functional.pad(out, [0, 0, 1, 0])
        mag_out = mag * mask_mag_out.squeeze(1)
        return mag_out

"""
복소수 입력을 다루는 복소수 연산 기반의 매핑 전략입니다.
복소수에 대한 encoder와 decoder를 설정하여 실수 및 허수 부분을 개선합니다.
"""
class ComplexOperationMapping(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation):
        super().__init__(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_ch)):
            if idx == 0:
                self.encoder.append(
                    complexConvLayer(1, hidd_ch[idx], kernel_size=kernel_size,
                                     dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    complexConvLayer(hidd_ch[idx - 1], hidd_ch[idx], kernel_size=kernel_size,
                                     dilation=dilation, norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_ch), 0, -1):
            if idx == 1:
                self.decoder.append(
                    complexUpConvLayer(hidd_ch[idx - 1], 1, kernel_size=kernel_size,
                                       dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    complexUpConvLayer(hidd_ch[idx - 1], hidd_ch[idx - 2], kernel_size=kernel_size,
                                       dilation=dilation, norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')

    def forward(self, real_imag):
        real = real_imag[0].unsqueeze(1)[:, :, 1:]
        imag = real_imag[1].unsqueeze(1)[:, :, 1:]

        real_out, imag_out = real, imag
        for idx, layer in enumerate(self.encoder):
            real_out, imag_out = layer(real_out, imag_out)

        for idx, layer in enumerate(self.decoder):
            real_out, imag_out = layer(real_out, imag_out)

        real_out = functional.pad(real_out, [0, 0, 1, 0]).squeeze(1)
        imag_out = functional.pad(imag_out, [0, 0, 1, 0]).squeeze(1)
        mag_out = torch.sqrt(real_out ** 2 + imag_out ** 2)
        return real_out, imag_out, mag_out
"""
복소수 입력에 대한 마스킹 기반 전략입니다.
복소수 입력의 mask를 적용하여 개선된 복소수 신호를 생성합니다.
"""
class ComplexOperationMasking(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation):
        super().__init__(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_ch)):
            if idx == 0:
                self.encoder.append(
                    complexConvLayer(1, hidd_ch[idx], kernel_size=kernel_size,
                                     dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    complexConvLayer(hidd_ch[idx - 1], hidd_ch[idx], kernel_size=kernel_size,
                                     dilation=dilation, norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_ch), 0, -1):
            if idx == 1:
                self.decoder.append(
                    complexUpConvLayer(hidd_ch[idx - 1], 1, kernel_size=kernel_size,
                                       dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    complexUpConvLayer(hidd_ch[idx - 1], hidd_ch[idx - 2], kernel_size=kernel_size,
                                       dilation=dilation, norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')

    def forward(self, real_imag):
        real, imag = real_imag[0], real_imag[1]
        hx_real = real.unsqueeze(1)[:, :, 1:]
        hx_imag = imag.unsqueeze(1)[:, :, 1:]

        real_mask_out, imag_mask_out = hx_real, hx_imag
        for idx, layer in enumerate(self.encoder):
            real_mask_out, imag_mask_out = layer(real_mask_out, imag_mask_out)

        for idx, layer in enumerate(self.decoder):
            real_mask_out, imag_mask_out = layer(real_mask_out, imag_mask_out)

        real_mask_out = functional.pad(real_mask_out, [0, 0, 1, 0])
        imag_mask_out = functional.pad(imag_mask_out, [0, 0, 1, 0])

        real_out = real * real_mask_out.squeeze(1)
        imag_out = imag * imag_mask_out.squeeze(1)
        mag_out = torch.sqrt(real_out ** 2 + imag_out ** 2)
        return real_out, imag_out, mag_out

"""
복소수 입력의 실수 및 허수 채널을 별도로 처리하여 개선합니다.
"""
class ComplexChannelMapping(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation):
        super().__init__(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_ch)):
            if idx == 0:
                self.encoder.append(
                    convLayer(2, hidd_ch[idx], kernel_size=kernel_size,
                              dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    convLayer(hidd_ch[idx - 1], hidd_ch[idx], kernel_size=kernel_size,
                              dilation=dilation, norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_ch), 0, -1):
            if idx == 1:
                self.decoder.append(
                    upConvLayer(hidd_ch[idx - 1], 2, kernel_size=kernel_size,
                                dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    upConvLayer(hidd_ch[idx - 1], hidd_ch[idx - 2], kernel_size=kernel_size,
                                dilation=dilation, norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')

    def forward(self, real_imag):
        real = real_imag[0].unsqueeze(1)[:, :, 1:]
        imag = real_imag[1].unsqueeze(1)[:, :, 1:]

        hx = torch.cat([real, imag], dim=1)

        out = hx
        for idx, layer in enumerate(self.encoder):
            out = layer(out)

        for idx, layer in enumerate(self.decoder):
            out = layer(out)

        real_out = out[:, 0]
        imag_out = out[:, 1]

        real_out = functional.pad(real_out, [0, 0, 1, 0])
        imag_out = functional.pad(imag_out, [0, 0, 1, 0])
        mag_out = torch.sqrt(real_out ** 2 + imag_out ** 2)
        return real_out, imag_out, mag_out

"""
복소수 입력의 각 채널에 대해 mask를 적용하여 개선된 복소수 신호를 생성합니다.
"""
class ComplexChannelMasking(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation):
        super().__init__(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_ch)):
            if idx == 0:
                self.encoder.append(
                    convLayer(2, hidd_ch[idx], kernel_size=kernel_size,
                              dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    convLayer(hidd_ch[idx - 1], hidd_ch[idx], kernel_size=kernel_size,
                              dilation=dilation, norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_ch), 0, -1):
            if idx == 1:
                self.decoder.append(
                    upConvLayer(hidd_ch[idx - 1], 2, kernel_size=kernel_size,
                                dilation=dilation, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    upConvLayer(hidd_ch[idx - 1], hidd_ch[idx - 2], kernel_size=kernel_size,
                                dilation=dilation, norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')

    def forward(self, real_imag):
        real, imag = real_imag[0], real_imag[1]
        hx_real = real.unsqueeze(1)[:, :, 1:]
        hx_imag = imag.unsqueeze(1)[:, :, 1:]

        hx = torch.cat([hx_real, hx_imag], dim=1)

        out = hx
        for idx, layer in enumerate(self.encoder):
            out = layer(out)

        for idx, layer in enumerate(self.decoder):
            out = layer(out)

        real_mask_out = out[:, 0]
        imag_mask_out = out[:, 1]

        real_mask_out = functional.pad(real_mask_out, [0, 0, 1, 0])
        imag_mask_out = functional.pad(imag_mask_out, [0, 0, 1, 0])

        real_out = real * real_mask_out
        imag_out = imag * imag_mask_out
        mag_out = torch.sqrt(real_out ** 2 + imag_out ** 2)
        return real_out, imag_out, mag_out
"""
선택된 전략에 따라 MagMapping, MagMasking, ComplexOperationMapping 등의 클래스를 생성하여 사용
"""

class ED_CNN(nn.Module):
    def __init__(self, win_len=400, hop_len=100, fft_len=512, processing_type='mag_mapping', norm='bn', act='relu',
                 hidd_ch=[32, 64, 128, 256], kernel_size=(3, 2), dilation=(1, 1)):
        super().__init__()

        # Initialize different strategies
        self.strategies = {
            'mag_mapping': MagMapping(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation),
            'mag_masking': MagMasking(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size, dilation),
            'complex_operation_mapping': ComplexOperationMapping(win_len, hop_len, fft_len, norm, act, hidd_ch,
                                                                 kernel_size, dilation),
            'complex_operation_masking': ComplexOperationMasking(win_len, hop_len, fft_len, norm, act, hidd_ch,
                                                                 kernel_size, dilation),
            'complex_channel_mapping': ComplexChannelMapping(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size,
                                                             dilation),
            'complex_channel_masking': ComplexChannelMasking(win_len, hop_len, fft_len, norm, act, hidd_ch, kernel_size,
                                                             dilation)
        }
        self.current_strategy = self.strategies[processing_type]
        self.stft = self.current_strategy.stft
        self.istft = self.current_strategy.istft

    def forward(self, x):
        return self.current_strategy(x)
