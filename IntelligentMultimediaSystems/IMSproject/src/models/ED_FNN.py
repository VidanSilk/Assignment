from utils import ConvSTFT, ConviSTFT
from .baseBlocks import *

#nn.Module 상속, 같이 사용하는 매개변수 초기화 
class EnhancementStrategy(nn.Module):
    def __init__(self, win_len, hop_len, fft_len, hidd_dim, norm, act):
        super().__init__()
        self.win_len = win_len #윈도우의 길이
        self.hop_len = hop_len #합 길이
        self.fft_len = fft_len # FFT 길이
        self.fft_half_len = fft_len // 2 #FFT길의 반
        self.norm = norm #정규화 방법
        self.act = act #활성화 함수
        self.hidd_dim = hidd_dim #숨겨진 차원

#EnhancementStrategy를 상속
#완전 연결 레이어(fcLayer)를 사용하여 
#오디오 신호의 진폭 스펙트럼(mag)을 매핑.
#ConvSTFT와 ConviSTFT를 사용 
#단시간 푸리에 변환을 수행
class MagMapping(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, hidd_dim, norm, act):
        super().__init__(win_len, hop_len, fft_len, hidd_dim, norm, act)
        self.encoder = nn.ModuleList() #인코더 모듈
        self.decoder = nn.ModuleList() #디코더 모듈 

        for idx in range(len(hidd_dim)):
            if idx == 0:
                self.encoder.append(
                    fcLayer(self.fft_half_len, hidd_dim[idx], norm=self.norm, act=self.act)
                ) #1계층
            else:
                self.encoder.append(
                    fcLayer(hidd_dim[idx-1], hidd_dim[idx], norm=self.norm, act=self.act)
                ) #2계층 이후 

        for idx in range(len(hidd_dim), 0, -1):
            if idx == 1:
                self.decoder.append(
                    fcLayer(hidd_dim[idx-1], self.fft_half_len, norm=self.norm, act=self.act)
                ) #1계층 
            else:
                self.decoder.append(
                    fcLayer(hidd_dim[idx-1], hidd_dim[idx-2], norm=self.norm, act=self.act)
                ) #2계층이후 

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='real')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='real')

    def forward(self, mag):
        hx = mag[:, 1:].permute(0, 2, 1) # mag 슬라이싱 및 차원 순서 변경

        out = hx
        for idx, layer in enumerate(self.encoder):
            out = layer(out) # 인코더 레이어 적용

        for idx, layer in enumerate(self.decoder):
            out = layer(out)# 디코더 레이어 적용

        mag_out = functional.pad(out, [1, 0, 0, 0]).permute(0, 2, 1) # 출력을 패딩 및 차원 순서 변경
        return mag_out

#MagMapping과 유사한 구조를 가짐 
#마스킹 접근법을 사용하여 오디오 신호의 진폭 스펙트럼을 수정.
#return에서 반환하는 값만 다름 
class MagMasking(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, hidd_dim, norm, act):
        super().__init__(win_len, hop_len, fft_len, hidd_dim, norm, act)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_dim)):
            if idx == 0:
                self.encoder.append(
                    fcLayer(self.fft_half_len, hidd_dim[idx], norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    fcLayer(hidd_dim[idx-1], hidd_dim[idx], norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_dim), 0, -1):
            if idx == 1:
                self.decoder.append(
                    fcLayer(hidd_dim[idx-1], self.fft_half_len, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    fcLayer(hidd_dim[idx-1], hidd_dim[idx-2], norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='real')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='real')

    def forward(self, mag):
        hx = mag[:, 1:].permute(0, 2, 1)

        out = hx
        for idx, layer in enumerate(self.encoder):
            out = layer(out)

        for idx, layer in enumerate(self.decoder):
            out = layer(out)

        mask_mag_out = functional.pad(out, [1, 0, 0, 0]).permute(0, 2, 1)
        mag_out = mag * mask_mag_out  # 마스크된 mag 계산
        return mag_out

#복소수 연산(real 및 imag 부분)을 다루는 클래스.
#복소수 값 스펙트럼에 대한 매핑 및 마스킹 연산을 수행.
#'complex' 특성 유형을 사용하여 
# 복소수 값 변환을 위해 ConvSTFT와 ConviSTFT를 사용.
class ComplexOperationMapping(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, hidd_dim, norm, act):
        super().__init__(win_len, hop_len, fft_len, hidd_dim, norm, act)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_dim)):
            if idx == 0:
                self.encoder.append(
                    complexFcLayer(self.fft_half_len, hidd_dim[idx], norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    complexFcLayer(hidd_dim[idx-1], hidd_dim[idx], norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_dim), 0, -1):
            if idx == 1:
                self.decoder.append(
                    complexFcLayer(hidd_dim[idx-1], self.fft_half_len, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    complexFcLayer(hidd_dim[idx-1], hidd_dim[idx-2], norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')

    def forward(self, real_imag):
        real = real_imag[0][:, 1:].permute(0, 2, 1)  # 실수 부분 슬라이싱 및 차원 순서 변경
        imag = real_imag[1][:, 1:].permute(0, 2, 1) #  허수 부분 슬라이싱 및 차원 순서 변경

        real_out, imag_out = real, imag
        for idx, layer in enumerate(self.encoder):
            real_out, imag_out = layer(real_out, imag_out)  # 복소수 인코더 레이어 적용

        for idx, layer in enumerate(self.decoder):
            real_out, imag_out = layer(real_out, imag_out)  # 복소수 인코더 레이어 적용

        real_out = functional.pad(real_out, [1, 0, 0, 0]).permute(0, 2, 1) # 출력을 패딩 및 차원 순서 변경
        imag_out = functional.pad(imag_out, [1, 0, 0, 0]).permute(0, 2, 1) # 출력을 패딩 및 차원 순서 변경
        mag_out = torch.sqrt(real_out ** 2 + imag_out ** 2) # 복소수 출력의 크기 계산
        return real_out, imag_out, mag_out

# ComplexOperationMapping과 동일
class ComplexOperationMasking(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, hidd_dim, norm, act):
        super().__init__(win_len, hop_len, fft_len, hidd_dim, norm, act)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_dim)):
            if idx == 0:
                self.encoder.append(
                    complexFcLayer(self.fft_half_len, hidd_dim[idx], norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    complexFcLayer(hidd_dim[idx-1], hidd_dim[idx], norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_dim), 0, -1):
            if idx == 1:
                self.decoder.append(
                    complexFcLayer(hidd_dim[idx-1], self.fft_half_len, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    complexFcLayer(hidd_dim[idx-1], hidd_dim[idx-2], norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')  # 복소수 특성으로 STFT 초기화
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')  # 복소수 특성으로 iSTFT 초기화

    def forward(self, real_imag):
        real, imag = real_imag[0], real_imag[1]  # 실수 및 허수 부분
        hx_real = real[:, 1:].permute(0, 2, 1) # 실수 부분 슬라이싱 및 차원 순서 변경
        hx_imag = imag[:, 1:].permute(0, 2, 1)# 실수 부분 슬라이싱 및 차원 순서 변경

        real_mask_out, imag_mask_out = hx_real, hx_imag 
        for idx, layer in enumerate(self.encoder):
            real_mask_out, imag_mask_out = layer(real_mask_out, imag_mask_out)

        for idx, layer in enumerate(self.decoder):
            real_mask_out, imag_mask_out = layer(real_mask_out, imag_mask_out)

        real_mask_out = functional.pad(real_mask_out, [1, 0, 0, 0]).permute(0, 2, 1) # 실수 부분 출력 패딩 및 차원 순서 변경
        imag_mask_out = functional.pad(imag_mask_out, [1, 0, 0, 0]).permute(0, 2, 1) # 허수 부분 출력 패딩 및 차원 순서 변경

        real_out = real * real_mask_out
        imag_out = imag * imag_mask_out
        mag_out = torch.sqrt(real_out ** 2 + imag_out ** 2) # 복소수 출력의 크기 계산
        return real_out, imag_out, mag_out

#이 클래스들은 복소수 오디오 데이터의 채널별 변환에 집중.
#real_imag[0], real_imag[1]을 연결한 후 
#(real, imag 부분), fully connected 레이어(fcLayer)를 적용.
class ComplexChannelMapping(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, hidd_dim, norm, act):
        super().__init__(win_len, hop_len, fft_len, hidd_dim, norm, act)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_dim)):
            if idx == 0:
                self.encoder.append(
                    fcLayer(self.fft_len, hidd_dim[idx], norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    fcLayer(hidd_dim[idx-1], hidd_dim[idx], norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_dim), 0, -1):
            if idx == 1:
                self.decoder.append(
                    fcLayer(hidd_dim[idx-1], self.fft_len, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    fcLayer(hidd_dim[idx-1], hidd_dim[idx-2], norm=self.norm, act=self.act)
                )
        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')

    def forward(self, real_imag):
        real = real_imag[0][:, 1:].permute(0, 2, 1)
        imag = real_imag[1][:, 1:].permute(0, 2, 1)

        hx = torch.cat([real, imag], dim=2)

        out = hx
        for idx, layer in enumerate(self.encoder):
            out = layer(out)

        for idx, layer in enumerate(self.decoder):
            out = layer(out)

        real_out = out[..., :self.fft_half_len]
        imag_out = out[..., self.fft_half_len:]

        real_out = functional.pad(real_out, [1, 0, 0, 0]).permute(0, 2, 1)
        imag_out = functional.pad(imag_out, [1, 0, 0, 0]).permute(0, 2, 1)
        mag_out = torch.sqrt(real_out ** 2 + imag_out ** 2)
        return real_out, imag_out, mag_out

# ComplexChannelMapping와 동일한 목적 
class ComplexChannelMasking(EnhancementStrategy):
    def __init__(self, win_len, hop_len, fft_len, hidd_dim, norm, act):
        super().__init__(win_len, hop_len, fft_len, hidd_dim, norm, act)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(hidd_dim)):
            if idx == 0:
                self.encoder.append(
                    fcLayer(self.fft_len, hidd_dim[idx], norm=self.norm, act=self.act)
                )
            else:
                self.encoder.append(
                    fcLayer(hidd_dim[idx-1], hidd_dim[idx], norm=self.norm, act=self.act)
                )

        for idx in range(len(hidd_dim), 0, -1):
            if idx == 1:
                self.decoder.append(
                    fcLayer(hidd_dim[idx-1], self.fft_len, norm=self.norm, act=self.act)
                )
            else:
                self.decoder.append(
                    fcLayer(hidd_dim[idx-1], hidd_dim[idx-2], norm=self.norm, act=self.act)
                )

        # for feature extract
        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, feature_type='complex')

    def forward(self, real_imag):
        real, imag = real_imag[0], real_imag[1]
        hx_real = real[:, 1:].permute(0, 2, 1)
        hx_imag = imag[:, 1:].permute(0, 2, 1)

        hx = torch.cat([hx_real, hx_imag], dim=2)

        out = hx
        for idx, layer in enumerate(self.encoder):
            out = layer(out)

        for idx, layer in enumerate(self.decoder):
            out = layer(out)

        real_mask_out = out[..., :self.fft_half_len]
        imag_mask_out = out[..., self.fft_half_len:]

        real_mask_out = functional.pad(real_mask_out, [1, 0, 0, 0]).permute(0, 2, 1)
        imag_mask_out = functional.pad(imag_mask_out, [1, 0, 0, 0]).permute(0, 2, 1)

        real_out = real * real_mask_out
        imag_out = imag * imag_mask_out
        mag_out = torch.sqrt(real_out ** 2 + imag_out ** 2)
        return real_out, imag_out, mag_out

"""
이 클래스는 다양한 향상 전략(MagMapping, MagMasking 등)을 감싸고 있음.
processing_type 매개변수('mag_mapping', 'mag_masking' 등)에 따라 해당하는 향상 전략을 초기화하며, 
필요한 경우에 따라 전략을 변경가능.
단시간 푸리에 변환 작업을 처리하기 위해 stft와 istft 속성을 포함.
"""
class ED_FNN(nn.Module):
    def __init__(self, win_len=400, hop_len=100, fft_len=512, hidd_dim=[256, 512, 768, 768], processing_type='mag_mapping', norm='bn', act='relu'):
        super().__init__()

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.fft_half_len = fft_len // 2
        self.norm = norm
        self.act = act

        # Initialize different strategies
        self.strategies = {
            'mag_mapping': MagMapping(win_len, hop_len, fft_len, hidd_dim, norm, act), # magnitude 매핑 전략
            'mag_masking': MagMasking(win_len, hop_len, fft_len, hidd_dim, norm, act),  # magnitude 마스킹 전략
            'complex_operation_mapping': ComplexOperationMapping(win_len, hop_len, fft_len, hidd_dim, norm, act), # 복소수 연산 매핑 전략
            'complex_operation_masking': ComplexOperationMasking(win_len, hop_len, fft_len, hidd_dim, norm, act), # 복소수 연산 마스킹 전략
            'complex_channel_mapping': ComplexChannelMapping(win_len, hop_len, fft_len, hidd_dim, norm, act), # 복소수 채널 매핑 전략
            'complex_channel_masking': ComplexChannelMasking(win_len, hop_len, fft_len, hidd_dim, norm, act) # 복소수 채널 마스킹 전략
        }
        self.current_strategy = self.strategies[processing_type] # 선택된 전략
        self.stft = self.current_strategy.stft # STFT 연산
        self.istft = self.current_strategy.istft  # iSTFT 연산

    def forward(self, x):
        return self.current_strategy(x) # 선택된 전략으로 순방향 전파

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.utils.data import DataLoader
from utils import ConvSTFT, ConviSTFT
from .baseBlocks import *
from options import *
from utils import get_normalization_layer, get_activation_layer

#nn.Module 상속, 같이 사용하는 매개변수 초기화 
class EnhancementStrategy(nn.Module):
    def __init__(self, window_size=256, hop_size=128, fft_size=512, hidden_size=128, 
                 norm_fn='batchnorm', activation='relu'):
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.hidden_size = hidden_size

        # Create STFT and inverse STFT modules
        self.stft = ConvSTFT(window_size, hop_size, fft_size)
        self.istft = ConviSTFT(window_size, hop_size, fft_size)

        # Normalize input features
        if norm_fn == 'batchnorm':
            self.norm = nn.BatchNorm1d(hidden_size)
        elif norm_fn == 'layernorm':
            self.norm = nn.LayerNorm(hidden_size)
        else:
            self.norm = None

        # Activation function
        if activation == 'relu':
            self.act = F.relu
        elif activation == 'tanh':
            self.act = F.tanh
        else:
            self.act = None

    def forward(self, x):
        # Convert audio signal to complex spectrum
        X = self.stft(x)

        # Apply enhancement strategy to complex spectrum
        enhanced_X = self.enhance(X)

        # Reconstruct audio signal from enhanced spectrum
        x_hat = self.istft(enhanced_X)

        return x_hat

class MagMapping(EnhancementStrategy):
    def __init__(self, window_size=256, hop_size=128, fft_size=512, hidden_size=128, num_layers=4, 
                 norm_fn='batchnorm', activation='relu'):
        super().__init__(window_size, hop_size, fft_size, hidden_size, norm_fn, activation)

        # Encoder and decoder networks
        self.encoder = nn.Sequential(*[
            Conv1d(fft_size // 2, hidden_size, kernel_size=1),
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

        self.decoder = nn.Sequential(*[
            Conv1d(hidden_size, fft_size // 2, kernel_size=1),
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

    def enhance(self, X):
        # Extract magnitude spectrum
        mag = torch.abs(X)

        # Encode magnitude spectrum
        encoded_mag = self.encoder(mag.transpose(1, 2))

        # Decode encoded magnitude
        enhanced_mag = self.decoder(encoded_mag).transpose(1, 2)

        # Combine enhanced magnitude with phase
        enhanced_X = X * torch.polar(enhanced_mag, torch.angle(X))

        return enhanced_X
    
#MagMapping과 유사한 구조를 가짐 
#마스킹 접근법을 사용하여 오디오 신호의 진폭 스펙트럼을 수정.
#return에서 반환하는 값만 다름 
class MagMasking(EnhancementStrategy):
    def __init__(self, window_size=256, hop_size=128, fft_size=512, hidden_size=128, num_layers=4, 
                 norm_fn='batchnorm', activation='relu'):
        super().__init__(window_size, hop_size, fft_size, hidden_size, norm_fn, activation)

        # Encoder and decoder networks
        self.encoder = nn.Sequential(*[
            Conv1d(fft_size // 2, hidden_size, kernel_size=1),
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

        self.decoder = nn.Sequential(*[
            Conv1d(hidden_size, fft_size // 2, kernel_size=1),
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

    def enhance(self, X):
        # Extract magnitude spectrum
        mag = torch.abs(X)

        # Encode magnitude spectrum
        encoded_mag = self.encoder(mag.transpose(1, 2))

        # Decode encoded mask
        mask = self.decoder(encoded_mag).transpose(1, 2)
        mask = torch.sigmoid(mask)  # Apply sigmoid to get values between 0 and 1

        # Apply learned mask to magnitude spectrum
        enhanced_mag = mag * mask

        # Combine enhanced magnitude with phase
        enhanced_X = X * torch.polar(enhanced_mag, torch.angle(X))

        return enhanced_X

#복소수 연산(real 및 imag 부분)을 다루는 클래스.
#복소수 값 스펙트럼에 대한 매핑 및 마스킹 연산을 수행.
#'complex' 특성 유형을 사용하여 
# 복소수 값 변환을 위해 ConvSTFT와 ConviSTFT를 사용.
class ComplexOperationMapping(EnhancementStrategy):
    def __init__(self, window_size=256, hop_size=128, fft_size=512, hidden_size=128, num_layers=4, 
                 norm_fn='batchnorm', activation='relu'):
        super().__init__(window_size, hop_size, fft_size, hidden_size, norm_fn, activation)

        # Encoder and decoder networks
        self.encoder = nn.Sequential(*[
            Conv1d(fft_size, hidden_size, kernel_size=1, groups=2),  # Complex convolution
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

        self.decoder = nn.Sequential(*[
            Conv1d(hidden_size, fft_size, kernel_size=1, groups=2),  # Complex convolution
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

    def enhance(self, X):
        # Convert to complex spectrum
        X_complex = X.view(X.shape[0], X.shape[1], 2)  # Split real and imaginary parts

        # Apply enhancement strategy to complex spectrum
        enhanced_X_complex = self.enhance_complex(X_complex)

        # Combine enhanced complex spectrum
        enhanced_X = torch.cat((enhanced_X_complex[..., 0], enhanced_X_complex[..., 1]), dim=-1)

        return enhanced_X

    def enhance_complex(self, X_complex):
        # Encode complex spectrum
        encoded_complex = self.encoder(X_complex.transpose(1, 2))

        # Decode encoded complex spectrum
        enhanced_complex = self.decoder(encoded_complex).transpose(1, 2)

        return enhanced_complex
    
# ComplexOperationMapping과 동일
class ComplexOperationMasking(EnhancementStrategy):
    def __init__(self, window_size=256, hop_size=128, fft_size=512, hidden_size=128, num_layers=4, 
                 norm_fn='batchnorm', activation='relu'):
        super().__init__(window_size, hop_size, fft_size, hidden_size, norm_fn, activation)

        # Encoder and decoder networks
        self.encoder = nn.Sequential(*[
            Conv1d(fft_size, hidden_size, kernel_size=1, groups=2),  # Complex convolution
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

        self.decoder = nn.Sequential(*[
            Conv1d(hidden_size, fft_size, kernel_size=1, groups=2),  # Complex convolution
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

    def enhance(self, X):
        # Convert to complex spectrum
        X_complex = X.view(X.shape[0], X.shape[1], 2)  # Split real and imaginary parts

        # Apply enhancement strategy to complex spectrum
        enhanced_X_complex = self.enhance_complex(X_complex)

        # Combine enhanced complex spectrum
        enhanced_X = torch.cat((enhanced_X_complex[..., 0], enhanced_X_complex[..., 1]), dim=-1)

        return enhanced_X

    def enhance_complex(self, X_complex):
        # Encode complex spectrum
        encoded_complex = self.encoder(X_complex.transpose(1, 2))

        # Decode encoded mask
        mask = self.decoder(encoded_complex).transpose(1, 2)
        mask = torch.sigmoid(mask)  # Apply sigmoid to get values between 0 and 1

        # Apply learned mask to complex spectrum
        enhanced_complex = X_complex * mask

        return enhanced_complex
    
    
#이 클래스들은 복소수 오디오 데이터의 채널별 변환에 집중.
#real_imag[0], real_imag[1]을 연결한 후 
#(real, imag 부분), fully connected 레이어(fcLayer)를 적용.
class ComplexChannelMapping(EnhancementStrategy):
    def __init__(self, window_size=256, hop_size=128, fft_size=512, hidden_size=128, num_layers=4, 
                 norm_fn='batchnorm', activation='relu'):
        super().__init__(window_size, hop_size, fft_size, hidden_size, norm_fn, activation)

        # Encoder and decoder networks
        self.encoder = nn.Sequential(*[
            Conv1d(fft_size, hidden_size, kernel_size=1),  # Complex convolution
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

        self.decoder = nn.Sequential(*[
            Conv1d(hidden_size, fft_size, kernel_size=1),  # Complex convolution
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

    def enhance(self, X):
        # Convert to complex spectrum
        X_complex = X.view(X.shape[0], X.shape[1], 2)  # Split real and imaginary parts

        # Apply enhancement strategy to complex spectrum
        enhanced_X_complex = self.enhance_complex(X_complex)

        # Combine enhanced complex spectrum
        enhanced_X = torch.cat((enhanced_X_complex[..., 0], enhanced_X_complex[..., 1]), dim=-1)

        return enhanced_X

    def enhance_complex(self, X_complex):
        # Encode complex spectrum
        encoded_complex = self.encoder(X_complex.transpose(1, 2))

        # Decode encoded complex spectrum
        enhanced_complex = self.decoder(encoded_complex).transpose(1, 2)

        return enhanced_complex

# ComplexChannelMapping와 동일한 목적 
class ComplexChannelMasking(EnhancementStrategy):
    def __init__(self, window_size=256, hop_size=128, fft_size=512, hidden_size=128, num_layers=4, 
                 norm_fn='batchnorm', activation='relu'):
        super().__init__(window_size, hop_size, fft_size, hidden_size, norm_fn, activation)

        # Encoder and decoder networks
        self.encoder = nn.Sequential(*[
            Conv1d(fft_size, hidden_size, kernel_size=1),  # Complex convolution
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

        self.decoder = nn.Sequential(*[
            Conv1d(hidden_size, fft_size, kernel_size=1),  # Complex convolution
            get_normalization_layer(norm_fn, hidden_size),
            get_activation_layer(activation)
        ] * num_layers)

    def enhance(self, X):
        # Convert to complex spectrum
        X_complex = X.view(X.shape[0], X.shape[1], 2)  # Split real and imaginary parts

        # Apply enhancement strategy to complex spectrum
        enhanced_X_complex = self.enhance_complex(X_complex)

        # Combine enhanced complex spectrum
        enhanced_X = torch.cat((enhanced_X_complex[..., 0], enhanced_X_complex[..., 1]), dim=-1)

        return enhanced_X

    def enhance_complex(self, X_complex):
        # Encode complex spectrum
        encoded_complex = self.encoder(X_complex.transpose(1, 2))

        # Decode encoded mask
        mask = self.decoder(encoded_complex).transpose(1, 2)
        mask = torch.sigmoid(mask)  # Apply sigmoid to get values between 0 and 1

        # Apply learned mask to complex spectrum
        enhanced_complex = X_complex * mask

        return enhanced_complex


#이 클래스는 다양한 향상 전략(MagMapping, MagMasking 등)을 감싸고 있음.
#processing_type 매개변수('mag_mapping', 'mag_masking' 등)에 따라 해당하는 향상 전략을 초기화하며, 
#필요한 경우에 따라 전략을 변경가능.
#단시간 푸리에 변환 작업을 처리하기 위해 stft와 istft 속성을 포함.

class ED_FNN(nn.Module):
    def __init__(self, win_len=400, hop_len=100, fft_len=512, hidd_dim=[256, 512, 768, 768], processing_type='mag_mapping', norm='bn', act='relu'):
        super().__init__()

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.fft_half_len = fft_len // 2
        self.norm = norm
        self.act = act

        # Initialize different strategies
        self.strategies = {
            'mag_mapping': MagMapping(win_len, hop_len, fft_len, hidd_dim, norm, act),  # magnitude 매핑 전략
            'mag_masking': MagMasking(win_len, hop_len, fft_len, hidd_dim, norm, act),  # magnitude 마스킹 전략
            'complex_operation_mapping': ComplexOperationMapping(win_len, hop_len, fft_len, hidd_dim, norm, act),  # 복소수 연산 매핑 전략
            'complex_operation_masking': ComplexOperationMasking(win_len, hop_len, hop_len, fft_len, hidd_dim, norm, act),  # 복소수 연산 마스킹 전략
            'complex_channel_mapping': ComplexChannelMapping(win_len, hop_len, fft_len, hidd_dim, norm, act),  # 복소수 채널 매핑 전략
            'complex_channel_masking': ComplexChannelMasking(win_len, hop_len, fft_len, hidd_dim, norm, act)  # 복소수 채널 마스킹 전략
        }

         # Select the specified processing type
        self.current_strategy = self.strategies[processing_type]

        # Initialize STFT and iSTFT modules
        self.stft = ConvSTFT(win_len, hop_len, fft_len)
        self.istft = ConviSTFT(win_len, hop_len, fft_len)

    def forward(self, x):
        # Convert audio signal to complex spectrum
        X = self.stft(x)

        # Apply enhancement strategy to complex spectrum
        enhanced_X = self.current_strategy(X)

        # Reconstruct audio signal from enhanced spectrum
        x_hat = self.istft(enhanced_X)

        return x_hat
"""