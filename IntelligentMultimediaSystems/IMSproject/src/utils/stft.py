import torch
import torch.nn as nn
import torch.nn.functional as functional
from scipy.signal import get_window
import numpy as np


######################################################################################################################
# this is from conv_stft https://github.com/huyanxin/DeepComplexCRN
######################################################################################################################
def init_kernels(win_len, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real'):
        super(ConvSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, self.fft_len, win_type)
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = functional.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = functional.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase  # , real, imag


class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real'):
        super(ConviSTFT, self).__init__()
        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, self.fft_len, win_type, invers=True)
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)

        outputs = functional.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = functional.conv_transpose1d(t, self.enframe, stride=self.stride)

        outputs = outputs / (coff + 1e-8)

        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]

        return outputs

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
import numpy as np

# 필요한 함수와 클래스 정의
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def build_mel_filter(sample_rate, n_mels, n_fft):
  mel_min = hz_to_mel(0)
  mel_max = hz_to_mel(sample_rate / 2)
  mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
  hz_points = mel_to_hz(mel_points)
  
  bin = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

  mel_filter_bank = np.zeros((n_mels, n_fft // 2 + 1))

  for i in range(1, n_mels + 1):
    left = bin[i - 1]
    center = bin[i]
    right = bin[i + 1]
    
    for j in range(left, center):
      mel_filter_bank[i - 1, j] = (j - left) / (center - left)
    for j in range(center, right):
      mel_filter_bank[i - 1, j] = (right - j) / (right - center)
  
  return torch.tensor(mel_filter_bank, dtype=torch.float32)

def init_kernels(win_len, fft_len, win_type=None, invers=False):
  if win_type == 'None' or win_type is None:
    window = np.ones(win_len)
  else:
    window = get_window(win_type, win_len, fftbins=True)

  N = fft_len
  fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
  real_kernel = np.real(fourier_basis)
  imag_kernel = np.imag(fourier_basis)
  kernel = np.concatenate([real_kernel, imag_kernel], 1).T

  if invers:
    kernel = np.linalg.pinv(kernel).T

  kernel = kernel * window
  kernel = kernel[:, None, :]
  return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))

class ConvSTFT(nn.Module):

  def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real'):
    super(ConvSTFT, self).__init__()

    if fft_len is None:
      self.fft_len = int(2 ** np.ceil(np.log2(win_len)))
    else:
      self.fft_len = fft_len

    kernel, _ = init_kernels(win_len, self.fft_len, win_type)
    self.register_buffer('weight', kernel)
    self.feature_type = feature_type
    self.stride = win_inc
    self.win_len = win_len
    self.dim = self.fft_len

  def forward(self, inputs):
    if inputs.dim() == 2:
        inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase

class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real'):
        super(ConviSTFT, self).__init__()
        if fft_len is None:
            self.fft_len = int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, self.fft_len, win_type, invers=True)
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)

        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)

        outputs = outputs / (coff + 1e-8)

        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]

        return outputs

class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate, n_mels, win_len, win_inc, fft_len=None, win_type='hamming'):
        super(MelSpectrogram, self).__init__()
        self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type, feature_type='real')
        mel_filter = self._build_mel_filter(sample_rate, n_mels, self.stft.dim // 2 + 1)
        self.register_buffer('mel_filter', mel_filter)

    def _build_mel_filter(self, sample_rate, n_mels, n_fft_bins):
        mel_min = hz_to_mel(0)
        mel_max = hz_to_mel(sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        bin = np.floor((n_fft_bins - 1) * hz_points / (sample_rate / 2)).astype(int)

        mel_filter_bank = np.zeros((n_mels, n_fft_bins))

        for i in range(1, n_mels + 1):
            left = bin[i - 1]
            center = bin[i]
            right = bin[i + 1]
            
            for j in range(left, center):
                mel_filter_bank[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                mel_filter_bank[i - 1, j] = (right - j) / (right - center)
        
        return torch.tensor(mel_filter_bank, dtype=torch.float32)

    def forward(self, inputs):
        mags, _ = self.stft(inputs)
        mel_spec = torch.matmul(self.mel_filter, mags)
        return mel_spec

# Mel-Spectrogram을 입력 데이터로 사용하기 위한 예제
class ExampleModel(nn.Module):
  def __init__(self, sample_rate, n_mels, win_len, win_inc, fft_len=None, win_type='hamming'):
    super(ExampleModel, self).__init__()
    self.mel_spectrogram = MelSpectrogram(sample_rate, n_mels, win_len, win_inc, fft_len, win_type)
    self.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.fc = nn.Linear(258560, 10)  # Updated line to match actual input dimension

  def forward(self, inputs):
    mel_spec = self.mel_spectrogram(inputs).unsqueeze(1)  # (B, 1, n_mels, T)
    x = self.conv(mel_spec)
    x = F.relu(x)
    x = x.view(x.size(0), -1)  # Flatten
    x = self.fc(x)
    return x

# 사용 예시
sample_rate = 16000
n_mels = 80
win_len = 400
win_inc = 160
fft_len = 512
win_type = 'hamming'

model = ExampleModel(sample_rate, n_mels, win_len, win_inc, fft_len, win_type)
inputs = torch.randn(1, 16000)  # 예제 입력 파형
outputs = model(inputs)
print(outputs.shape)  # (1, 10)
"""
