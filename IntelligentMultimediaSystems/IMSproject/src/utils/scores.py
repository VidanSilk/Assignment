from pesq import pesq
from pystoi import stoi
from joblib import Parallel, delayed
import numpy as np

"""
cal_pesq 함수는 두 개의 오디오 신호(clean_wav와 dirty_wav)에 대해 PESQ 점수를 계산합니다.
pesq 함수는 PESQ 점수를 계산하기 위해 사용됩니다. pesq(FS, clean_wav, dirty_wav, "wb")에서 FS는 샘플링 주파수를 나타내며, "wb"는 wideband 모드를 의미합니다.
예외가 발생할 경우(예: 오디오 신호가 없거나 예상치 못한 오류), -1을 반환하고 에러 메시지를 출력합니다.
"""
def cal_pesq(clean_wav, dirty_wav, FS=16000):
    try:
        pesq_score = pesq(FS, clean_wav, dirty_wav, "wb")
    except:
        print(' No utterances error')
        pesq_score = -1
    return pesq_score

"""
cal_pesq_batch 함수는 배치로 여러 개의 clean_wav와 dirty_wav 쌍에 대해 평균 PESQ 점수를 계산합니다.
Parallel(n_jobs=1)(delayed(cal_pesq)(c, n, FS=FS) for c, n in zip(clean_wavs, dirty_wavs)) 구문은 병렬로 각 clean_wav와 dirty_wav 쌍에 대해 cal_pesq 함수를 실행하고, 각각의 PESQ 점수를 리스트로 반환합니다.
이후 np.mean(pesq_score)는 반환된 PESQ 점수들의 평균을 계산하여 최종 평균 PESQ 점수를 반환합니다.
"""
def cal_pesq_batch(clean_wavs, dirty_wavs, FS=16000):
    pesq_score = Parallel(n_jobs=1)(delayed(cal_pesq)(c, n, FS=FS) for c, n in zip(clean_wavs, dirty_wavs))
    pesq_score = np.array(pesq_score)
    return np.mean(pesq_score)

"""
cal_stoi_batch 함수는 배치로 여러 개의 clean_wav와 dirty_wav 쌍에 대해 평균 STOI 점수를 계산합니다.
Parallel(n_jobs=1)(delayed(stoi)(c, n, FS, extended=False) for c, n in zip(clean_wavs, dirty_wavs)) 구문은 병렬로 각 clean_wav와 dirty_wav 쌍에 대해 stoi 함수를 실행하고, 각각의 STOI 점수를 리스트로 반환합니다.
이후 np.mean(stoi_score)는 반환된 STOI 점수들의 평균을 계산하여 최종 평균 STOI 점수를 반환합니다.
"""
def cal_stoi_batch(clean_wavs, dirty_wavs, FS=16000):
    stoi_score = Parallel(n_jobs=1)(delayed(stoi)(c, n, FS, extended=False) for c, n in zip(clean_wavs, dirty_wavs))
    stoi_score = np.array(stoi_score)
    return np.mean(stoi_score)
