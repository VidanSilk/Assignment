import os
import numpy as np
import soundfile


def scan_directory(dir_name):
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()

    addrs = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".wav"):
                filepath = subdir + file
                addrs.append(filepath)
    return addrs


def find_pair(noisy_file_name):
    clean_dirs = []
    for i in range(len(noisy_file_name)):
        addrs = noisy_file_name[i]
        if addrs.endswith(".wav"):
            clean_addrs = str(addrs).replace('noisy', 'clean')
            clean_dirs.append(clean_addrs)
    return clean_dirs


def addr2wav(noisy_addr, clean_addr, fs_set=16000, norm_type='None'):
    noisy_wav, fs = soundfile.read(noisy_addr)
    clean_wav, _ = soundfile.read(clean_addr)
    assert fs == fs_set
    noisy_wav, clean_wav = normalize_wav(noisy_wav, clean_wav, norm_type)
    return noisy_wav, clean_wav


# make a new dir
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("Already exist...")


def normalize_wav(noisy_wav, clean_wav, norm_type='min_max'):
    if norm_type == 'None':
        noisy_norm, clean_norm = noisy_wav, clean_wav
    elif norm_type == 'MinMax':
        min_val = min(np.min(clean_wav), np.min(noisy_wav))
        max_val = max(np.max(clean_wav), np.max(noisy_wav))
        clean_norm = (clean_wav - min_val) / (max_val - min_val)
        noisy_norm = (noisy_wav - min_val) / (max_val - min_val)
    elif norm_type == 'Zscore':
        mean_val = np.mean(np.concatenate((clean_wav, noisy_wav)))
        std_val = np.std(np.concatenate((clean_wav, noisy_wav)))
        clean_norm = (clean_wav - mean_val) / std_val
        noisy_norm = (noisy_wav - mean_val) / std_val
    elif norm_type == 'MaxAbs':
        max_abs_val = max(np.max(np.abs(clean_wav)), np.max(np.abs(noisy_wav)))
        clean_norm = clean_wav / max_abs_val
        noisy_norm = noisy_wav / max_abs_val
    elif norm_type == 'Robust':
        median_val = np.median(np.concatenate((clean_wav, noisy_wav)))
        iqr = np.percentile(np.concatenate((clean_wav, noisy_wav)), 75) - np.percentile(np.concatenate((clean_wav, noisy_wav)), 25)
        clean_norm = (clean_wav - median_val) / iqr
        noisy_norm = (noisy_wav - median_val) / iqr
    else:
        raise ValueError("Unknown normalization method specified")

    return noisy_norm, clean_norm
