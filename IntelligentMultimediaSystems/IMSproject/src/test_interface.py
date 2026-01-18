"""
Test interface for speech enhancement!
You can just run this file.
"""
#Test_interface_code
import argparse
import options
import utils
import torch
import random
import numpy as np
import time
from dataloader import create_dataloader
from scipy.io.wavfile import write
#추가
import os
######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
opt = options.Options().init(argparse.ArgumentParser(description='speech enhancement')).parse_args()
print(opt)

######################################################################################################################
#                                                   Model init                                                       #
######################################################################################################################
# set device
DEVICE = torch.device(opt.device)
# set seeds
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
# define model  
model = utils.get_arch(opt)
total_params = utils.cal_total_params(model)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))
# load the params
print('Load the pretrained model...')

# 파일 존재 여부 확인 -추가
if not os.path.exists(opt.pretrain_model_path):
    raise FileNotFoundError(f"Pretrained model not found at {opt.pretrain_model_path}")

chkpt = torch.load(opt.pretrain_model_path)
model.load_state_dict(chkpt['model'])
model = model.to(DEVICE)
######################################################################################################################
######################################################################################################################
#                                             Main program - train                                                   #
######################################################################################################################
######################################################################################################################
print('Test start...')
st_time = time.time()
opt.test_database = opt.noisy_dirs_for_test
test_loader = create_dataloader(opt, mode='test')
data_num = 0
cln_all = []
enh_all = []
# test
model.eval()

if opt.wav_write_flag:
    utils.mkdir('./wav_result/Base_C6.4FNN_ver/{}'.format(opt.test_name)) #모델 마다 다르게 설정 필요 
    utils.mkdir('./wav_result/N_C6.4Fnn_Ver')
    utils.mkdir('./wav_result/C_C6.4Fnn_Ver')

with torch.no_grad():
    for inputs, targets in utils.Bar(test_loader):
        data_num += 1
        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)
        # generator
        input_mags, input_phase = model.stft(inputs)
        input_mags = utils.power_compress_mag(input_mags, compression_factor=opt.cmpr)

        out_mags = model(input_mags)
        out_mags = utils.power_uncompress_mag(out_mags, compression_factor=opt.cmpr)
        out_real = out_mags * torch.cos(input_phase)
        out_imag = out_mags * torch.sin(input_phase)
        out_specs = torch.cat([out_real, out_imag], dim=1)
        outputs = model.istft(out_specs).squeeze(1)

        # get score
        if len(targets[0]) > len(outputs[0]):
            clean_wavs = targets.cpu().detach().numpy()[:, :outputs.size(1)]
            enhanced_wavs = outputs.cpu().detach().numpy()
        else:
            clean_wavs = targets.cpu().detach().numpy()
            enhanced_wavs = outputs.cpu().detach().numpy()[:, :targets.size(1)]

        if opt.wav_write_flag:
            noisy_wavs = inputs.cpu().detach().numpy()[:, :outputs.size(1)]
            write('./wav_result/N_C6.4Fnn_Ver/{}_Ch6.4Fnoisy.wav'.format(data_num), opt.fs, noisy_wavs[0]) #cnn fnn 마다 폴더 다르게  /모델마다 다르게 
            write('./wav_result/C_C6.4Fnn_Ver/{}_Ch6.4Fclean.wav'.format(data_num), opt.fs, clean_wavs[0]) 
            write('./wav_result/Base_C6.4FNN_ver/{}/{}_Ch6.4Fenhanced.wav'.format(opt.test_name, data_num), opt.fs, enhanced_wavs[0])

        cln_all.extend(clean_wavs)
        enh_all.extend(enhanced_wavs)
        del inputs, targets, outputs
        torch.cuda.empty_cache()

# if not opt.wav_write_flag:
test_log_fp = open('./test_log.txt', 'a')

avg_stoi = utils.cal_stoi_batch(cln_all, enh_all)
avg_pesq = utils.cal_pesq_batch(cln_all, enh_all) #수정 cal_mos_batch -> cal_pesq_batch
print('\nTotal score')
print('PESQ: {:.4f}  STOI: {:.4f}'
      .format(avg_pesq, avg_stoi))
print('System has been finished.')

test_log_fp.write(opt.pretrain_model_path + '\n\n')
test_log_fp.write('Total score\n')
test_log_fp.write('PESQ: {:.4f}  STOI: {:.4f}\n'
                  .format(avg_pesq, avg_stoi))

test_log_fp.close()