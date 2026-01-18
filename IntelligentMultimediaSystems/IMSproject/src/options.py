"""
Docstring for Options
"""

class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=36, help='batch size') # 배치 사이즈 수정해보기?  36 -> 64 
        parser.add_argument('--nepoch', type=int, default=10, help='training epochs') #epochs크기 수정 
        parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer for training') # 최적화 알고리즘을 수정해본다면? 
        parser.add_argument('--lr_initial', type=float, default=5e-4, help='initial learning rate') #초기 학습률 수정 10e-4로 해본다면? 
        parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay") #학습률 조절 
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay') # 과적합 

        # train settings
        parser.add_argument('--arch', type=str, default='ED_FNN', help='architecture')  #아키텍쳐 설정 fnn or cnn 
        parser.add_argument('--disc', type=str, default=False, help='discriminator')
        parser.add_argument('--loss_type', type=str, default='base', help='type: [base, joint]')
        parser.add_argument('--loss_oper', type=str, default='l2', help='loss function operation type') #손실함수 연산 유형 기본 l2, l1으로한다면?
        parser.add_argument('--c', type=list, default=[0.1, 0.9, 0.5, 0.5], help='coupling constant') #결합 함수로 가중치 값을 바꿔본다면? 0.1, 0.9, 0.5, 0.5
        parser.add_argument('--device', type=str, default='cuda', help='gpu or cpu')
        parser.add_argument('--input_type', type=str, default='mag',
                            help='type: [mag, complex_operation, complex_channel]')
        parser.add_argument('--target_type', type=str, default='masking', help='type: [mapping, masking]')

        # network settings
        parser.add_argument('--hidd_ch', type=list, default=[32, 64, 128, 256], help='channel size for hidden dim') # 히든층  수정함 32 -> 64 128
        parser.add_argument('--norm_layer', type=str, default='bn', help='type: [bn, in, gn, ln]')
        parser.add_argument('--act', type=str, default='relu', help='type: [sigmoid, relu, leakyrelu, prelu, tanh]')
        parser.add_argument('--kernel_size', type=tuple, default=(3, 2), help='kernel size for a convolutional layer') #커널 사이즈를 조절 해본다면? 
        parser.add_argument('--stride', type=tuple, default=(2, 1), help='stride size for a convolutional layer') # 필터를 얼마나 이동 시킬까? 
        parser.add_argument('--dilation', type=tuple, default=(1, 1), help='dilation size for a convolutional layer')

        # pretrained
        parser.add_argument('--env', type=str, default='base', help='log name')
        parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained_weights')
        parser.add_argument('--pretrained_init', type=bool, default=False, help='load pretrained_weights')
        parser.add_argument('--pretrain_model_path', type=str, default='D:/YonseiCode/IMS Project data set/240529 IMS Project/SE_Tutorials-main/src/log/ED_FNN_616_base/models/chkpt_1.pt',
                            help='path of pretrained_weights') #test에서 경로 위치 수정  /logs/Base/models/chkpt_1.pt 
        parser.add_argument('--test_name', type=str, default='Base', help='wave write')
        parser.add_argument('--wav_write_flag', type=bool, default=True, help='wave write')

        # dataset
        parser.add_argument('--database', type=str, default='VBD', help='database')
        parser.add_argument('--fft_len', type=int, default=512, help='fft length') # 주파수의 길이로 주파수 해상도를 조절한다면?
        parser.add_argument('--win_len', type=int, default=400, help='window length')
        parser.add_argument('--hop_len', type=int, default=100, help='hop length') #계산 비용 
        parser.add_argument('--fs', type=int, default= 16000, help='sampling frequency') #샘플링 값을 바꿔 본다면? 16000
        parser.add_argument('--chunk_size', type=int, default=64000, help='chunk size') #청크 값을 바꿔본다면?  16000 32000 64000
        parser.add_argument('--norm_type', type=str, default='None',
                            help='normalization type: [None, MinMax, Zscore, MaxAbs, Robust]')
        parser.add_argument('--cmpr', type=float, default=0.2, help='compression factor') #압축 비율을 바꾸면? 높으면 손실율 일어남

        parser.add_argument('--noisy_dirs_for_train', type=str,
                            default='../Dataset/VBD_V2/train/noisy/',
                            # default='../test_data/train/noisy/',
                            help='noisy dataset addr for train')
        parser.add_argument('--noisy_dirs_for_valid', type=str,
                            default='../Dataset/VBD_V2/test/noisy/',
                            # default='../test_data/test/noisy/',
                            help='noisy dataset addr for valid')
        parser.add_argument('--noisy_dirs_for_test', type=str,
                            default='../test_data/test/noisy/',
                            help='noisy dataset addr for test') # 파일만듬 

        return parser
