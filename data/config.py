cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[10, 20], [32, 64], [128, 256]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}
 
cfg_vgg = {
    'name': 'vgg',
    #'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    #'min_sizes': [[5, 8, 10, 16],[20, 26, 32, 40],[48, 56, 64, 72],[80, 96, 106, 118],[128, 144, 160, 192],[208, 224, 256]],
    #'min_sizes': [[6, 10, 16, 24], [28, 36, 42, 52,], [60, 64, 80, 96], [128, 192, 256]],first verson
    'min_sizes': [[6, 10, 16, 24], [36, 42, 56, 64], [72, 80, 96, 118], [144, 192, 224]], #0520 version
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 2,
    'epoch': 250,
    'decay1': 120,
    'decay2': 200,
    'image_size': 300
}
cfg_shufflenet = {
    'name': 'shufflenet',
    #'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    #'min_sizes': [[5, 8, 10, 16],[20, 26, 32, 40],[48, 56, 64, 72],[80, 96, 106, 118],[128, 144, 160, 192],[208, 224, 256]],
    #'min_sizes': [[6, 10, 16, 24], [28, 36, 42, 52,], [60, 64, 80, 96], [128, 192, 256]],first verson
    'min_sizes': [[6, 10, 16, 24], [36, 42, 56, 64], [72, 80, 96, 118], [144, 192, 224]], #0520 version
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 2,
    'epoch': 250,
    'decay1': 120,
    'decay2': 200,
    'image_size': 300
}
 
 
cfg_efficientdet = {
    'name': 'efficientdet',
    #'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    #'min_sizes': [[5, 8, 10, 16],[20, 26, 32, 40],[48, 56, 64, 72],[80, 96, 106, 118],[128, 144, 160, 192],[208, 224, 256]],
    'min_sizes': [[6, 10, 16, 24], [36, 42, 56, 64], [72, 80, 96, 118], [144, 192, 224]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 120,
    'decay2': 200,
    'image_size': 320
}
 
cfg_rfb = {
    'name': 'RFB',
    'min_sizes': [[6, 10, 16, 24], [36, 42, 56, 64], [72, 80, 96, 118], [144, 192, 224]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 2,
    'epoch': 250,
    'decay1': 120,
    'decay2': 200,
    'image_size': 300
}
 
 
 
cfg_mobilenet = {
    'name': 'mobilenet',
    #'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    #'min_sizes': [[5, 8, 10, 16],[20, 26, 32, 40],[48, 56, 64, 72],[80, 96, 106, 118],[128, 144, 160, 192],[208, 224, 256]],
    'min_sizes': [[6, 10, 16, 24], [36, 42, 56, 64], [72, 80, 96, 118], [144, 192, 224]],
    'steps': [16, 32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 96,
    'ngpu': 2,
    'epoch': 250,
    'decay1': 120,
    'decay2': 200,
    'image_size': 300
}
cfg_slim = {
    'name': 'slim',
    #'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'min_sizes': [[8, 10, 16, 24], [30, 42, 56], [64, 80, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 128,
    'ngpu': 2,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300
}
 
 
cfg_rfb_480 = {
    'name': 'RFB',
    #'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'min_sizes': [[8, 16, 24, 32], [48, 64, 96], [128, 192, 256], [300, 364, 416]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 128,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 120,
    'decay2': 190,
    'image_size': 480
}
 
 
cfg_zhuapai = {
    'name': 'slim',
    #'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'min_sizes': [[5 ,8, 10, 16], [24, 30, 42], [56, 64, 80], [96, 128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 416
}
 
'''
cfg_rfb = {
    'name': 'RFB',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300
}
'''
cfg_lightface = {
    'name': 'lightface',
    #'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'min_sizes': [[8, 10, 16, 24], [30, 42, 56], [64, 80, 96], [128, 192, 256]],
    'steps': [16, 32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 300
}