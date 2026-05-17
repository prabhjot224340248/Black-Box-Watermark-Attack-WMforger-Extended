import os

# This dictionary mirrors the structure of attack_2/CIN/codes/options/opt.yml
# so that the CIN model initialises without any KeyErrors.

CIN_OPT = {

    'train/test': 'test',

    'train': {
        'manual_seed':            None,
        'batch_size':             32,
        'num_workers':            0,
        'epoch':                  1,
        'set_start_epoch':        1,
        'start_step':             1,
        'logs_per_step':          20,
        'logTest_per_step':       5,
        'val':                    {'per_epoch': 5, 'logs_per_step': 10},
        'checkpoint_per_epoch':   1,
        'saveTrainImgs_per_step': 100,
        'saveValImgs_in_step':    1,
        'saveTestImgs_per_step':  100,
        'saveStacked':            True,
        'saveFormat':             '.jpg',
        'msg_save':               False,
        'noise_list':             ['None'],
        'resume': {
            'one_pth':  True,
            'Partial':  True,
            'Empty':    None,
        },
        'os_environ': '-1',
    },

    'lr': {
        'start_lr':   0.0001,
        'milestones': [3000, 50000, 1000000],
        'optimizer':  'Adam',
        'gamma':      0.1,
    },

    'noise': {
        'StrengthFactor': {'S': 1.0},
        'option':         'GaussianBlur',
        'Identity':       {},
        'GaussianBlur':   {'kernel_sizes': 7, 'sigmas': 2},
        'Salt_Pepper':    {'snr': 0.9, 'p': 1.0},
        'GaussianNoise':  {'mean': 0, 'variance': 1, 'amplitude': 0.25, 'p': 1},
        'Resize':         {'p': 0.5, 'interpolation_method': 'nearest'},
        'Jpeg':           {'differentiable': True, 'Q': 50, 'subsample': 2},
        'JpegTest':       {'differentiable': True, 'Q': 50, 'subsample': 2},
        'Dropout':        {'p': 0.3},
        'Cropout':        {'p': 0.3},
        'Crop':           {'p': 0.035},
        'Brightness':     {'f': 2},
        'Contrast':       {'f': 2},
        'Saturation':     {'f': 2},
        'Hue':            {'f': 0.1},
        'Rotation':       {'degrees': 180, 'p': 1},
        'Affine':         {'degrees': 0, 'translate': 0.1,
                           'scale': [0.7, 0.7], 'shear': 30, 'p': 1},
        'Combined': {
            'names': ['JpegTest', 'Crop', 'Cropout', 'Resize',
                      'GaussianBlur', 'Salt_Pepper', 'GaussianNoise',
                      'Dropout', 'Brightness', 'Contrast', 'Saturation', 'Hue'],
        },
        'Superposition': {
            'shuffle': True,
            'si_pool': ['Identity', 'Resize', 'GaussianBlur', 'Salt_Pepper',
                        'GaussianNoise', 'Cropout', 'Dropout', 'Saturation', 'Contrast'],
        },
    },

    'path': {
        'root':              os.getcwd(),
        'logs':              os.getcwd(),
        'logs_folder':       os.getcwd(),
        'folder_temp':       os.getcwd(),
        'train_folder':      os.getcwd(),
        'test_folder':       os.getcwd(),
        'resume_state_1pth': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "CIN", "cinNet&nsmNet.pth"),
    },

    'datasets': {
        'nDatasets': {'num': 10000, 'nTrain': 0.98, 'nval': 0.02},
        'test':      {'num': 1},
        'msg': {
            'mod_a': True,   # messages in [0, 1]
            'mod_b': False,  # messages in [-1, 1]
        },
        'H': 128,
        'W': 128,
    },

    'network': {
        'input': {
            'num_of_imgs': 1,
            'in_img_nc':   3,
        },
        'InvBlock': {
            'type':         'DBNet',
            'block_num':    16,
            'split1_img':   12,
            'split2_repeat':12,
            'downscaling': {
                'use_down':   True,
                'use_conv1x1':False,
                'in_nc':      3,
                'current_cn': 3,
                'down_num':   1,
                'scale':      0.5,
                'type':       'haar',
            },
            'IM': {'in_nc': 3, 'out_nc': 3, 'nf': 64, 'nb': 8},
        },
        'cs': {
            'in_nc':  3,
            'out_nc': 3,
        },
        'H':              128,
        'W':              128,
        'message_length': 30,
        'RGB2YUV':        False,
        'fusion': {
            'option':           True,
            'fusion_length':    256,
            'upconvT_channels': 1,
            'repeat_channel':   3,
            'blocks':           3,
        },
    },

    # Top-level shortcuts used by some modules
    'H':        128,
    'W':        128,
    'is_train': False,
    'subfolder': None,
}
