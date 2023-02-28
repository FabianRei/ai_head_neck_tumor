import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_MAX_THREADS"] = "1"

import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import pickle
import argparse
import socket
import torch


print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

from data_loading.data_set import EcatDFDataset, get_default_tio_transform
from torch.utils.data import DataLoader
from deep_learning.Lit3dResnet import Lit3dResnet

# Get argpars args
parser = argparse.ArgumentParser()
parser.add_argument('-col_id', '--col_id', help='Column index to excecute', default=0)
parser.add_argument('-cv_split', '--cv_split', help='Cross validation split', default=0)
parser.add_argument('-run_id', '--run_id', help='Run id', default=1)
args = vars(parser.parse_args())

col_id = int(args['col_id'])
cv_split = int(args['cv_split'])
run_id = int(args['run_id'])

columns = ['CT_img_path', 'CT_roi_path', 'PT_img_path', 'PT_roi_path']
indices = [[0], [2], [0, 1], [2, 3], [0, 1, 2, 3], [0, 2], [0, 2, 3]]
img_columns = [columns[i] for i in indices[col_id]]
info_cols = ''.join(img_columns)

###############################################
# Run 1
bs = 8
lr = 0.001
adam_regularization = 0.01
lr_decay = 0.95
resnet_model_size = 50
model_chan_in = len(img_columns)
n_classes = 2
epochs = 1
###############################################
# run 2
if run_id == 2:
    lr = 0.01

info = f'Lit3dResnet_50_1chan_28feb23_{info_cols}_cv_{cv_split}_{epochs}_epochs_run_{run_id}TEST'




if socket.gethostname() == 'blue':
    path_df = Path('/home/fabian/projects/phd/ai_radiation_therapy/ai_head_neck_tumor/data/clinical_data_has_CT_PT_21feb23.feather')
    df = pd.read_feather(path_df)
    data_path = Path(df.iloc[0].blue_data_path)
    data_path_col = 'blue_data_path'
else:
    path_df = Path("/home/freith/projects/ai_head_neck_tumor/data/clinical_data_has_CT_PT_21feb23.feather")
    df = pd.read_feather(path_df)
    data_path = Path(df.iloc[0].mdc_data_path)
    data_path_col = 'mdc_data_path'
path_checkpoints = path_df.parent / 'checkpoints'



# Data stuff
ds_train = EcatDFDataset(df, mode='train', cv_split=cv_split,  img_columns=img_columns, transform=get_default_tio_transform('train'),
                         data_path_col=data_path_col)
ds_valid = EcatDFDataset(df, mode='valid', cv_split=cv_split,  img_columns=img_columns, transform=get_default_tio_transform('valid'),
                         data_path_col=data_path_col)
dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=4)
dl_valid = DataLoader(ds_valid, batch_size=bs, shuffle=False, num_workers=4)

# Model & training stuff
params = {'lr': lr, 'adam_regularization': adam_regularization, 'lr_decay': lr_decay, 'resnet_model_size': resnet_model_size, 
          'model_chan_in': 1, 'n_classes': 2, 'bs': bs, 'img_column': img_columns, 'data_path': data_path, 'path_df': path_df}
model = Lit3dResnet(params)
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epochs, enable_checkpointing=False, default_root_dir=path_checkpoints)
trainer.fit(model, dl_train, dl_valid)

# Saving stuff
params['loss_acc_dict'] = model.loss_acc_dict
trainer.save_checkpoint(path_checkpoints / f"{info}.ckpt")
pickle.dump(params, open(path_checkpoints / f"{info}_params.pkl", "wb"))