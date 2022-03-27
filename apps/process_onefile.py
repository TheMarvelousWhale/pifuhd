# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
from lib.data import  EvalDataset
from lib.model import HGPIFuNetwNML, HGPIFuMRNet
import yaml
from apps.recon import gen_mesh
import argparse

def init_config():
    with open("../config.yaml", "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return data

parser = argparse.ArgumentParser()
parser.add_argument("image_path")

CONFIG = init_config()
state_dict_path = './checkpoints/pifuhd.pt'
cuda = 'cpu' #torch.device('cuda:%d' % opt.gpu_id if torch.cuda.is_available() else 'cpu')
state_dict = None

if state_dict_path is not None and os.path.exists(state_dict_path):
    print('Resuming from ', state_dict_path)
    state_dict = torch.load(state_dict_path, map_location=cuda)    
    print('Warning: opt is overwritten.')
    dataroot = './sample_images/'
    resolution = CONFIG['resolution']
    results_path = './results/'
    loadSize = 1024
    
    opt = state_dict['opt']
    opt.dataroot = dataroot
    opt.resolution = resolution
    opt.results_path = results_path
    opt.loadSize = loadSize
else:
    raise Exception('failed loading state dict!', state_dict_path)

projection_mode = 'orthogonal'

opt_netG = state_dict['opt_netG']
netG = HGPIFuNetwNML(opt_netG, projection_mode).to(device=cuda)
netMR = HGPIFuMRNet(opt, netG, projection_mode).to(device=cuda)

def set_eval():
    netG.eval()

# load checkpoints
netMR.load_state_dict(state_dict['model_state_dict'])

os.makedirs(opt.checkpoints_path, exist_ok=True)
os.makedirs(opt.results_path, exist_ok=True)
os.makedirs('%s/%s/recon' % (opt.results_path, opt.name), exist_ok=True)


def do(filepath,opt,netMR,cuda,):
    test_dataset = EvalDataset(opt)
    with torch.no_grad():
        set_eval()

        index = 0
        for j,item in enumerate(test_dataset.img_files):
            if filepath in item:
                index = j

        test_data = test_dataset[index]

        save_path = '%s/%s/recon/result_%s_%d.obj' % (opt.results_path, opt.name, test_data['name'], opt.resolution)

        gen_mesh(opt.resolution, netMR, cuda, test_data, save_path, components=opt.use_compose)

  

  
if __name__ == "__main__":
    args = parser.parse_args()
    do(args.image_path,opt,netMR,cuda)
