import argparse
import sys
import pickle

import torch
import torchvision.transforms as standard_transforms
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2
from engine import *
from models.sasnet import SASNet
import os
import warnings
warnings.filterwarnings('ignore')

from models import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for SASNet evaluation', add_help=False)
    
    # parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    # parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_drop', default=300, type=int)
    parser.add_argument('--block_size', default=32, type=int)

    parser.add_argument('--data_root', default='./datas',
                        help='path where the dataset is')

    parser.add_argument('--checkpoints_dir', default='./weights',
                        help='path where to save checkpoints, empty for no saving')
    # parser.add_argument('--tensorboard_dir', default='./runs',
    #                     help='path where to save, empty for no saving')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--log_para', default=1000, type=int, help='scaling factor')
    
    parser.add_argument('--output_dir', default='/home/ubuntu/SASNet_FIBY/outputs_image',
                        help='path where to save')
    parser.add_argument('--weight_path', default='/home/ubuntu/SASNet_FIBY/ckpt/best_rmse_model-epoch=11-val_rmse=78.28.ckpt',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):
    print(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    
    model = build_model(args).cuda(args.gpu_id)
    
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    # del checkpoint['state_dict']['criterion.empty_weight']
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])

    if debug:
        sys.exit()

    model.eval()
    
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    # image_folder = "/home/ubuntu/P2PNet/DATA_ROOT/test/images"
    # image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(".jpg")]
    
    # predict_cnt = 0
    # total_dev = 0
    
    # for img_path in image_paths:
    img_path = '/home/ubuntu/SASNet_FIBY/datas/test/images/frame_001.jpg'
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    img_raw = img_raw.resize((128, 128), resample=Image.NEAREST)
    # pre-proccessing
    img = transform(img_raw).unsqueeze(0).cuda(args.gpu_id)

    pred_density_map = model(img)
    
    pred_density_map = torch.exp(pred_density_map / model.log_para)
    
    pred_count = torch.sum(pred_density_map).item()
    print('Predicted count:', pred_count)

    # draw the predictions
    # size = 3
    # img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    # for p in points:
    #     img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    #     # save the visualized image
    #     cv2.imwrite(os.path.join(args.output_dir, 'pred_lightning{}.jpg'.format(predict_cnt)), img_to_draw)
    # # print(total_dev/len(image_paths))
    # # print(predict_cnt)
    f, axarr = plt.subplots(1, 2)
    plt.rcParams['image.cmap'] = 'jet' # required for CUDA arrays
    img = img.detach().permute(0, 2, 3, 1).squeeze().cuda() 
    axarr[0].imshow(img.cpu().numpy())
    pred_density_map = pred_density_map.squeeze().cpu().detach().numpy()
    axarr[1].imshow(pred_density_map, cmap=plt.cm.jet)
    plt.savefig(os.path.join(args.output_dir, 'test_{}.jpg'.format(pred_count)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)