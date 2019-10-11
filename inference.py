#!/usr/bin/env python3
#
import argparse
import csv
import os
import os.path as osp

import gluoncvth as gcv
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, required=True,
                        help="input data dir")
    parser.add_argument('--model', type=str, required=True,
                        help="model file")
    parser.add_argument('--outputdir', type=str, default=None,
                        help='output dir')
    parser.add_argument('--aux', action='store_true',
                        help='use aux layer')

    return parser.parse_args()


def get_gleason_grade(segmentation):
    segmentation = segmentation.flatten()
    u, count = np.unique(segmentation, return_counts=True)

    ind = np.argsort(count)
    if u.size == 1:
        primary = u[ind][-1]
        result = primary*2
    elif u.size == 2:
        primary = u[ind][-1]
        secondary = u[ind][-2]
        result = primary + secondary
    else:
        primary = u[ind][-1]
        result = primary + u.max()

    return result


if __name__ == '__main__':
    args = getargs()
    os.makedirs(osp.join(args.outputdir, 'task1'), exist_ok=True)
    os.makedirs(osp.join(args.outputdir, 'task2'), exist_ok=True)
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 6
    model = gcv.models.get_psp_resnet101_ade(pretrained=True)
    model.auxlayer.conv5[-1] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
    model.head.conv5[-1] = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)

    model_data = torch.load(args.model, map_location='cpu')
    model.load_state_dict(model_data['model'])
    model = model.to(device)
    tf = transforms.Compose([
        transforms.Resize(800),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    with torch.no_grad():
        model.eval()
        grade = {}
        for imgfile in os.listdir(args.inputdir):
            data = Image.open(osp.join(args.inputdir, imgfile))
            w = data.width
            h = data.height
            data = tf(data)
            data = data.to(device).unsqueeze(0)
            y, y_aux = model(data)
            # y_aux takes lower weight, or even 0 if you like
            y = y + 0.5 * y_aux
            y = y.argmax(dim=1).cpu().squeeze().numpy().astype(np.uint8)

            y[y == 2] = 6
            result = Image.fromarray(y)
            result = transforms.Resize((h, w))(result)
            result.save(osp.join(args.outputdir, 'task1', imgfile))
            grade[imgfile[:-4]] = get_gleason_grade(y)

    with open(osp.join(args.outputdir, 'task2', 'task2.csv'), 'w') as f:
        writer = csv.writer(f)
        for key in grade.keys():
            writer.writerow([key, grade[key]])
    print('Done')
