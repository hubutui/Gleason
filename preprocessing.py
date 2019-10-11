#!/usr/bin/env python3
#
# This script use STAPLE to generate ground truth from 6 experts
# and convert the label to 0-5
# original label 0, 1, 3, 4, 5, 6 -> 0, 1, 3, 4, 5, 2
# so remember to convert it back after inference
#
import argparse
import os
import os.path as osp
import multiprocessing
from multiprocessing import Pool

import SimpleITK as sitk


def staple(item, inputdirs, outputdir, undecidedlabel):
    print("processing {}...".format(item))

    imgs = []
    for p in inputdirs:
        if osp.isfile(osp.join(p, item)):
            imgs.append(sitk.ReadImage(osp.join(p, item)))
    result = sitk.MultiLabelSTAPLE(imgs, 255)
    p1_data = sitk.GetArrayFromImage(imgs[0])
    result_data = sitk.GetArrayFromImage(result)
    if undecidedlabel:
        result_data[result_data == 255] = undecidedlabel
        result_data[result_data == 6] = 2
    else:
        result_data[result_data == 255] = p1_data[result_data == 255]
        result_data[result_data == 6] = 2
    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(imgs[0])
    sitk.WriteImage(result, osp.join(outputdir, item))


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdirs', type=str, nargs='+', default='/home/hubutui/Downloads/dataset/MICCAI2019/Gleason-2019',
                        help='input dirs of all masks')
    parser.add_argument('--outputdir', type=str, default='/home/hubutui/Downloads/dataset/MICCAI2019/Gleason-2019/preprocessed/finalmask-255',
                        help='output dir')
    parser.add_argument('--undecidedlabel', type=int, default=None,
                        help="label value for undecided pixels, we simply use the one expert's label value"
                             "in the order of arg inputdirs, "
                             "you could also use 0-6 if needed,"
                             "or just use 255, and ignore this label at training")
    parser.add_argument('--pool-size', type=int, default=None,
                        help='processes to run in parallel, default value is CPU count')

    return parser.parse_args()


if __name__ == '__main__':
    args = getargs()
    if args.undecidedlabel not in [None, 0, 1, 2, 3, 4, 5, 6, 255]:
        raise ValueError("unexpected label value for undecided pixels".format(args.undecidedlabel))
    os.makedirs(args.outputdir, exist_ok=True)
    maskfiles = []
    for i in args.inputdirs:
        maskfiles = maskfiles + os.listdir(i)
    maskfiles = set(maskfiles)

    if args.pool_size:
        processes = args.pool_size
    else:
        processes = multiprocessing.cpu_count()
    with Pool(processes=processes) as pool:
        results = [pool.apply_async(staple,
                                    args=(maskfile, args.inputdirs,
                                          args.outputdir, args.undecidedlabel))
                   for maskfile in maskfiles]
        _ = [_.get() for _ in results]
    print("Done")
