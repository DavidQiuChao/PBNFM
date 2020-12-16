#!/usr/bin/evn python
#coding=utf-8


import os
import tqdm
import numpy as np
import func as fc


def testByDir(srcDir,outDir):
    noiseModel = np.load('jointDistribution_T1pro.npy',allow_pickle=True).item()
    names = os.listdir(srcDir)
    for name in tqdm.tqdm(names):
        path = os.path.join(srcDir,name)
        im,wb = fc.loadRawDng(path)
        nsim,fac = fc.getNoisePair(noiseModel,im,'T1pro')
        fc.showSample(nsim,im,wb,outDir)
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--srcDir',\
            help='source image dir')
    parser.add_argument('-o','--outDir',\
            help='output image dir')
    args = parser.parse_args()
    srcDir = args.srcDir
    outDir = args.outDir
    testByDir(srcDir,outDir)
