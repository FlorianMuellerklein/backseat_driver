import argparse
import sys
'''
Parse common arguments to many of these files.
Each module has access to the command line arguments, so they will all see the same thing without
explicitly passing stuff in
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', type=str, default='', help='experiment label')
    parser.add_argument('-p', '--pixels', type=int, default=128, help='pixels')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--fold', type=int, default=0)

    args, unknown_args = parser.parse_known_args()
    args.label += '_%d_fold%02d'%(args.pixels, args.fold)
    #args.label += '_%d'%args.pixels

    return args, unknown_args
