import argparse
import sys
'''
Parse common arguments to many of these files.  
Each module has access to the command line arguments, so they will all see the same thing without
explicitly passing stuff in
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', type=str, required=True, help='experiment label')
    parser.add_argument('-p', '--pixels', type=int, default=64, help='pixels')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    args.label += '_%d'%args.pixels

    return args
