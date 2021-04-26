import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", type=str)

args = parser.parse_args()

print(args.m)
