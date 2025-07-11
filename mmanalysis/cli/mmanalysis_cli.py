#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:52:31 2024

@author: roncofaber
"""

import argparse

import mmanalysis
import mmanalysis.main_analysis

#%%

def str2bool(value):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got "{value}".')

def main():
    parser = argparse.ArgumentParser(description="Run MMAnalysis with specified parameters.")
    parser.add_argument('-f', '--folder', type=str, default=None, help="Path to the folder to analyze.")

    parser.add_argument(
        "-g", "--giwaxs", type=str2bool, default=True,
        help="Whether or not giwaxs data will be provided (default: True)"
    )
    parser.add_argument(
        "-p", "--pl", type=str2bool, default=True,
        help="Whether or not photoluminescence data will be provided (default: True)"
    )
    parser.add_argument(
        "-l", "--logdata", type=str2bool, default=True,
        help="Whether or not logging data data will be provided (default: True)"
    )
    parser.add_argument(
        "-i", "--igor", type=str2bool, default=False,
        help="Boolean selection of either node-centered (True, suited for Igor Pro) or Pixel-centered (False, suited for Origin) 2D data maps (default: False)"
    )
    parser.add_argument(
        "-r", "--restart_file", type=str, default=None,
        help="Path to recover from a .pkl file (default: None)"
    )

    args = parser.parse_args()

    mmanalysis.main_analysis.main(
        name="MMA-Sample",
        restart_file=args.restart_file, 
        folder=args.folder,
        giwaxs=args.giwaxs,
        pl=args.pl,
        logdata=args.logdata,
        igor=args.igor
        )

if __name__ == "__main__":
    main()
