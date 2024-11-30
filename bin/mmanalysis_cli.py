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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run MMAnalysis with specified parameters.")
    parser.add_argument('-f', '--folder', type=str, default=None, help="Path to the folder to analyze.")
    
    args = parser.parse_args()

    mmanalysis.main_analysis.main(folder=args.folder)