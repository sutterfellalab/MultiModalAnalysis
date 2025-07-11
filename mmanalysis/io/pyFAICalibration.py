# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:01:58 2025

@author: timko
"""

import numpy as np
from tkinter import filedialog
import subprocess
import fabio
import pyFAI
from pyFAI.calibrant import Calibrant
from pyFAI.geometry import Geometry
from pyFAI.goniometer import SingleGeometry
from pyFAI.gui import jupyter

# import matplotlib.pyplot as plt

def createGIWAXSMask(image, threshold = 1e9):
    
    mask = np.ones_like(image)  # Create a mask with the same shape as your data
    mask[image > threshold] = 0
    image = mask * image
    # plt.imshow(image)
    
    return image

def giwaxsCalibration(calibImage, calibData, defaultPONI):
    
    
    # Path to the default calibration file
    energy = '10' #At 12.3.2, we typically use 10 keV. Should make this an input at some point
    
    calibCommand = [
    'pyFAI-calib2',
    '--poni', defaultPONI,
    '--c', calibData,
    '--e', energy,
    calibImage
    ]
    
    # Launch the pyFAI calibration GUI using command line
    print("Launching pyFAI calibration GUI. This will take a few seconds...")
    subprocess.run(calibCommand, capture_output=True, text=True)

    # Prompt the user to select the calibration file
    calib_file_path = filedialog.askopenfilename(title="Select the calibration file", filetypes=[("Calibration files", "*.poni")])
    if not calib_file_path:
        print("No calibration file selected. Exiting.")
        return
    
    return calib_file_path
    
def refine_calibration(sampleName, image, initial_poni, calibrant_file, refined_poni):
    """
    Refine calibration using an initial .poni file and a custom calibrant.
    

    Parameters:
        image_path (str): Path to the calibration image.
        initial_poni (str): Path to the initial .poni file.
        calibrant_file (str): Path to the custom calibrant file.
        refined_poni (str): Path to save the refined .poni file.
    """
    
    # Load the custom calibrant
    calibrant = Calibrant(filename=calibrant_file)

    # Load the initial geometry
    initial = Geometry()
    initial.load(initial_poni)
    pilatus = pyFAI.detector_factory("Pilatus1M")
    maskedImage = createGIWAXSMask(image)
  
    # (Optional) Add custom refinement logic if necessary.
    sg = SingleGeometry("Recalibration of Sample " + sampleName, maskedImage, calibrant=calibrant, detector=pilatus, geometry=initial)
    sg.extract_cp(max_rings=5)
    sg.geometry_refinement.refine2(fix=["rot1", "rot2", "rot3", "wavelength"])
    sg.get_ai()
    
    # Verify refinement
    ax = jupyter.display(sg=sg)

    # Save refined .poni file
    sg.geometry_refinement.save(refined_poni)
    
    return 