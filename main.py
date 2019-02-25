#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:10:05 2018

Main function that calls the classify_image class

This version slices the images then performs image recognition

@author: akhorshid
"""

import os
import pandas as pd
import time
from classify_image_opp import NodeLookup
import image_slicer
import shutil
#import matplotlib.pyplot as plt
#from PIL import Image


classifier = NodeLookup()

#os.chdir('/Users/akhorshid/Documents/Python/Image_Classification')     # sets the working directory
home = os.getcwd()     # Get the current working directory


### Enter file directory and name

data_dir = home + '/data/'

output_dir = home + '/output/'

try:
    os.makedirs(output_dir)
except FileExistsError:
    pass

###############################################################################
############################# Start Image Slicing #############################

files = os.listdir(data_dir) # os.listdir does not sort files
files = sorted(files) # Sorts files in ascending order (doesn't sort in place)

for i in range(1,len(files)):
    
    image_file = data_dir + str(files[i])
    tiles = image_slicer.slice(image_file, 4, save=False)
    
    imagepath = output_dir + str(files[i][0:-4])
    try:
        os.makedirs(imagepath)
    except FileExistsError:
        pass
    
    image_slicer.save_tiles(tiles, directory=imagepath , prefix=str(files[i][0:-4]), format='jpg')
    
    # Save whole image to the new folder:
    shutil.copy(image_file, imagepath)

print('\n')
print('Image slicing done!')
print('\n')

###############################################################################

description = []
score = []
main_filename = []
sliced_filename = []

files_folder = os.listdir(output_dir)
files_folder = sorted(files_folder)

for i in range(1,len(files_folder)):
    
        t1 = time.time()
        
        files_sliced = os.listdir(output_dir + str(files_folder[i]))
        files_sliced = sorted(files_sliced)
        
        for j in range(0,len(files_sliced)):
            
            # Define the image_file for the tf code:
            image_file = output_dir + str(files_folder[i]) + '/' + str(files_sliced[j])
            
            
            value = classifier.main(image_file) # Returns the predictions in a str array
            description.append(value[0])
            score.append(value[1])
            main_filename.append(files_folder[i])
            sliced_filename.append(files_sliced[j])
            
            df = pd.DataFrame({"File Name": main_filename, "Sliced File Name": sliced_filename, "Description": description, "Certainty": score})
            df.to_csv('Results2' + '.csv', index=False)
            
        t2 = time.time()
        
        print('\n')
        print('elapsed time = ' + str(t2-t1))
        print('\n')
        

###############################################################################
        


#fig = plt.figure(figsize = (8,8))
#
#for i in range(2,len(files)):
#    try:
#        tile = Image.open(data_dir + '/' + str(files[i]))
#        x = x+1
#        ax  = fig.add_subplot(4,4,x)
#        ax.set_title(str(x))               # Converts integer to string
#        plt.imshow(tile)
#    except OSError:
#        pass
#
#plt.show()
