#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 18:07:43 2018

@author: akhorshid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from classify_image_opp import NodeLookup
from PIL import Image
import image_slicer
import shutil

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)

class main():
    
    def __init__(self):
        home = os.getcwd()
        self.file = home + '/Results.csv'
        self.df = pd.read_csv(self.file)
        
    def analyze_files(self):
        classifier = NodeLookup()
        
        home = os.getcwd()     # Get the current working directory
           
        ### Enter files directory and name:
        
        data_dir = home + '/data/'
        
        output_dir = home + '/output/'
        
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass
        
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
                    df.to_csv('test' + '.csv', index=False)
                    
                t2 = time.time()
                
                print('\n')
                print('elapsed time = ' + str(t2-t1))
                print('\n')
        
        
    def organize_csv_file(self):    
        # function that finds the first comma, then removes the succeeding text:
        truncated_description = []
        
        for i in range(0, len(self.df['Description'])):
            
            for j in range(0,len(self.df['Description'][i])):
                
                if (self.df['Description'][i][j] == ','):
                    comma_position = j
                    break
                
                else:
                    comma_position = len(self.df['Description'][i])
                               
            truncated_description.append(self.df['Description'][i][0:comma_position])
            
        
        df_truncated_description = pd.DataFrame({'Truncated Description': truncated_description})
        self.df = self.df.join(df_truncated_description)
        self.df = self.df[['File Name', 'Sliced File Name', 'Truncated Description', 'Description', 'Certainty']]         
        self.df.to_csv('organized_results_noduplicates' + '.csv', index=False)
        
        return self.df
    
    def obtain_score(self):
        file_name = 'image362' # Name of the file we are comparing to
#        file_name = 'image2088'
        
        score = []
        self.df = pd.read_csv('/Users/akhorshid/Documents/Python/Image_Classification_V1/Organized Results.csv')
        self.df = self.df[['File Name', 'Truncated Description', 'Certainty']]
        
        # Return unique file name with a list of descriptions:
        self.df1 = self.df.groupby(['File Name'])['Truncated Description'].apply(list).reset_index() # reset_index() restores dataframe to conventional structure
        df_certainty = self.df.groupby(['File Name'])['Certainty'].apply(list).reset_index()
        self.df1 = self.df1.join(df_certainty['Certainty'])   # adding the list of certainties
        # Dropping duplicates from the list of descriptions of each image:
        self.df1['Truncated Description'] = list(map(set,self.df1['Truncated Description']))
        # Converting set (description) into list:
        self.df1['Truncated Description'] = self.df1['Truncated Description'].apply(list)
        
        
        file_index = self.df1['File Name'][self.df1['File Name'] == file_name].index[0] # index value of the file
        file_values = self.df1['Truncated Description'][file_index]
        
        
        
        
        for i in range(0, len(self.df1['File Name'])):
            score.append(0) # initial score = 0 (i.e. no matches)
            
            if (self.df1['File Name'][i] == file_name):
                pass 
            else:
                
                for j in range(0, len(self.df1['Truncated Description'][i])):

                    for k in range(0, len(file_values)):
                        
                        if (file_values[k] == self.df1['Truncated Description'][i][j]):
#                            score[i] = score[i] + 1
                            score[i] = score[i] + self.df1['Certainty'][i][j] + 1
                        else:
                            pass
        
        header_score ='Score (' + str(file_name)+ ')'
        df_score = pd.DataFrame({header_score: score})
        self.df1 = self.df1.join(df_score)
        self.df1 = self.df1.sort_values(by=[header_score], ascending = False).reset_index()
        
        # Reordering columns (using double [[]]):
        self.df1 = self.df1[['index', 'File Name', 'Truncated Description', header_score, 'Certainty']]
        
        # Plotting figure:
        fig = plt.figure(figsize = (12,8))
        fig.subplots_adjust(wspace=0.5)
        
        tile = Image.open('/Users/akhorshid/Documents/Python/Image_Classification_V1/data/' + file_name + '.jpg')
        ax  = fig.add_subplot(2,3,2)
        ax.set_title('Sample image: ' + str(file_name), fontsize = 12)
        plt.imshow(tile)
        plt.xlabel(str(file_values), fontsize = 10)
        x = 3
        
        for i in range(0,3):
            tile = Image.open('/Users/akhorshid/Documents/Python/Image_Classification_V1/data/' + self.df1['File Name'][i] + '.jpg')
            x = x+1
            ax  = fig.add_subplot(2,3,x)
            ax.set_title('Suggested image: ' + str(self.df1['File Name'][i]), fontsize = 12)
            plt.imshow(tile)
            plt.xlabel(str(self.df1['Truncated Description'][i]), fontsize = 9)
                
        plt.show()
        
        return self.df, self.df1
                

    
    def plot_histogram(self):
        
        self.df = pd.read_csv('/Users/akhorshid/Documents/Python/Image_Classification_V1/Organized Results.csv')
        # Dropping all rows that contain an underscore within the str:
#        self.df1 = self.df[~self.df['Sliced File Name'].str.contains('_')]
        
        self.df1 = self.df.groupby(['Truncated Description']).size().reset_index()
        self.df1 = self.df1.sort_values([0], ascending = False).reset_index()
        
        self.df1.rename(columns={0:'Count'}, inplace=True)   # Rename count header
        
        description_array = np.asarray(self.df1['Truncated Description'][0:50])
        count_array = np.asarray(self.df1['Count'][0:50]/len(self.df))
        
        plt.bar(description_array, count_array, label= '24970 images')
        plt.xticks(rotation=90, fontsize = 7)
        plt.ylabel('Count')
        plt.legend(loc='upper right', fontsize = 10)
        plt.show()
        plt.savefig('Figure_9.pdf', bbox_inches = "tight")
        
        
        return self.df1
    
    
    def compare_histogram(self):
        
        self.df = pd.read_csv('/Users/akhorshid/Documents/Python/Image_Classification_V1/Organized Results.csv')
        # Dropping all rows that contain an underscore within the str:
        self.df1 = self.df[~self.df['Sliced File Name'].str.contains('_')]   # Orginal images
        self.df2 = self.df[self.df['Sliced File Name'].str.contains('_')]    # Sliced images
        
        length1 = len(self.df1)
        length2 = len(self.df2)
        
        self.df1 = self.df1.groupby(['Truncated Description']).size().reset_index()
        self.df1 = self.df1.sort_values([0], ascending = False).reset_index()
        self.df1.rename(columns={0:'Count'}, inplace=True)   # Rename count header
        
        description1 = np.asarray(self.df1['Truncated Description'])
    
        
        self.df2 = self.df2.groupby(['Truncated Description']).size().reset_index()
        self.df2 = self.df2.sort_values([0], ascending = False).reset_index()
        self.df2.rename(columns={0:'Count'}, inplace=True)   # Rename count header
        
        self.df2 = self.df2[self.df2['Truncated Description'].isin(description1)].reset_index()
        
        description2 = np.asarray(self.df2['Truncated Description'])
        
        self.df1 = self.df1[self.df1['Truncated Description'].isin(description2)].reset_index()
        
        # sorting both dataframes to the same order:
        sorter = np.asarray(self.df1['Truncated Description'])
        self.df2['Truncated Description'] = self.df2['Truncated Description'].astype("category")
        self.df2['Truncated Description'].cat.set_categories(sorter, inplace=True)
        self.df2 = self.df2.sort_values(['Truncated Description'])
        self.df2.drop('level_0', axis=1, inplace=True)
        
        self.df2 = self.df2.reset_index()
        
        self.df1['Count'] = self.df1['Count']/length1
        self.df2['Count'] = self.df2['Count']/length2

        
        ratio = self.df2['Count'][0:50]/self.df1['Count'][0:50]
        
        
        
        description_array1 = np.asarray(self.df1['Truncated Description'][0:50])
        count_array1 = np.asarray(self.df1['Count'][0:50]*100)
        
        description_array2 = np.asarray(self.df2['Truncated Description'][0:50])
        count_array2 = np.asarray(self.df2['Count'][0:50]*100)
        
        plt.bar(description_array1, count_array1, label= str(length1) + ' images (Original)')
        
        plt.bar(description_array2, count_array2, label= str(length2) + ' images (Sliced)', color = 'red')
        plt.xticks(rotation=90, fontsize = 7)
        plt.ylabel('Count (%)')
        plt.legend(loc='upper right', fontsize = 10)
        
        
#        plt.show()
        plt.savefig('Frequency.pdf', bbox_inches = "tight")
        
        
        return self.df1, self.df2, ratio
    
    def plot_single_figure(self):
    
        file_name = 'image10'
        
        self.df = pd.read_csv('/Users/akhorshid/Documents/Python/Image_Classification_V1/Organized Results.csv')
        self.df = self.df[['File Name', 'Truncated Description', 'Certainty']]
        
        # Return unique file name with a list of descriptions:
        self.df1 = self.df.groupby(['File Name'])['Truncated Description'].apply(list).reset_index() # reset_index() restores dataframe to conventional structure
        df_certainty = self.df.groupby(['File Name'])['Certainty'].apply(list).reset_index()
        self.df1 = self.df1.join(df_certainty['Certainty'])   # adding the list of certainties
        
        file_index = self.df1['File Name'][self.df1['File Name'] == file_name].index[0] # index value of the file
        
        file_values = self.df1['Truncated Description'][file_index]
        file_score = self.df1['Certainty'][file_index]
#        fig = plt.figure(figsize = (12,8))

        
        tile = Image.open('/Users/akhorshid/Documents/Python/Image_Classification_V1/data/' + file_name + '.jpg')
        plt.imshow(tile)
        plt.title(str(file_values[0]) + ' (score = ' + str(file_score[0]) + ')', fontsize = 10)
        
        plt.show()
        
    
            
main_class = main()
file = main_class.obtain_score()

