#!/usr/bin/python3

import os
def get_sorted_DMT_files():

   filePaths = []
   for root, subdirs, files in os.walk(incoming_dir, topdown=True):
      filePaths.extend(files)#sort()

   filePaths.sort(key = lambda path: os.path.split(path)[1])
   return filePaths
   #print(filePaths)
#fileNames = [os.path.split(path)[1] for path in filePaths]

