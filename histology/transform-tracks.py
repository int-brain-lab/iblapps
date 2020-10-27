"""Re-Scale and Transform Traced Electrode Tracks

This script will convert traced tracks from full resolution stacks, into
the downsampled stack and finally transform them into Standard (ARA)
space.


DEPENDENCIES

For this script to run, elastix must be installed.  See instruction at
elastix.isi.uu.nl/download.php.

Check elastix is installed by running at command line:

$ elastix --version
elastix version: 4.800


GENERATING TRACED TRACKS

The tracks must be generated using Lasagna and the add_line_plugin.


The resulting .csv files must be saved into a directory titled
sparsedata/ inside the Sample ROOT Directory:

  - [KS006]:
    - sparsedata:
      - track01.csv
      - track02.csv

NB: [KS006] can be any sample ID.

The Sample ROOT Directory must also have the following structure and
files:

  - [KS006]:
    - downsampledStacks_25:
      - dsKS006_190508_112940_25_25_GR.txt
      - ARA2sample:
        - TransformParameters.0.txt
        - TransformParameters.1.txt

The script uses information in dsKS006_190508_112940_25_25_GR.txt, 
TransformParameters.0.txt and TransformParameters.1.txt to perform the
transformation.

The output of the script is a series of new csv files, with the track
coordinates now in the registered ARA space.

- [KS006]:
    - downsampledStacks_25:
      - rescaled-transformed-sparsedata:
        - track01.csv
        - track02.csv


"""

import sys
import os
import re
import glob
import csv
import subprocess
from os import listdir
from os.path import isfile, join
from pathlib import Path
import platform

#print("Python version")
#print (sys.version)
#print("Version info.")
#print (sys.version_info)

#print( os.getcwd() )

xy = 0.0 # num to hold x/y resolution
z = 0.0 # num to hold z resolution


rescaleFile = glob.glob("downsampledStacks_25" + os.sep + "ds*GR.txt")[0]

print(rescaleFile)


with open (rescaleFile, 'rt') as myfile:

    for line in myfile:

        #print(line)

        if line.lower().find("x/y:") != -1:
            for t in line.split():
                try:
                    xy = float(t)
                except ValueError:
                    pass

        if line.lower().find("z:") != -1:
            for t in line.split():
                try:
                    z = float(t)
                except ValueError:
                    pass

#print(xy)

#print(z)


# NEXT - change the path in TransformParameters.1.txt file to point to correct ABSOLUTE path to TransformParameters.0.txt file:

tp = os.getcwd() + os.sep + "downsampledStacks_25" +  os.sep + "ARA2sample" +  os.sep + "TransformParameters.0.txt"

#print(tp)


#tp1 = "downsampledStacks_25" +  os.sep + "ARA2sample" +  os.sep + "TransformParameters.1.txt"

#for line in fileinput.input(tp1, inplace = 1): 
#      print line.replace("foo", "bar"),

fp = open("downsampledStacks_25" +  os.sep + "ARA2sample" +  os.sep + "TransformParameters.1.txt","r+")

fg = open("downsampledStacks_25" +  os.sep + "ARA2sample" +  os.sep + "new_TransformParameters.1.txt","w")

for line in fp:

    if line.find("(InitialTransformParametersFileName") != -1:
        new_line = line.replace(line,'(InitialTransformParametersFileName '+tp+')\n')
        fg.write(new_line)

    else:
        fg.write(line)

fg.close()

fp.close()

os.remove("downsampledStacks_25" +  os.sep + "ARA2sample" +  os.sep + "TransformParameters.1.txt")
os.rename("downsampledStacks_25" +  os.sep + "ARA2sample" +  os.sep + "new_TransformParameters.1.txt","downsampledStacks_25" +  os.sep + "ARA2sample" +  os.sep + "TransformParameters.1.txt")


# have extracted the resolution change - now open each CSV file of track coords, and transform them:

# Read the CSV files:

# get list of files:
csvFiles = [f for f in listdir("sparsedata" + os.sep) if isfile(join("sparsedata" + os.sep, f))]

for csvFilePath in csvFiles:
    
    print(csvFilePath)
    # get the total number of lines in the file first:
    csvfile = open("sparsedata" + os.sep + csvFilePath)
    linenum = len(csvfile.readlines())
    csvfile.close()

    csvFilePathNoExt = csvFilePath[:-4]


    with open( ("sparsedata" + os.sep + csvFilePath), newline='') as csvfile:
        # write to rescaled-file:
        csvOut = open( ('downsampledStacks_25' +  os.sep + 'rescaled-' + csvFilePath), 'w')
        pts = open('downsampledStacks_25' +  os.sep + 'rescaled-'+csvFilePathNoExt+'.txt', 'w')
        pts.write("point\n")
        pts.write("%s\n" % linenum )
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            # NB: CSV file is laid out ZXY!
            row[0] = round( (float(row[0])/ z), 6)
        
            row[1] = round( (float(row[1])/ xy), 6)
        
            row[2] = round( (float(row[2])/ xy), 6)
        
            # re-write as PTS:  XYZ!
            pts.write("%s " % row[1])
            pts.write("%s " % row[2])
            pts.write("%s\n" % row[0])

            # also re-write as csv:  ZXY!
            csvOut.write("%s," % row[0])
            csvOut.write("%s," % row[1])
            csvOut.write("%s\n" % row[2])

        pts.close()
        csvOut.close()


    # next, use system call to convert the points into ARA space:
    # Use the INVERSE of sample2ARA -> ARA2sample!

    pts = "downsampledStacks_25" +  os.sep + "rescaled-"+csvFilePathNoExt+'.txt'
    out = "downsampledStacks_25" +  os.sep + "rescaled-transformed-sparsedata" +  os.sep
    # NOTE - need to use TransformParameters.1.txt - which points to TP0!
    tp = "downsampledStacks_25" +  os.sep + "ARA2sample" +  os.sep + "TransformParameters.1.txt"
    
    # make output DIR:
    if os.path.isdir(out)==False:
      os.mkdir(out)

    cmd = "transformix -def " + pts + " -out " + out + " -tp " + tp

    print(cmd)

    returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
    print('returned value:', returned_value)

    # finally, want to convert outputpoints.txt into the ORIGINAL CSV format:
    outputPts = open(out+"outputpoints.txt", "r")

    outPtsCsv = open( (r"" + out + csvFilePathNoExt + "_transformed.csv"), 'w')

    for aline in outputPts:
        values = aline.split()
        # XYZ:
        # print("XYZ: ", values[22], values[23], values[24] )
        # NB: CSV file is laid out ZXY!
        outPtsCsv.write("%s," % values[24])
        outPtsCsv.write("%s," % values[22])
        outPtsCsv.write("%s\n" % values[23])

    outputPts.close()

    # remove the temp files:
    os.remove('downsampledStacks_25' +  os.sep + 'rescaled-' + csvFilePath)
    os.remove('downsampledStacks_25' +  os.sep + 'rescaled-'+csvFilePathNoExt+'.txt')
    os.remove(out+"outputpoints.txt")
    os.remove(out+"transformix.log")


