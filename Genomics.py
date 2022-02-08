# Author: Conner Warnock
# Huge project. This program looks at genomic data ( ~400x90000 matrix, 73 MB).
# Feature selection, descriptive statistics, data cleaning, data exploration, data visualization,
# k-medoids and k-means clustering (from scratch), heatmaps, network + community discovery, degree centrality,
# intersections of sets, and a lot more. My longest project.
# Date: April 24, 2020
# Open file

import math
import copy
from collections import Counter
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Read data file
file_name = "GSE64881_segmentation_at_30000bp.passqc.multibam.txt"
file = open(file_name)

# Variable list/declarations
data = []
presentWindows = 0
presentWindowsList = []
# numOfNPs = 0
# numOfGWs = 0

# Read file into 2D list

for line in file:
    number_strings = line.split()
    data.append(number_strings)

# Read feature table (csv)
featureTable = []
with open("gam_feature_community(1).csv") as featureTableFile:
    csvReader = csv.reader(featureTableFile)
    for row in csvReader:
        featureTable.append(row[:])

# Gets numOfGWs
def get_print_numofgws():
    numOfGWs = len(data) - 1  # Genomic Windows (rows minus 1)
    print("Genomic Windows: ", numOfGWs)
    return numOfGWs


# Gets numOfNPs
def get_print_numofnps():
    numOfNPs = len(data[0]) - 3  # Nuclear Profiles (columns minus 3)
    print("Nuclear Profiles: ", numOfNPs)
    return numOfNPs


# Acquires a list of: count of present windows in NPs.
# Note: Data not cleaned, largest outlier is 4.5 stdevs out.
def get_list_present_windows(presentWindows, presentWindowsList):
    for i in range(3, len(data[0])):
        for j in range(1, len(data)):

            if data[j][i] == "1":
                presentWindows += 1

        presentWindowsList.append(presentWindows)
        presentWindows = 0

    return presentWindowsList


# Average, smallest, and largest number of present genomic windows per NP
def avg_large_small_num_present_windows(numOfGWs, presentWindowsList):
    sumOfPresentWindows = 0
    smallestNumOfPresentWindows = numOfGWs  # Set to max
    largestNumOfPresentWindows = 0  # Set to min

    for i in range(numOfNPs):
        sumOfPresentWindows += presentWindowsList[i]
        if presentWindowsList[i] < smallestNumOfPresentWindows:
            smallestNumOfPresentWindows = presentWindowsList[i]
        if presentWindowsList[i] > largestNumOfPresentWindows:
            largestNumOfPresentWindows = presentWindowsList[i]

    averageNumOfGWs = sumOfPresentWindows / len(presentWindowsList)
    print("Average Number of Present GWs per NP: ", averageNumOfGWs)
    print("Smallest Number of Present GWs per NP: ", smallestNumOfPresentWindows)
    print("Largest Number of Present GWs per NP: ", largestNumOfPresentWindows)

    return averageNumOfGWs, smallestNumOfPresentWindows, largestNumOfPresentWindows


# Calculate standard deviation of NPs per GW
def st_dev_present_gws(presentWindowsList, averageNumOfGWs):
    stDevPlaceholder = 0
    for i in range(len(presentWindowsList)):
        stDevPlaceholder += (presentWindowsList[i] - averageNumOfGWs) ** 2

    stDevPlaceholder = stDevPlaceholder / (len(presentWindowsList) - 1)
    stDevPresentGWs = math.sqrt(stDevPlaceholder)

    print("Standard Deviation of Present GWs per NP: ", stDevPresentGWs)

    return stDevPresentGWs


# Acquires a list of: count of NPs in a given genomic window
def get_present_nps_list(data, numOfGWs, numOfNPs):
    presentNPs = 0
    presentNPsList = []

    for j in range(1, numOfGWs + 1):
        for i in range(3, numOfNPs + 3):

            if data[j][i] == "1":
                presentNPs += 1
        # Cleans data by not including any GWs with over 100 NPs present
        if presentNPs > 100:
            presentNPs = 0
            continue
        presentNPsList.append(presentNPs)
        presentNPs = 0

    return presentNPsList


# Average, smallest, and largest number of NPs per genomic window
def avg_large_small_num_present_nps(numOfNPs, presentNPsList):
    sumOfPresentNPs = 0
    smallestNumOfPresentNPs = numOfNPs  # Set to max
    largestNumOfPresentNPs = 0  # Set to min

    for i in range(len(presentNPsList)):
        sumOfPresentNPs += presentNPsList[i]
        if presentNPsList[i] < smallestNumOfPresentNPs:
            smallestNumOfPresentNPs = presentNPsList[i]
        if presentNPsList[i] > largestNumOfPresentNPs:
            largestNumOfPresentNPs = presentNPsList[i]

    averageNumOfNPs = sumOfPresentNPs / len(presentNPsList)
    print("Average Number of Present NPs per GW: ", averageNumOfNPs)
    print("Smallest Number of Present NPs per GW: ", smallestNumOfPresentNPs)
    print("Largest Number of Present NPs per GW: ", largestNumOfPresentNPs)

    return averageNumOfNPs, smallestNumOfPresentNPs, largestNumOfPresentNPs


# Calculate standard deviation of NPs per GW
def st_dev_present_gws(presentNPsList, averageNumOfNPs):
    stDevPlaceholder = 0
    for i in range(len(presentNPsList)):
        stDevPlaceholder += (presentNPsList[i] - averageNumOfNPs) ** 2

    stDevPlaceholder = stDevPlaceholder / (len(presentNPsList) - 1)
    stDevPresentNPs = math.sqrt(stDevPlaceholder)

    print("Standard Deviation of Present NPs per GW: ", stDevPresentNPs)

    return stDevPresentNPs


# Create a scatter plot of NPs per GW
def total_windows_plot(averageNumOfNPs, presentNPsList, stDevPresentNPs):
    averageNumOfNPsList = [averageNumOfNPs] * len(presentNPsList)
    stDevPresentNPsHigh = averageNumOfNPs + stDevPresentNPs
    stDevPresentNPsListHigh = [stDevPresentNPsHigh] * len(presentNPsList)
    stDevPresentNPsLow = averageNumOfNPs - stDevPresentNPs
    stDevPresentNPsListLow = [stDevPresentNPsLow] * len(presentNPsList)

    area = 2
    fig, ax = plt.subplots()
    ax.scatter(range(0, len(presentNPsList)), presentNPsList, label="Data", s=area, color="darkslategray")
    ax.scatter(range(0, len(presentNPsList)), averageNumOfNPsList, label="Mean", s=area, color="firebrick")
    ax.scatter(range(0, len(presentNPsList)), stDevPresentNPsListHigh, label="Standard Deviation", s=area,
               color="darkorange")
    ax.scatter(range(0, len(presentNPsList)), stDevPresentNPsListLow, s=area, color="darkorange")
    plt.title("Present NPs per Genomic Window")
    plt.xlabel("Window Index")
    plt.ylabel("Window Detection Frequency")
    plt.legend(loc="upper right")
    plt.show()


# Find HIST1 region and extract (Chr13, 21.7 Mb Start - 24.1 Mb Stop)
def extract_hist1_region(numOfGWs, data):
    hist1 = []
    hist1StartIndex = 0
    hist1StopIndex = 0

    for i in range(0, numOfGWs + 1):
        if data[i][0] == "chr13":
            if int(data[i][1]) >= 21690000:
                if int(data[i][2]) <= 21720000:
                    hist1StartIndex = i
                if int(data[i][2]) <= 24120000:
                    hist1StopIndex = i

    print(hist1StartIndex)  # 69715 w/ cleaning
    print(hist1StopIndex)  # 69795 w/ cleaning

    hist1 = data[hist1StartIndex:hist1StopIndex+1]
    print("Number of Genomic Windows in HIST1 region: ", len(hist1))

    return hist1, hist1StartIndex, hist1StopIndex


# 81 windows in hist1 region, 163 NPs
# Extract relevant NPs in HIST1
def extract_relevant_hist1_nps(hist1):
    hist1NPs = []

    for i in range(0, len(hist1)):
        for j in range(3, len(data[0])):
            if hist1[i][j] == "1":
                hist1NPs.append(j)

    hist1NPs2 = []

    for i in hist1NPs:
        if i not in hist1NPs2:
            hist1NPs2.append(i)

    hist1NPs = hist1NPs2
    print("Number of Relevant NPs in HIST1 region: ", len(hist1NPs))
    hist1NPs.sort()
    print("Relevant HIST1 NPs: ", hist1NPs)

    return hist1NPs


# Construct new hist1 with only relevant NPs (extracted relevant NPs)
def construct_relevant_hist1_nps(hist1):
    newHist1 = []
    newHist1Row = []

    for i in range(0, len(hist1)):
        newHist1Row.clear()
        for j in hist1NPs:
            newHist1Row.append(hist1[i][j])
        newHist1.append(newHist1Row[:])

    hist1 = newHist1

    return hist1


# Average, smallest, largest number of windows present in NPs in hist1 region
def avg_large_small_num_present_windows_hist1(hist1, supressOutput):
    hist1PresentWindows = 0
    averageHist1PresentWindows = 0
    smallestHist1NumOfPresentWindows = len(hist1)  # Set to max
    largestHist1NumOfPresentWindows = 0  # Set to min
    hist1PresentWindowsList = []

    for j in range(0, len(hist1[0])):
        if j == 163:
            a = 0
        for i in range(0, len(hist1)):
            if hist1[i][j] == "1":
                hist1PresentWindows += 1

        hist1PresentWindowsList.append(hist1PresentWindows)
        hist1PresentWindows = 0

    for i in range(len(hist1PresentWindowsList)):
        averageHist1PresentWindows += hist1PresentWindowsList[i]
        if hist1PresentWindowsList[i] < smallestHist1NumOfPresentWindows:
            smallestHist1NumOfPresentWindows = hist1PresentWindowsList[i]
        if hist1PresentWindowsList[i] > largestHist1NumOfPresentWindows:
            largestHist1NumOfPresentWindows = hist1PresentWindowsList[i]

    averageHist1PresentWindows = averageHist1PresentWindows / len(hist1PresentWindowsList)
    if supressOutput == False:
        print(hist1PresentWindowsList)
        print("Average Number of Present Windows per NP in HIST1 region: ", averageHist1PresentWindows)
        print("Smallest number of Present Windows in an NP in HIST1 region: ", smallestHist1NumOfPresentWindows)
        print("Largest number of Present Windows in an NP in HIST1 region: ", largestHist1NumOfPresentWindows)

    return hist1PresentWindowsList, averageHist1PresentWindows, smallestHist1NumOfPresentWindows, largestHist1NumOfPresentWindows


# Average, smallest, largest number of NPs present in a window in hist1 region
def avg_large_small_num_present_nps_hist1(hist1, supressOutput):
    hist1PresentNPs = 0
    averageHist1PresentNPs = 0
    smallestHist1NumOfPresentNPs = len(hist1[0])  # Set to max
    largestHist1NumOfPresentNPs = 0  # Set to min
    hist1PresentNPsList = []

    for i in range(0, len(hist1)):
        for j in range(0, len(hist1[0])):
            if hist1[i][j] == "1":
                hist1PresentNPs += 1

        hist1PresentNPsList.append(hist1PresentNPs)
        hist1PresentNPs = 0

    print(hist1PresentNPsList)

    for i in range(len(hist1PresentNPsList)):
        averageHist1PresentNPs += hist1PresentNPsList[i]
        if hist1PresentNPsList[i] < smallestHist1NumOfPresentNPs:
            smallestHist1NumOfPresentNPs = hist1PresentNPsList[i]
        if hist1PresentNPsList[i] > largestHist1NumOfPresentNPs:
            largestHist1NumOfPresentNPs = hist1PresentNPsList[i]
    averageHist1PresentNPs = averageHist1PresentNPs / len(hist1PresentNPsList)
    if supressOutput == False:
        print("Average Number of Present NPs per Window in HIST1 region: ", averageHist1PresentNPs)
        print("Smallest number of Present NPs per Window in HIST1 region: ", smallestHist1NumOfPresentNPs)
        print("Largest number of Present NPs per Window in HIST1 region: ", largestHist1NumOfPresentNPs)

    return hist1PresentNPsList, averageHist1PresentNPs, smallestHist1NumOfPresentNPs, largestHist1NumOfPresentNPs

# The average radial position of NPs in HIST1 given from the previous radial position partitions
# Quintiles: 0 - 82, 83 - 163, 164 - 245, 246 - 327, 328 - 407
#              1        2          3          4          5
def avg_radial_position_of_hist1(sortedGWsList, hist1NPs):
    hist1Radial = []
    hist1RadialScores = []
    for i in hist1NPs:
        i -= 3
        for j in range(0, len(sortedGWsList)):
            windows, index = sortedGWsList[j]
            if i == index:
                hist1Radial.append(j)

    averageHist1Radial = 0
    for i in hist1Radial:
        if i <= 82:
            hist1RadialScores.append(1)
        if i >= 83 and i <= 163:
            hist1RadialScores.append(2)
        if i >= 164 and i <= 245:
            hist1RadialScores.append(3)
        if i >= 246 and i <= 327:
            hist1RadialScores.append(4)
        if i >= 328 and i <= 407:
            hist1RadialScores.append(5)
    for i in hist1RadialScores:
        averageHist1Radial += i
    averageHist1Radial = averageHist1Radial / len(hist1Radial)
    print("Average Radial Position in HIST1 Region: ", averageHist1Radial)

    # Getting frequencies of radial scores
    radialFrequencies = Counter(hist1RadialScores)
    print(radialFrequencies)

    return hist1RadialScores, averageHist1Radial

# Estimation of locus volume (higher decile = decondensed, lower decile = condensed)
# After cleaning above 100, total is now 90582
# Quintiles: 0 - 9058, 9059 - 18116, 18117 - 27175, 27176 - 36233, 36234 - 45291, 45292 - 54349, 54350 - 63407, 63408 - 72466, 72467 - 81524, 81525 - 90582
#              1            2              3              4              5              6              7              8              9              10
def avg_locus_volume_of_hist1(sortedPresentNPsList, hist1NPs, hist1StartIndex, hist1StopIndex):
    hist1LocusVolume = []
    hist1LocusVolumeScores = []
    for i in range(hist1StartIndex, hist1StopIndex):
        for j in range(0, len(sortedPresentNPsList)):
            nps, index = sortedPresentNPsList[j]
            if i == index:
                hist1LocusVolume.append(j)

    averageHist1LocusVolume = 0
    for i in hist1LocusVolume:
        if i <= 9058:
            hist1LocusVolumeScores.append(1)
        if i >= 9059 and i <= 18116:
            hist1LocusVolumeScores.append(2)
        if i >= 18117 and i <= 27175:
            hist1LocusVolumeScores.append(3)
        if i >= 27176 and i <= 36233:
            hist1LocusVolumeScores.append(4)
        if i >= 36234 and i <= 45291:
            hist1LocusVolumeScores.append(5)
        if i >= 45292 and i <= 54349:
            hist1LocusVolumeScores.append(6)
        if i >= 54350 and i <= 63407:
            hist1LocusVolumeScores.append(7)
        if i >= 63408 and i <= 72466:
            hist1LocusVolumeScores.append(8)
        if i >= 72467 and i <= 81524:
            hist1LocusVolumeScores.append(9)
        if i >= 81525 and i <= 90582:
            hist1LocusVolumeScores.append(10)
    for i in hist1LocusVolumeScores:
        averageHist1LocusVolume += i
    averageHist1LocusVolume = averageHist1LocusVolume / len(hist1LocusVolume)

    print("Average Locus Volume in HIST1 Region: ", averageHist1LocusVolume)

    # Getting frequencies of radial scores
    locusVolumeFrequencies = Counter(hist1LocusVolumeScores)
    print(locusVolumeFrequencies)

    return

# 163 x 163 matrix for Jaccard Index (Similarities Matrix)
# Formula: ( Number of NPs in Common / Total Number of NPs Present)
def get_jaccard_indexes(hist1, hist1PresentWindowsList):
    NPJaccardRow = []
    NPJaccardIndexes = []

    # print(hist1PresentWindowsList)

    for i in range(len(hist1[0])):
        NPJaccardRow.clear()
        for j in range(len(hist1[0])):
            totalNumNPs = 0
            numNPsInCommon = 0
            NPJaccardIndex = 0
            for k in range(len(hist1)):
                # if hist1[k][i] == "1":
                #   totalNumNPs += 1
                # if hist1[k][j] == "1":
                #   totalNumNPs += 1
                if hist1[k][i] == "1" or hist1[k][j] == "1":
                    if hist1[k][i] == hist1[k][j]:
                        numNPsInCommon += 1
            # Use minimum number of present windows to normalize
            if hist1PresentWindowsList[i] <= hist1PresentWindowsList[j]:
                totalNumNPs = hist1PresentWindowsList[i]
            else:
                totalNumNPs = hist1PresentWindowsList[j]
            # print(totalNumNPs)
            NPJaccardIndex = (numNPsInCommon / totalNumNPs)
            NPJaccardRow.append(NPJaccardIndex)
        print(i, ": ", NPJaccardRow)
        NPJaccardIndexes.append(NPJaccardRow[:])

    return NPJaccardIndexes


# 163 x 163 matrix for Jaccard Distances
# Formula: 1 - Jaccard Index
def get_jaccard_distances(NPJaccardIndexes):
    NPJaccardDistancesRow = []
    NPJaccardDistances = []

    for i in range(len(NPJaccardIndexes)):
        NPJaccardDistancesRow.clear()
        NPJaccardDistancesRow = [1 - j for j in NPJaccardIndexes[i]]
        NPJaccardDistances.append(NPJaccardDistancesRow[:])
        # print(NPJaccardDistancesRow)
    return NPJaccardDistances


# Plot Similarities Matrix
def plot_similarities_matrix(NPJaccardIndexes):
    plt.figure(1)
    sns.heatmap(NPJaccardIndexes, cmap="hot_r", linewidths=0, square=True)
    plt.title("NP Similarities Matrix")
    plt.xlabel("Nuclear Profile Index")
    plt.ylabel("Nuclear Profile Index")


# Plot Distances Matrix
def plot_distances_matrix(NPJaccardDistances):
    plt.figure(2)
    sns.heatmap(NPJaccardDistances, cmap="hot_r", linewidths=0, square=True)
    plt.title("NP Distances Matrix")
    plt.xlabel("Nuclear Profile Index")
    plt.ylabel("Nuclear Profile Index")
    plt.show()


# Remove NPs with under X windows present from hist1
def clean_hist1(hist1, hist1NPs, hist1PresentWindowsList):
    print(hist1PresentWindowsList)
    tempHist1 = []
    tempHist1Row = []
    cleanedHist1NPs = []
    for i in range(0, len(hist1)):
        tempHist1Row.clear()
        for j in range(0, len(hist1[0])):
            if hist1PresentWindowsList[j] > 9:
                tempHist1Row.append(hist1[i][j])
                cleanedHist1NPs.append(j)
        tempHist1.append(tempHist1Row[:])
    hist1 = copy.deepcopy(tempHist1)

    # Store NP location
    cleanedHist1NPsTemp = []

    for i in cleanedHist1NPs:
        if i not in cleanedHist1NPsTemp:
            cleanedHist1NPsTemp.append(i)
    cleanedHist1NPs = cleanedHist1NPsTemp
    cleanedHist1NPsTemp2 = []
    aboveThreshNPs = []
    aboveThreshNPs = copy.deepcopy(cleanedHist1NPs)

    # Map to original NP location
    for i in cleanedHist1NPs:
        cleanedHist1NPsTemp2.append(hist1NPs[i])

    cleanedHist1NPs = cleanedHist1NPsTemp2

    # Set cleanedHist1NPs back 3, so that each corresponds to the column number in the dataset
    cleanedHist1NPsTemp3 = []
    for i in range(0, len(cleanedHist1NPs)):
        cleanedHist1NPsTemp3.append(cleanedHist1NPs[i] - 3)
    cleanedHist1NPs = cleanedHist1NPsTemp3

    return hist1, cleanedHist1NPs, aboveThreshNPs


# Beginning of clustering:
# Randomly select k = 3 distinct data points (generate 3 random NPs) - these can be added to hist1
# These are the initial clusters
def create_random_k_clusters(hist1):
    clusterHist1 = copy.deepcopy(hist1)
    clusterHist1Row = []

    for i in range(0, len(clusterHist1)):
        clusterHist1Row.clear()
        for j in range(0, 3):
            clusterHist1[i].append(str(random.randint(0, 1)))
    return clusterHist1


# Measure distance between each data point and 'k' clusters
# This is the Jaccard Index between an NP and the 'k' clusters
# Returns matrix 3 rows x 163 columns showing Jaccard Index for every NP compared to the 3 'k' clusters
def create_cluster_jaccard_index(clusterHist1, clusterHist1PresentWindowsList):
    clusterNPJaccardIndexes = []
    clusterNPJaccardRow = []

    # print(hist1PresentWindowsList)

    for i in range(len(clusterHist1[0])-3, len(clusterHist1[0])):
        clusterNPJaccardRow.clear()
        for j in range(0, len(clusterHist1[0])-3):
            totalNumNPs = 0
            numNPsInCommon = 0
            clusterNPJaccardIndex = 0
            for k in range(len(hist1)):
                # if hist1[k][i] == "1":
                #   totalNumNPs += 1
                # if hist1[k][j] == "1":
                #   totalNumNPs += 1
                if clusterHist1[k][i] == "1" or clusterHist1[k][j] == "1":
                    if clusterHist1[k][i] == clusterHist1[k][j]:
                        numNPsInCommon += 1

            # Use minimum number of present windows to normalize
            if clusterHist1PresentWindowsList[i] <= clusterHist1PresentWindowsList[j]:
                totalNumNPs = clusterHist1PresentWindowsList[i]
            else:
                totalNumNPs = clusterHist1PresentWindowsList[j]
            # print(totalNumNPs)
            if totalNumNPs != 0:
                clusterNPJaccardIndex = (numNPsInCommon / totalNumNPs)
                clusterNPJaccardRow.append(clusterNPJaccardIndex)
            if totalNumNPs == 0:
                clusterNPJaccardIndex = 0
                clusterNPJaccardRow.append(clusterNPJaccardIndex)
        #print("Cluster ", (i - 162), " Similarities: ", clusterNPJaccardRow)
        clusterNPJaccardIndexes.append(clusterNPJaccardRow[:])

    return clusterNPJaccardIndexes


# Assign each point to the nearest cluster
# Assign each NP to the 'k' cluster with the highest Jaccard Index
# Find highest value in column from previous matrix and assign to 'k' cluster
def assign_to_clusters(clusterNPJaccardIndexes):
    kClusters = []
    kClustersRow1 = []
    kClustersRow2 = []
    kClustersRow3 = []
    kClustersIndexes = []
    kClustersIndexesRow1 = []
    kClustersIndexesRow2 = []
    kClustersIndexesRow3 = []
    kClusterIndex = []
    for j in range(0, len(clusterNPJaccardIndexes[0])):
        largestJaccardIndex = 0
        largestIndex = 0
        kClusterIndex.clear()

        for i in range(0, len(clusterNPJaccardIndexes)):
            if clusterNPJaccardIndexes[i][j] > largestJaccardIndex:
                largestJaccardIndex = clusterNPJaccardIndexes[i][j]
                largestIndex = i
        if largestIndex == 0:
            kClustersRow1.append(largestJaccardIndex)
            kClusterIndex.append(largestIndex)
            kClusterIndex.append(j)
            kClustersIndexesRow1.append(kClusterIndex[:])
        if largestIndex == 1:
            kClustersRow2.append(largestJaccardIndex)
            kClusterIndex.append(largestIndex)
            kClusterIndex.append(j)
            kClustersIndexesRow2.append(kClusterIndex[:])
        if largestIndex == 2:
            kClustersRow3.append(largestJaccardIndex)
            kClusterIndex.append(largestIndex)
            kClusterIndex.append(j)
            kClustersIndexesRow3.append(kClusterIndex[:])
    #print(kClustersRow1)
    #print(kClustersRow2)
    #print(kClustersRow3)
    #print(kClustersIndexesRow1)
    #print(kClustersIndexesRow2)
    #print(kClustersIndexesRow3)
    kClustersIndexes.append(kClustersIndexesRow1[:])
    kClustersIndexes.append(kClustersIndexesRow2[:])
    kClustersIndexes.append(kClustersIndexesRow3[:])
    kClusters.append(kClustersRow1[:])
    kClusters.append(kClustersRow2[:])
    kClusters.append(kClustersRow3[:])
    # print(kClusters)
    # print(kClustersIndexes)

    return kClusters, kClustersIndexes


# Calculate mean of each cluster by finding the normalized consensus of NPs
# Count number of "1"s across NPs in a cluster, if (Num of "1"s / Length of Cluster) = over 50%,
# then value for a Window is 1, else 0
# Compute similarity with the previous cluster value, then assign cluster to be this mean
# Repeat until cluster mean does not change since last iteration
# Makes new hist1 2D arrays for each cluster
def get_cluster_mean(kClusters, kClustersIndexes, clusterHist1, NPJaccardIndexes, meanOrMedoid):
    kClusterMean = []
    meanIsChanging = True
    kClusterMeanHistory = []
    totalRuns = 0

    while meanIsChanging:
        meanIsChanging = False
        kClusterMean.clear()
        highestJaccardIndexIndexList = []
        highestJaccardIndexList = []
        highestJaccardIndexIndexList.clear()
        highestJaccardIndexList.clear()
        totalClusterVariance = 0
        totalSimilarities = 0
        for s in range(0, 3):
            kClusterMeanRow = []
            kClusterMeanRow.clear()
            highestJaccardIndex = 0    # Set to min
            highestJaccardIndexIndex = 0
            clusterVariance = 0

            # New method of choosing mean (medoid)
            # First, find NP in a cluster with the largest average Jaccard Indexes with the other NPs in the cluster
            # Assign this NP as new mean
            if meanOrMedoid == "medoid":
                averageofAverageJaccardIndex = 0
                for i in range(0, len(kClustersIndexes[s])):
                    row, col = kClustersIndexes[s][i]
                    averageJaccardIndex = 0
                    for j in range(0, len(kClustersIndexes[s])):
                        row2, col2 = kClustersIndexes[s][j]
                        averageJaccardIndex += NPJaccardIndexes[col][col2]
                        totalSimilarities += NPJaccardIndexes[col][col2]
                    averageJaccardIndex = averageJaccardIndex / len(kClustersIndexes[s])
                    if averageJaccardIndex > highestJaccardIndex:
                        highestJaccardIndex = averageJaccardIndex
                        highestJaccardIndexIndex = col
                # Variance calculation:
                for i in range(0, len(kClustersIndexes[s])):
                    row, col = kClustersIndexes[s][i]
                    clusterVariance += (NPJaccardIndexes[highestJaccardIndexIndex][col] - highestJaccardIndex)**2

                highestJaccardIndexIndexList.append(highestJaccardIndexIndex)
                highestJaccardIndexList.append(highestJaccardIndex)
                totalClusterVariance += clusterVariance

            # Old method of choosing mean
            if meanOrMedoid == "mean":
                for i in range(0, len(clusterHist1)):
                    totalOnes = 0
                    totalOnesInHist1 = 0
                    meanValue = "0"
                    for j in range(0, len(kClustersIndexes[s]) - 3):
                        row, col = kClustersIndexes[s][j]
                        # print(clusterHist1[i][col])
                        if clusterHist1[i][col] == "1":
                            totalOnes += 1
                    # This is the normalization
                    for j in range(0, len(clusterHist1[s]) - 3):
                        if clusterHist1[i][j] == "1":
                            totalOnesInHist1 += 1
                    if totalOnesInHist1 > 0:
                        if (totalOnes / totalOnesInHist1) >= 0.5:
                            meanValue = "1"
                    kClusterMeanRow.append(meanValue)

                kClusterMean.append(kClusterMeanRow[:])


        if meanOrMedoid == "medoid":
            # Change clusterHist1 to include new medoids
            for i in range(0, len(clusterHist1)):
                for j in range(len(clusterHist1[0])-3, len(clusterHist1[0])):
                    # If the new mean is different from the old mean, set equal to the new mean
                    # This means function should iterate again because mean is still changing
                    if clusterHist1[i][j] != clusterHist1[i][highestJaccardIndexIndexList[j - (len(clusterHist1[0])-3)]]:
                        meanIsChanging = True
                        clusterHist1[i][j] = clusterHist1[i][highestJaccardIndexIndexList[j - (len(clusterHist1[0])-3)]]

        if meanOrMedoid == "mean":
        # Change clusterHist1 to include new means
            for i in range(0, len(clusterHist1)):
                for j in range(len(clusterHist1[0])-3, len(clusterHist1[0])):
                    # If the new mean is different from the old mean, set equal to the new mean
                    # This means function should iterate again because mean is still changing
                    if clusterHist1[i][j] != kClusterMean[j - len(clusterHist1[0])][i]:
                        meanIsChanging = True
                        clusterHist1[i][j] = kClusterMean[j - len(clusterHist1[0])][i]

        kClusterMeanHistory.append(highestJaccardIndexIndexList)

        # If the medoid has flip-flopped twice, end (meanIsChanging = false)
        if totalRuns > 2:
            if kClusterMeanHistory[totalRuns] == kClusterMeanHistory[totalRuns-2]:
                if kClusterMeanHistory[totalRuns-1] == kClusterMeanHistory[totalRuns-3]:
                    meanIsChanging = False

        totalRuns += 1
        print("Total Runs: ", totalRuns)
        print("'k' Cluster Indexes: ", highestJaccardIndexIndexList)

        # Reassign NPs to new mean 'k' clusters
        clusterHist1PresentWindowsList, averageClusterHist1PresentNPs, smallestClusterHist1NumOfPresentWindows, largestClusterHist1NumOfPresentWindows = avg_large_small_num_present_windows_hist1(
            clusterHist1, True)
        clusterNPJaccardIndexes = create_cluster_jaccard_index(clusterHist1, clusterHist1PresentWindowsList)
        kClusters, kClustersIndexes = assign_to_clusters(clusterNPJaccardIndexes)

        # Compute average similarities for each 'k' cluster, total average
        # Find how many NPs assigned to each cluster

    # Return average similarities, indexes, new clusterHist1, total cluster variance

    return highestJaccardIndexList, highestJaccardIndexIndexList, clusterHist1, totalClusterVariance, kClustersIndexes, totalSimilarities

# Heatmap of best cluster run
def heatmap_best_clusters(kClustersIndexes, hist1):
    bestClusterHist1Heatmap = []
    bestClusterHist1HeatmapRow = []
    for s in range(0, 3):
        for i in range(0, len(kClustersIndexes[s])):
            bestClusterHist1HeatmapRow.clear()
            row, col = kClustersIndexes[s][i]
            for j in range(0, len(hist1)):
                bestClusterHist1HeatmapRow.append(int(hist1[j][col]))
            bestClusterHist1Heatmap.append(bestClusterHist1HeatmapRow[:])
    fig = plt.figure(1, figsize=(12, 5))
    ax = fig.add_subplot(111)
    sns.heatmap(ax=ax, data=bestClusterHist1Heatmap, cmap="hot_r", linewidths=0)
    #for i in range(0, (len(kClustersIndexes[0]) // 3) + 1):
    #    ax.get_yticklabels()[i].set_color("red")
    #for i in range((len(kClustersIndexes[0]) // 3) + 1, (len(kClustersIndexes[1]) // 3) + (len(kClustersIndexes[0]) // 3) + 1):
    #    ax.get_yticklabels()[i].set_color("green")
    #for i in range((len(kClustersIndexes[1]) // 3) + (len(kClustersIndexes[0]) // 3) + 1, 21):
    #    ax.get_yticklabels()[i].set_color("blue")
    plt.title("Best Cluster Heatmap")
    plt.xlabel("Genomic Windows")
    plt.ylabel("Nuclear Profiles")
    return

# Percentage of Windows in each NP that have Histone genes and LADs
# LAD can be found at column 9 of feature table, Hist1 at column 13
def get_features(hist1, kClustersIndexes, clusterHist1PresentWindowsList, featureTable):
    histones = []
    histoneCount = 0
    histonePercentage = 0
    histonesRow = []
    LADs = []
    LADCount = 0
    LADRow = []
    LADPercentage = 0
    for s in range(0, 3):
        histonesRow.clear()
        LADRow.clear()
        for row, col in kClustersIndexes[s]:
            histoneCount = 0
            LADCount = 0
            for i in range(0, len(hist1)):
                if featureTable[i+1][13] == "1" and hist1[i][col] == "1":
                    histoneCount += 1
                if featureTable[i+1][9] == "1" and hist1[i][col] == "1":
                    LADCount += 1
            histonePercentage = histoneCount / clusterHist1PresentWindowsList[col]
            histonesRow.append(histonePercentage)
            LADPercentage = LADCount / clusterHist1PresentWindowsList[col]
            LADRow.append(LADPercentage)
        histones.append(histonesRow[:])
        LADs.append(LADRow[:])

    return histones, LADs

# Make box plot of Histone and LAD results
def boxplot_features(histones, LADs):
    plt.figure(2, figsize=(5, 4))
    plt.boxplot(histones, labels=["Cluster 1", "Cluster 2", "Cluster 3"])
    plt.title("Histone Percentages")
    plt.ylabel("Percent of NPs with Histone")
    plt.figure(3, figsize=(5, 4))
    plt.boxplot(LADs, labels=["Cluster 1", "Cluster 2", "Cluster 3"])
    plt.title("LAD Percentages")
    plt.ylabel("Percent of NPs with LAD")

    return

# Make bar graph of radial positions
def bargraph_radial_pos(hist1Radial, averageHist1Radial, cleanedHist1NPs, bestKClustersIndexes, aboveThreshNPs):

    # Trim radial scores to be after cleaning
    afterCleanRadial = []
    for i in aboveThreshNPs:
        afterCleanRadial.append(hist1RadialScores[i])

    # Divide into clusters
    clusterRadial = []
    clusterRadialRow = []
    for i in range(0, len(bestKClustersIndexes)):
        clusterRadialRow.clear()
        for j in range(0, len(bestKClustersIndexes[i])):
            row, col = bestKClustersIndexes[i][j]
            clusterRadialRow.append(afterCleanRadial[col])
        clusterRadial.append(clusterRadialRow[:])

    print("Radial Positions by Cluster: ", clusterRadial)

    radialFrequencies1 = Counter(clusterRadial[0])
    radialFrequencies2 = Counter(clusterRadial[1])
    radialFrequencies3 = Counter(clusterRadial[2])
    radFreqList1 = []
    radFreqList2 = []
    radFreqList3 = []
    for i in range(1, 6):
        radFreqList1.append(radialFrequencies1[i])
    for i in range(1, 6):
        radFreqList2.append(radialFrequencies2[i])
    for i in range(1, 6):
        radFreqList3.append(radialFrequencies3[i])
    x = [1, 2, 3, 4, 5]

    plt.figure(4)
    width = 0.3
    plt.bar(np.arange(1, 6), radFreqList1, width=width)
    plt.bar(np.arange(1, 6) + width, radFreqList2, width=width)
    plt.bar(np.arange(1, 6) + width + width, radFreqList3, width=width)
    plt.title("Radial Frequencies")
    plt.xlabel("Radial Score")
    plt.ylabel("Frequency")
    plt.show()

    return

# Find how often windows are present by percentage in a cluster, repeat for all clusters
def get_window_frequencies(hist1, kClustersIndexes, cleanedHist1NPs):
    clusterPresentNPs = []
    clusterPresentNPsRow = []

    for s in range(0,3):
        clusterPresentNPsRow.clear()
        for i in range(0, len(hist1)):
            presentNPs = 0
            for j in range(0, len(kClustersIndexes[s])):
                row, col = kClustersIndexes[s][j]
                if hist1[i][col] == "1":
                    presentNPs += 1
            clusterPresentNPsRow.append(presentNPs)
        clusterPresentNPs.append(clusterPresentNPsRow[:])


    return clusterPresentNPs

# Choose most present windows, like over 90% of NPs, then create a "prototype" of each cluster
def create_cluster_prototypes(clusterPresentNPs, kClusterIndexes):
    clusterPrototypes = []
    clusterPrototypesRow = []
    for i in range(0,3):
        clusterPrototypesRow.clear()
        for j in clusterPresentNPs[i]:
            if j >= ( 0.4 * len(kClustersIndexes[i])):
                clusterPrototypesRow.append(1)
            else:
                clusterPrototypesRow.append(0)
        clusterPrototypes.append(clusterPrototypesRow[:])
        print(clusterPrototypesRow)


    return clusterPrototypes

def get_intersection(clusterPrototype1, clusterPrototype2):
    intersection = []
    intersectionValue = 0
    intersectionNumerator = 0
    intersectionDenominator = 0
    count1 = 0
    count2 = 0
    for i in range(0, len(clusterPrototype1)):
        if clusterPrototype1[i] == 1 and clusterPrototype2[i] == 1:
            intersection.append(1)
            intersectionNumerator += 1
            intersectionDenominator += 1
        else:
            intersection.append(0)
            if clusterPrototype1[i] == 1:
                count1 += 1
            if clusterPrototype2[i] == 1:
                count2 += 1
    if count1 < count2:
        intersectionDenominator += count1
    else:
        intersectionDenominator += count2
    intersectionValue = intersectionNumerator / intersectionDenominator

    return intersection, intersectionValue

# Find lowest intersection, get unity of these two, then take intersection with third
# If intersection is high, third contains the other two
def get_unity(clusterPrototype1, clusterPrototype2):
    union = []
    for i in range(0, len(clusterPrototype1)):
        if clusterPrototype1[i] == 1 or clusterPrototype2[i] == 1 :
            union.append(1)
        else:
            union.append(0)
    return union

def create_intersection_heatmap(intersectionValue12, intersectionValue13, intersectionValue23):
    intersectionMatrix = [[1, intersectionValue12, intersectionValue13], [intersectionValue12, 1, intersectionValue23], [intersectionValue13, intersectionValue23, 1]]
    sns.heatmap(intersectionMatrix, cmap="Greens")
    plt.title("Intersection of Clusters")
    plt.show()

    return

# Compute detection frequency (Fa) for all windows in HIST1
def get_detection_frequencies(hist1):
    detectionFrequencies = []
    for i in range(0, len(hist1)):
        detectionFrequency = 0
        count = 0
        for j in range(0, len(hist1[0])):
            if hist1[i][j] == "1":
                count += 1
        detectionFrequency = count / len(hist1[0])
        detectionFrequencies.append(detectionFrequency)

    return detectionFrequencies

# Compute co-segregation (Fab) for all window pairs in HIST1
def get_cosegregations(hist1):
    cosegregations = []
    cosegregationRow = []
    for i in range(0, len(hist1)):
        cosegregationRow.clear()
        for k in range(0, len(hist1)):
            cosegregation = 0
            count = 0
            for j in range(0, len(hist1[0])):
                if hist1[i][j] == "1" and hist1[k][j] == "1":
                    count += 1
            cosegregation = count / len(hist1[0])
            cosegregationRow.append(cosegregation)
        cosegregations.append(cosegregationRow[:])

    return cosegregations

# Compute linkage (D) for all pairs, D = fab - fafb
def get_linkages(detectionFrequencies, cosegregations):
    linkages = []
    linkageRow = []
    for i in range(0, len(detectionFrequencies)):
        linkageRow.clear()
        for j in range(0, len(detectionFrequencies)):
            linkage = cosegregations[i][j] - (detectionFrequencies[i] * detectionFrequencies[j])
            linkageRow.append(linkage)
        linkages.append(linkageRow[:])

    return linkages

# Compute maximum linkage (Dmax) for all pairs
def get_max_linkages(detectionFrequencies, linkages):
    maxLinkages = []
    maxLinkagesRow = []
    for i in range(0, len(linkages)):
        maxLinkagesRow.clear()
        for j in range(0, len(linkages[0])):
            maxLinkage = 0
            if linkages[i][j] >= 0:
                term1 = detectionFrequencies[i] * detectionFrequencies[j]
                term2 = (1 - detectionFrequencies[i]) * (1 - detectionFrequencies[j])
                if term1 < term2:
                    maxLinkage = term1
                else:
                    maxLinkage = term2
            if linkages[i][j] < 0:
                term1 = detectionFrequencies[j] * (1 - detectionFrequencies[i])
                term2 = detectionFrequencies[i] * (1 - detectionFrequencies[j])
                if term1 < term2:
                    maxLinkage = term1
                else:
                    maxLinkage = term2
            maxLinkagesRow.append(maxLinkage)
        maxLinkages.append(maxLinkagesRow[:])


    return maxLinkages

# Compute normalized linkage (D') for all pairs
def get_norm_linkages(linkages, maxLinkages):
    normLinkages = []
    normLinkagesRow = []
    print("Normalized Linkages Table: ")
    for i in range(0, len(linkages)):
        normLinkagesRow.clear()
        for j in range(0, len(linkages[0])):
            if linkages[i][j] != 0 and maxLinkages[i][j] != 0:
                normLinkage = linkages[i][j] / maxLinkages[i][j]
            else:
                normLinkage = 0
            normLinkagesRow.append(normLinkage)
        print(normLinkagesRow)
        normLinkages.append(normLinkagesRow[:])

    return normLinkages

# Heatmap of normalized linkage
def create_normlinkage_heatmap(normLinkages):
    sns.heatmap(normLinkages)
    plt.title("Normalized Linkages")
    plt.xlabel("Window")
    plt.ylabel("Window")
    plt.show()

    return

# Compute normalized linkage average (D'avg) for all pairs
def get_norm_linkage_average(normLinkages):
    normLinkageAverage = 0
    sum = 0
    for i in range(0, len(normLinkages)):
        for j in range(0, len(normLinkages[0])):
            sum += normLinkages[i][j]

    normLinkageAverage = sum / (len(normLinkages) * len(normLinkages[0]))
    print("Normalized Linkage Average: ", normLinkageAverage)

    return normLinkageAverage

# Get edges of all window pairs when D'(A,B) > D'avg
def get_norm_linkage_edges(normLinkages, normLinkageAverage):
    normLinkEdges = []
    normLinkEdgesRow = []
    print("NormLinkEdges: ")
    for i in range(0, len(normLinkages)):
        normLinkEdgesRow.clear()
        for j in range(0, len(normLinkages[0])):
            if normLinkages[i][j] > normLinkageAverage:
                edge = i, j
                normLinkEdgesRow.append(edge)
        print(normLinkEdgesRow)
        normLinkEdges.append(normLinkEdgesRow[:])

    return normLinkEdges

# Compute degree centrality for all windows (unique windows / total windows - 1)
# Unique windows = (total (don't include A == B in total)) / 2
# Note: total windows != length of normLinkEdges, they must be counted
def get_degree_centrality(normLinkEdges):
    degreeCentrality = []
    windowCount = 0

    for s in range(0, len(normLinkEdges)):
        count = 0
        for i in range(0, len(normLinkEdges)):
            for j in range(0, len(normLinkEdges[i])):
                A, B = normLinkEdges[i][j]
                if A != B:
                    if A == s or B == s:
                        count += 1
        if count != 0:
            windowCount += 1
        degreeCentrality.append( (count/2) )
    degreeCentrality[:] = [x / (windowCount - 1) for x in degreeCentrality]
    print(degreeCentrality)

    return degreeCentrality

# Rank degree centrality in ascending order, max, min, avg
def get_degree_centrality_stats(degreeCentrality):
    sortedDegCent = []
    maxDegCent = 0
    minDegCent = 0
    avgDegCent = 0

    sortedDegCent = sorted(((v, i) for i, v in enumerate(degreeCentrality)))
    print("Degree Centrality | Window Number")
    for i in range(len(sortedDegCent)):
        x, y = sortedDegCent[i]
        print(x, " | ", y)
        avgDegCent += x
    maxDegCent = sortedDegCent[len(sortedDegCent) - 1][0]
    minDegCent = sortedDegCent[0][0]
    avgDegCent = avgDegCent / len(sortedDegCent)
    print("Maximum Degree Centrality: ", maxDegCent)
    print("Minimum Degree Centrality: ", minDegCent)
    print("Average Degree Centrality: ", avgDegCent)

    return sortedDegCent, maxDegCent, minDegCent, avgDegCent

# Get hubs (top 5 nodes in degree centrality) and communities (all their egdes)
def get_hubs_and_comms(sortedDegCent, normLinkEdges):
    hubs = []
    communities = []
    communitiesRow = []
    communities = []
    communitiesRow = []

    for i in reversed(range(0, 5)):
        x, y = sortedDegCent[len(sortedDegCent) - 1 - i]
        hubs.append(y)

    for s in hubs:
        communitiesRow.clear()
        for j in range(0, len(normLinkEdges[s])):
            x, y = normLinkEdges[s][j]
            if x != y:
                if s == x:
                    communitiesRow.append(y)
                if s == y:
                    communitiesRow.append(x)
        communities.append(communitiesRow[:])
        print("Hub ", s, ", Community Size = ", len(communitiesRow), ", List: ", communitiesRow)

    return hubs, communities

# Gets percentage of nodes in a community with Hist1/LAD
def get_comm_features(hubs, communities, featureTable):
    commLadPercentages = []
    commLadSums = []
    commHist1Percentages = []
    commHist1Sums = []

    for i in range(0, len(communities)):
        commLadSum = 0
        commHist1Sum = 0
        for j in range(0, len(communities[i])):
            if featureTable[communities[i][j] + 1][9] == "1":
                commLadSum += 1
            if featureTable[communities[i][j] + 1][13] == "1":
                commHist1Sum += 1
        commLadSums.append(commLadSum)
        commHist1Sums.append(commHist1Sum)

    for i in range(0, len(communities)):
        commLadPercentages.append(commLadSums[i] / len(communities[i]))
        commHist1Percentages.append(commHist1Sums[i] / len(communities[i]))

    print("LAD Percentages: ", commLadPercentages)
    print("HIST1 Percentages: ", commHist1Percentages)


    return commLadPercentages, commHist1Percentages

# Create 81 x 81 array of edges for each community where 1 is present and 0 is not present
# This will be a matrix of the edges of all nodes in the community that link to each other (not to outside)
def create_comm_matrix(hubs, communities, normLinkEdges):
    commMatrixes = []
    commMatrixRow = []
    commMatrix = []

    for s in range(0, len(communities)):
        commMatrix.clear()
        for i in range(0, 81):
            commMatrixRow.clear()
            for j in range(0, 81):
                foundi = False
                foundj = False
                foundEdge = False
                for k in range(0, len(communities[s])):
                    if communities[s][k] == i:
                        foundi = True
                    if communities[s][k] == j:
                        foundj = True
                    if foundi == True and foundj == True:
                        break
                if foundi == True and foundj == True:
                    for n in range(0, len(normLinkEdges)):
                        for m in range(0, len(normLinkEdges[n])):
                            x, y = normLinkEdges[n][m]
                            if x == i and y == j:
                                foundEdge = True
                                break
                        if foundEdge == True:
                            break
                if foundEdge == True:
                    commMatrixRow.append(1)
                else:
                    commMatrixRow.append(0)
            commMatrix.append(commMatrixRow[:])
        commMatrixes.append(commMatrix[:])



    return commMatrixes

# Create a heatmap of edges for each community
def create_comm_heatmap(commMatrixes):
    plt.figure(1)
    sns.heatmap(commMatrixes[0])
    plt.title("Community 1 (Hub = 19)")
    plt.xlabel("Windows")
    plt.ylabel("Windows")

    plt.figure(2)
    sns.heatmap(commMatrixes[1])
    plt.title("Community 2 (Hub = 37)")
    plt.xlabel("Windows")
    plt.ylabel("Windows")

    plt.figure(3)
    sns.heatmap(commMatrixes[2])
    plt.title("Community 3 (Hub = 48)")
    plt.xlabel("Windows")
    plt.ylabel("Windows")

    plt.figure(4)
    sns.heatmap(commMatrixes[3])
    plt.title("Community 4 (Hub = 34)")
    plt.xlabel("Windows")
    plt.ylabel("Windows")

    plt.figure(5)
    sns.heatmap(commMatrixes[4])
    plt.title("Community 5 (Hub = 13)")
    plt.xlabel("Windows")
    plt.ylabel("Windows")

    plt.show()

    return

# Gets numOfGWs
numOfGWs = get_print_numofgws()

# Gets numOfNPs
numOfNPs = get_print_numofnps()

# Acquires a list of: count of present windows in NPs.
# Note: Data not cleaned, largest outlier is 4.5 stdevs out.
presentWindowsList = get_list_present_windows(presentWindows, presentWindowsList)

# Average, smallest, and largest number of present genomic windows per NP
averageNumOfGWs, smallestNumOfPresentWindows, largestNumOfPresentWindows = avg_large_small_num_present_windows(numOfGWs,
                                                                                                               presentWindowsList)
# Calculate standard deviation of NPs per GW
stDevPresentGWs = st_dev_present_gws(presentWindowsList, averageNumOfGWs)

# Estimate radial position of each NP: sort list and partition into quintiles (Higher = equatorial, lower = apical)
# After cleaning above 100, total is now 90582
# Deciles: 0 - 41, 42 - 82, 83 - 122, 123 - 163, 164 - 204, 205 - 245, 246 - 286, 287 - 327, 328 - 367, 368 - 408
#            1        2        3          4          5          6          7          8          9          10
sortedGWsList = sorted(((v, i) for i, v in enumerate(presentWindowsList)))
# Order: (Present Windows, Pre-Sort Index)

# Acquires a list of: count of NPs in a given genomic window
presentNPsList = get_present_nps_list(data, numOfGWs, numOfNPs)

# Average, smallest, and largest number of NPs per genomic window
averageNumOfNPs, smallestNumOfPresentNPs, largestNumOfPresentNPs = avg_large_small_num_present_nps(numOfNPs,
                                                                                                   presentNPsList)

# Calculate standard deviation of NPs per GW
stDevPresentNPs = st_dev_present_gws(presentNPsList, averageNumOfNPs)

# Create a scatter plot of NPs per GW
# total_windows_plot(averageNumOfNPs, presentNPsList, stDevPresentNPs)

# Estimation of locus volume (higher decile = decondensed, lower decile = condensed)
# After cleaning above 100, total is now 90582
# Quintiles: 0 - 18116, 18117 - 36233, 36234 - 54349, 54350 - 72466, 72467 - 90582
#              1              2              3              4              5
sortedPresentNPsList = sorted(((v, i) for i, v in enumerate(presentNPsList)))

# Find HIST1 region and extract (Chr13, 21.7 Mb Start - 24.1 Mb Stop)
hist1, hist1StartIndex, hist1StopIndex = extract_hist1_region(numOfGWs, data)

# 81 windows in hist1 region, 163 NPs
# Extract relevant NPs in HIST1
hist1NPs = extract_relevant_hist1_nps(hist1)

# Construct new hist1 with only relevant NPs (extracted relevant NPs)
hist1 = construct_relevant_hist1_nps(hist1)

# Average, smallest, largest number of windows present in NPs in hist1 region
hist1PresentWindowsList, averageHist1PresentWindows, smallestHist1NumOfPresentNPs, largestHist1NumOfPresentNPs = avg_large_small_num_present_windows_hist1(
    hist1, False)

# Average, smallest, largest number of NPs present in a window in hist1 region
hist1PresentNPsList, averageHist1PresentNPs, smallestHist1NumOfPresentWindows, largestHist1NumOfPresentWindows = avg_large_small_num_present_nps_hist1(
    hist1, False)

# Fix: these should be the average radial position given from the previous radial position partitions
# Estimate radial position of each NP: sort list and partition into quintiles (Higher = equatorial, lower = apical)
# Deciles: 0 - 16, 17 - 33, 34 - 49, 50 - 65, 66 - 82, 83 - 98, 99 - 114, 115 - 130, 131 - 145, 146 - 163
#            1        2        3        4        5        6        7          8          9          10
hist1RadialScores, averageHist1Radial = avg_radial_position_of_hist1(sortedGWsList, hist1NPs)

# Estimation of locus volume (higher decile = decondensed, lower decile = condensed)
# Quintiles: 0 - 16, 17 - 32, 33 - 49, 50 - 65, 66 - 81
#              1        2        3        4        5
avg_locus_volume_of_hist1(sortedPresentNPsList, hist1NPs, hist1StartIndex, hist1StopIndex)

# 163 x 163 matrix for Jaccard Index (Similarities Matrix)
# Formula: ( Number of NPs in Common / Total Number of NPs Present)
NPJaccardIndexes = get_jaccard_indexes(hist1, hist1PresentWindowsList)

# 163 x 163 matrix for Jaccard Distances (Distance Matrix)
# Formula: 1 - Jaccard Index
NPJaccardDistances = get_jaccard_distances(NPJaccardIndexes)

# Plot Similarities Matrix
plot_similarities_matrix(NPJaccardIndexes)

# Plot Distances Matrix
plot_distances_matrix(NPJaccardDistances)

# Beginning of clustering:
# Randomly select k = 3 distinct data points (generate 3 random NPs) - these can be added to hist1
# These are the initial clusters, returning matrix will be 81 rows x 166 columns
# clusterHist1 = create_random_k_clusters(hist1)

# Get new present windows list

# Measure distance between each data point and 'k' clusters
# This is the Jaccard Index between an NP and the 'k' clusters
# Returns matrix 3 rows x 163 columns showing Jaccard Index for every NP compared to the 3 'k' clusters
# clusterNPJaccardIndexes = create_cluster_jaccard_index(clusterHist1, clusterHist1PresentWindowsList)

# Assign each point to the nearest cluster
# Assign each NP to the 'k' cluster with the highest Jaccard Index
# Find highest value in column from previous matrix and assign to 'k' cluster
# The returned indexes are in form, [0, 4] (for example) or ['k' Cluster, NP Number]
# kClusters, kClustersIndexes = assign_to_clusters(clusterNPJaccardIndexes)

# Calculate medoid/mean of each cluster by finding the consensus of NPs

# Repeat the above function for certain number of iterations, select clusterHist1
# with best similarities/variance/other metric
bestJaccardIndexList = []
bestJaccardIndexIndexList = []
bestClusterHist1 = []
bestTotalClusterVariance = 99999999 # set to max
bestTotalJaccardIndexes = 0
totalJaccardIndexes = 0
bestKClustersIndexes = []
bestRun = 0
bestTotalSimilariites = 0

# Remove NPs with under 12 windows present from hist1, cleanedHist1NPs corresponds to the NP number in the hist1 array
# e.g. if cleanedHist1NPs[1] = 9, this means that the second NP included can be found at data[i][9]
# cleanedHistNPs are essentially the names of the NPs included in hist1
hist1, cleanedHist1NPs, aboveThreshNPs = clean_hist1(hist1, hist1NPs, hist1PresentWindowsList)

for i in range(0, 100):
    clusterHist1 = create_random_k_clusters(hist1)
    clusterHist1PresentWindowsList, averageClusterHist1PresentNPs, smallestClusterHist1NumOfPresentWindows, largestClusterHist1NumOfPresentWindows = avg_large_small_num_present_windows_hist1(
        clusterHist1, True)
    clusterNPJaccardIndexes = create_cluster_jaccard_index(clusterHist1, clusterHist1PresentWindowsList)
    kClusters, kClustersIndexes = assign_to_clusters(clusterNPJaccardIndexes)
    highestJaccardIndexList, highestJaccardIndexIndexList, clusterHist1, totalClusterVariance, kClustersIndexes, totalSimilarities = get_cluster_mean(kClusters, kClustersIndexes, clusterHist1, NPJaccardIndexes, "medoid")
    print("K-Means Cluster Run", i, "Variance: ", totalClusterVariance)
    #print("--------------------------------------------------------------------------------------------")
    #print("--------------------------------------------------------------------------------------------")
    totalJaccardIndexes = highestJaccardIndexList[0] + highestJaccardIndexList[1] + highestJaccardIndexList[2]
    # By variance:
    #if totalClusterVariance < bestTotalClusterVariance:
    # By greatest similarities:
    #if totalJaccardIndexes > bestTotalJaccardIndexes:
    # By sum of similarities:
    if totalSimilarities > bestTotalSimilariites:
        bestTotalJaccardIndexes = totalJaccardIndexes
        bestJaccardIndexList = highestJaccardIndexList
        bestJaccardIndexIndexList = highestJaccardIndexIndexList
        bestClusterHist1 = clusterHist1
        bestTotalClusterVariance = totalClusterVariance
        bestKClustersIndexes = kClustersIndexes
        bestRun = i

print("Best Run: ", bestRun)
print("Best Total Similarities: ", totalSimilarities)
print("Variance: ", bestTotalClusterVariance)
print("Jaccard Indexes: ", bestJaccardIndexList)
print("Jaccard Index Total: ", bestTotalJaccardIndexes)
print("Medoid NPs: ", bestJaccardIndexIndexList)
print("Best HIST1 with Clusters: ", bestClusterHist1)

# Heatmap of best cluster run
print("Number in Cluster 1: ", len(bestKClustersIndexes[0]))
print("Number in Cluster 2: ", len(bestKClustersIndexes[1]))
print("Number in Cluster 3: ", len(bestKClustersIndexes[2]))
print("Number in Cluster 1: ", bestKClustersIndexes[0])
print("Number in Cluster 2: ", bestKClustersIndexes[1])
print("Number in Cluster 3: ", bestKClustersIndexes[2])
print("HIST1 NPs after Filtering: ", cleanedHist1NPs)
heatmap_best_clusters(bestKClustersIndexes, hist1)

# Percentage of Windows in each NP that have Histone genes and LADs
# LAD can be found at column 9 of feature table, Hist1 at column 13
histones, LADs = get_features(hist1, bestKClustersIndexes, clusterHist1PresentWindowsList, featureTable)

# Make box plot of Histone and LAD results
boxplot_features(histones, LADs)

# Make bar graph of radial positions
bargraph_radial_pos(hist1RadialScores, averageHist1Radial, cleanedHist1NPs, bestKClustersIndexes, aboveThreshNPs)

# Find how often windows are present by percentage in a cluster, repeat for all clusters
clusterPresentNPs = get_window_frequencies(hist1, bestKClustersIndexes, cleanedHist1NPs)

# Choose most present windows, like over 90% of NPs, then create a "prototype" of each cluster
clusterPrototypes = create_cluster_prototypes(clusterPresentNPs, bestKClustersIndexes)

# Find the intersection of each prototype with each other
# Should find that two are complementary, and a third contains the other two
intersection12, intersectionValue12 = get_intersection(clusterPrototypes[0], clusterPrototypes[1])
print("Intersection of 1 and 2: ", intersection12)
print("Intersection Value of 1 and 2: ", intersectionValue12)
intersection13, intersectionValue13 = get_intersection(clusterPrototypes[0], clusterPrototypes[2])
print("Intersection of 1 and 3: ", intersection13)
print("Intersection Value of 1 and 3: ", intersectionValue13)
intersection23, intersectionValue23 = get_intersection(clusterPrototypes[1], clusterPrototypes[2])
print("Intersection of 2 and 3: ", intersection23)
print("Intersection Value of 2 and 3: ", intersectionValue23)

# Find lowest intersection, get unity of these two, then take intersection with third
# If intersection is high, third contains the other two
if intersectionValue12 < intersectionValue13 :
    if intersectionValue12 < intersectionValue23 :
        union = get_unity(clusterPrototypes[0], clusterPrototypes[1])
        print("Union of 1 and 2: ", union)
        intersection, intersectionValue = get_intersection(union, clusterPrototypes[2])
        print("Intersection of Union (of 1 and 2) and 3: ", intersection)
        print("Intersection Value of Union (of 1 and 2) and 3: ", intersectionValue)
    else:
        union = get_unity(clusterPrototypes[1], clusterPrototypes[2])
        print("Union of 2 and 3: ", union)
        intersection, intersectionValue = get_intersection(union, clusterPrototypes[0])
        print("Intersection of Union (of 2 and 3) and 1: ", intersection)
        print("Intersection Value of Union (of 2 and 3) and 1: ", intersectionValue)
else:
    if intersectionValue13 < intersectionValue23 :
        union = get_unity(clusterPrototypes[0], clusterPrototypes[2])
        print("Union of 1 and 3: ", union)
        intersection, intersectionValue = get_intersection(union, clusterPrototypes[1])
        print("Intersection of Union (of 1 and 3) and 2: ", intersection)
        print("Intersection Value of Union (of 1 and 3) and 2: ", intersectionValue)
    else:
        union = get_unity(clusterPrototypes[1], clusterPrototypes[2])
        print("Union of 2 and 3: ", union)
        intersection, intersectionValue = get_intersection(union, clusterPrototypes[0])
        print("Intersection of Union (of 2 and 3) and 1: ", intersection)
        print("Intersection Value of Union (of 2 and 3) and 1: ", intersectionValue)

# Create 3x3 heatmap of intersections
create_intersection_heatmap(intersectionValue12, intersectionValue13, intersectionValue23)

# Compute detection frequency (Fa) for all windows in HIST1
detectionFrequencies = get_detection_frequencies(hist1)

# Compute co-segregation (Fab) for all window pairs in HIST1
cosegregations = get_cosegregations(hist1)

# Compute linkage (D) for all pairs, D = fab - fafb
linkages = get_linkages(detectionFrequencies, cosegregations)

# Compute maximum linkage (Dmax) for all pairs
maxLinkages = get_max_linkages(detectionFrequencies, linkages)

# Compute normalized linkage (D') for all pairs
normLinkages = get_norm_linkages(linkages, maxLinkages)

# Heatmap of normalized linkage
create_normlinkage_heatmap(normLinkages)

# Compute normalized linkage average (D'avg) for all pairs
normLinkageAverage = get_norm_linkage_average(normLinkages)

# Get edges of all window pairs when D'(A,B) > D'avg
normLinkEdges = get_norm_linkage_edges(normLinkages, normLinkageAverage)

# Compute degree centrality for all windows (unique windows / total windows - 1)
# Unique windows = (total ( - 1 if A == B included)) / 2
# Note: total windows != length of normLinkEdges, they must be counted
degreeCentrality = get_degree_centrality(normLinkEdges)

# Rank degree centrality in ascending order, max, min, avg
sortedDegCent, maxDegCent, minDegCent, avgDegCent = get_degree_centrality_stats(degreeCentrality)

# Get hubs (top 5 nodes in degree centrality) and communities (all their egdes)
hubs, communities = get_hubs_and_comms(sortedDegCent, normLinkEdges)

# Gets percentage of nodes in a community with Hist1/LAD
commLadPercentages, commHist1Percentages = get_comm_features(hubs, communities, featureTable)

# Create 81 x 81 array of edges for each community where 1 is present and 0 is not present
# This will be a matrix of the edges of all nodes in the community that link to each other (not to outside)
commMatrixes = create_comm_matrix(hubs, communities, normLinkEdges)

# Create a heatmap of edges for each community
create_comm_heatmap(commMatrixes)





