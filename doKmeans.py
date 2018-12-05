from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

class doKmeans:

    df = pd.DataFrame() # store a data frame of the reading data form
    k = 0 # number of centroids
    times = 0 # running times of kmeans

    # a big list contains three small lists (Jun, Jul, Aug), a small list has data points of SWD in a month
    # this list can be seen as a intermediate variable, will not be used outside
    dataList = []

    # these four lists can be seen as output variables, will be used to make transition matrix
    sprayList = [] # a big list contaions small lists which are number of sprays in Jun, Jul, and Aug
    labels = [] # a big list contains three small lists which are labels of states in each of three months
    centroids = []
    std_centroids = []

    # variable to make mean distance plot
    kk = 0 # centroids start from 1 to kk
    tt = 0 # running time of kmeans

    def __init__(self, filename, k = 0, times = 0, kk = 0, tt = 0):
        self.df = pd.read_excel(filename)
        if self.df.empty:
            return
        self.k = k
        self.times = times
        self.kk = kk
        self.tt = tt
        self.generateData()
        if len(self.dataList) == 0:
            return
        self.generateClusters()

    # this function reads data frame and fill two lists which are dataList and sprayList
    # the dataList is used to do kmeans
    # the sprayList is used to make transition matrix later
    def generateData(self):

        dataListJun = []
        dataListJul = []
        dataListAug = []
        sprayList = []

        df = self.df
        field = ''
        trap = ''
        monthSWD = [[0, 0], [0, 0], [0, 0]]
        monthSpray = [0, 0, 0]
        pushFlag = 0
        for i in range(df.shape[0]):
            if df['Field'][i] == field and df['Trap'][i] == trap:
                pass
            else:
                field = df['Field'][i]
                trap = df['Trap'][i]
                monthSWD = [[0, 0], [0, 0], [0, 0]]
                monthSpray = [0, 0, 0]
                pushFlag = 0
            if df['Month'][i] == "Jun":
                data = [df['Mean.SWD.Male'][i], df['Mean.SWD.Female'][i]]
                monthSWD[0] = data
                monthSpray[0] = df['Number.Of.Sprays'][i]
            elif df['Month'][i] == "Jul":
                data = [df['Mean.SWD.Male'][i], df['Mean.SWD.Female'][i]]
                monthSWD[1] = data
                monthSpray[1] = df['Number.Of.Sprays'][i]
            else:
                data = [df['Mean.SWD.Male'][i], df['Mean.SWD.Female'][i]]
                monthSWD[2] = data
                monthSpray[2] = df['Number.Of.Sprays'][i]
            pushFlag += 1
            if pushFlag == 3:
                dataListJun.append(monthSWD[0])
                dataListJul.append(monthSWD[1])
                dataListAug.append(monthSWD[2])
                sprayList.append(monthSpray)
                pushFlag = 0
                monthSWD = [[0, 0], [0, 0], [0, 0]]
                monthSpray = [0, 0, 0]

        self.dataList.append(dataListJun)
        self.dataList.append(dataListJul)
        self.dataList.append(dataListAug)
        self.sprayList = sprayList

    # this function use dataList to do kmeans
    # this function fills labels and centroids
    def generateClusters(self):

        labelsJun = []
        labelsJul = []
        labelsAug = []
        cenJun = []
        cenJul = []
        cenAug = []
        stdcenJun = []
        stdcenJul = []
        stdcenAug = []
        month = 6

        #print("Clustering starting")
        # TODO: code here can be simpler
        for dl in self.dataList:

            #print("Clustering month " + str(month))
            X = np.array(dl)
            k = self.k
            times = self.times
            kmeans = KMeans(init='k-means++', n_clusters=k, n_init=times)
            kmeans = kmeans.fit(X)
            if month == 6:
                labelsJun = kmeans.labels_
                cenJun = kmeans.cluster_centers_
                stdcenJun = StandardScaler().fit_transform(cenJun)
            if month == 7:
                labelsJul = kmeans.labels_
                cenJul = kmeans.cluster_centers_
                stdcenJul = StandardScaler().fit_transform(cenJul)
            if month == 8:
                labelsAug = kmeans.labels_
                cenAug = kmeans.cluster_centers_
                stdcenAug = StandardScaler().fit_transform(cenAug)

            month += 1

        self.labels.append(labelsJun)
        self.labels.append(labelsJul)
        self.labels.append(labelsAug)
        self.centroids.append(cenJun)
        self.centroids.append(cenJul)
        self.centroids.append(cenAug)
        self.std_centroids.append(stdcenJun)
        self.std_centroids.append(stdcenJul)
        self.std_centroids.append(stdcenAug)

        #print("Clustering done")

    # draw a 2D plot of clusters in a month
    def generatePlot(self, monthNum):
        # monthNum value: 0 is Jun, 1 is Jul, 2 is Aug
        if monthNum > 2 or monthNum < 0:
            print("wrong month number")
            return
        data = self.dataList[monthNum]
        labels = self.labels[monthNum]
        finalDf = pd.concat([pd.DataFrame(data=data, columns = ['male', 'female']), pd.DataFrame(data=labels, columns = ['labels'])], axis=1)
        x = np.array(finalDf['male'])
        y = np.array(finalDf['female'])
        ax = plt.axes()
        ax.scatter(x, y, c=finalDf["labels"], s=10)
        ax.set_xlabel('Male SWD', fontsize=10)
        ax.set_ylabel('Female SWD', fontsize=10)
        if monthNum == 0:
            month = 'June'
        elif monthNum == 1:
            month = 'July'
        else:
            month = 'August'
        ax.set_title('Clusters of ' + month, fontsize=15)
        plt.show()

    # make a plot to show mean distance between data points and their cluster centroid
    # show the plot for 1 cluster, 2 clusters ... k clusters
    def makeMeanDisPlot(self, data):

        if len(data) == 0:
            return

        dist_list = []
        x = []

        for i in range(1, self.kk+1):
            kmeans = KMeans(init='k-means++', n_clusters=i, n_init=self.tt)
            # Fitting the input data
            kmeans = kmeans.fit(data)
            # Centroid values
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_
            dist = 0
            for j in range(len(data)):
                dist += np.linalg.norm(data[j] - centroids[labels[j]])
            dist_list.append(dist)
            x.append(i)

        if len(dist_list) == 0:
            return

        plt.plot(x, dist_list)
        plt.show()





