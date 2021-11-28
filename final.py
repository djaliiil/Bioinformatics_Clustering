from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPalette, QBrush, QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, \
    QGridLayout, QListWidget, QComboBox, QLineEdit, QFileDialog, QLabel, QCalendarWidget, QSizePolicy,\
    QSplashScreen, QProgressBar, QLCDNumber, QFrame, QShortcut, QAction
from PyQt5.QtCore import QSize, QThread, QTime, QTimer
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from random import randint
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtQuick import QQuickView
import time
from datetime import datetime
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
import pandas as pd
import pandas.plotting as pdplt
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.offline
import random


def import_data(pwd):
    data = pd.read_excel(r''+str(pwd))
    df = pd.DataFrame(data)
    return df

# Calculer la distance de Levenshtein
def distance_levenshtein(mot1, mot2):
    m1 = len(mot1)+2
    m2 = len(mot2)+2
    a = [[0]*m1]*m2
    mat = np.mat(a, dtype=np.dtype(object))
    for i in range(2,m1):
        mat[1,i] = i-1
        mat[0,i] = mot1[i-2]
    for i in range(2,m2):
        mat[i,1] = i-1
        mat[i,0] = mot2[i-2]
    mat[1,1] = 0
    mat[0,0], mat[0,1], mat[1,0] = ' ', ' ', ' '
    for i in range(2,m1):
        for j in range(2,m2):
            if( mat[0,i] == mat[j,0]):
                mat[j,i] = min( int(mat[j-1,i-1]),  int(mat[j-1,i]+1),  int(mat[j,i-1]+1) )
            else:        
                mat[j,i] = min( int(mat[j-1,i-1]+1),  int(mat[j-1,i]+1),  int(mat[j,i-1]+1) )             
    dist = mat[mat.shape[0]-1,mat.shape[1]-1]
    return dist


def new_center(liste):
    seq = ""    
    dist = []
    for i in range(len(liste[0])):
        pos = []
        for j in range(len(liste)):
            pos.append(liste[j][i])
        a = pos.count('A')
        c = pos.count('C')
        g = pos.count('G')
        t = pos.count('T')
        gap = pos.count('-')
        mx = a
        ltmx = 'A'
        if(mx < c):
            mx = c
            ltmx = 'C'
        if(mx < g):
            mx = g
            ltmx = 'G'
        if(mx < t):
            mx = t
            ltmx = 'T'
        if(mx < gap):
            mx = gap
            ltmx = '-'
        seq += ltmx
    for i in range(len(liste)):
        dist.append(distance_levenshtein(liste[i],seq))
    index = dist.index(min(dist))
    return liste[index]

def init_centers_kmeans(df, k):
    n = df.shape[0]
    centers = []
    i = 0
    while (i < k):
        x = random.randint(0, n-1)
        if(x not in centers):
            centers.append(list(df['Sequence'])[x])
            i += 1
    return centers


def intra_class(clusters, centers):
    somme = 0

    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            somme += distance_levenshtein(clusters[i][j], centers[i])

    return somme

def inter_class(clusters, cg):
    somme = 0

    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            somme += distance_levenshtein(clusters[i][j], cg[i])

    return somme


def clustering_kmeans(df, centers, n, k):
    n_clusters = []
    for i in range(len(centers)):
        n_clusters.append([])   
    for i in range(n):
        dist = []

        if(df[i] not in centers):
            for j in range(len(centers)):
                dist.append(distance_levenshtein(df[i], centers[j]))
            index = dist.index(min(dist))   
            n_clusters[index].append(df[i])
    
    return n_clusters

def centre_global(df):
    cg = new_center(list(df['Sequence']))
    return cg

def clustering_kmedoids(df, centers, n, k):
    n_clusters = []
    for i in range(len(centers)):
        n_clusters.append([])
    for i in range(n):
        dist = []
        if(df[i] not in centers):
            for j in range(len(centers)):
                dist.append(distance_levenshtein(df[i], centers[j]))
            index = dist.index(min(dist))   
            n_clusters[index].append(df[i])
    return n_clusters

def init_medoids(df, k):
    l = list(df['Sequence'])
    n = len(l)
    centers = []
    i = 0
    while (i < k):
        x = random.randint(0, n-1)
        if(l[x] not in centers):
            centers.append(l[x])
            i += 1
    return centers

def init_medoids_clus(clusters):
    print("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/")
    print(clusters)
    print("/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/")
    centers = []
    for i in range(len(clusters)):
        n = len(clusters[i])
        c = random.randint(0, n-1)
        if(len(clusters[i][c]) != 0):
            centers.append(clusters[i][c])
        else:
            centers.append()
    
    return centers


def silhouette_a(clus):
    n = len(clus) 
    l = []
    for i in range(len(clus)):
        a = 0
        for j in range(len(clus)):
            a += distance_levenshtein(clus[i], clus[j])
        a /= (n-1)
        l.append(a)
    return l


def silhouette_b(clusters, clus):
    s = 0
    l = []
    alll = []

    for i in range(len(clus)):
        l = []
        for j in range(len(clusters)):
            if(clus != clusters):
                n = len(clusters[j])
                s = 0
                for k in range(len(clusters[j])):
                    s += distance_levenshtein(clus[i], clusters[j][k])
                l.append(s)      
        alll.append(min(l))
    return alll


def silhouette(clusters):
    silhouette = []
    for i in range(len(clusters)):
        s = []
        clus = clusters[i]
        a = silhouette_a(clus)
        b = silhouette_b(clusters, clus)

        print("**  clus : ",len(clus), "     a : ", len(a), "    b : ", len(b))
        for j in range(len(clus)):
            val = (b[j] - a[j]) / (max(a[j], b[j]))
            s.append(val)
        silhouette.append(s)

    print(silhouette)
    return silhouette


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    labels[P] = C
    i = 0
    while i < len(NeighborPts):                    
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
           labels[Pn] = C        
        elif labels[Pn] == 0:
            labels[Pn] = C            
            PnNeighborPts = regionQuery(D, Pn, eps)            
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts            
        i += 1


def regionQuery(D, P, eps):
    neighbors = []
    for Pn in range(0, len(D)):
        # If the distance is below the threshold, add it to the neighbors list.
        
        if (distance_levenshtein(D[P], D[Pn]) < eps):
           neighbors.append(Pn)       
    return neighbors


def indice_min(mat):
    min=mat[0][1]
    ind=(0,1)
    for i in range(len(mat)):
        for j in range(len(mat)):
            if(i != j):
                if(mat[i][j] < min):
                    min = mat[i][j]
                    ind=(i,j)
    return ind


def min_matrix(mat):
    mini = mat[0,1]
    a = 0
    b = 1
    for i in range(0,len(mat)-1):
        for j in range(i+1,len(mat)):
            if(mat[i,j] < mini):
                mini = mat[i,j]
                a=i
                b=j
    return (mini, a, b)


def matrix_distance(data):
    length = len(data)
    mat = []
    for i in range(length):
        mat.append(np.array([0]*length))
    for i in range(0,len(mat)):
        for j in range(i,len(mat)):
            s = 0
            coeff = 0
            for k in range(len(data[i])):
                for l in range(len(data[j])):
                    s += distance_levenshtein(data[i][k], data[j][l]) 
                    coeff += 1
            s = s / coeff
            mat[i][j] = s
    return np.array(mat)


def affect_clusters(ite):
    clusters = [[]]*(len(ite))
    for i in range(len(ite)):
        clusters[i].append(ite[i])

def group(dataset, index, ind):
    mx = max(ind[1], ind[2])
    mn = min(ind[1], ind[2])

    dataset[mn].extend(dataset[mx])
    index[mn].extend(index[mx])
    dataset.pop(mx)
    index.pop(mx)

    return (dataset, index)


def dendrogram(data, dist):
    x = np.random.rand(len(dist), len(dist))
    names = list(data)
    dendro = ff.create_dendrogram(dist, labels=names)
    dendro['layout'].update({'width':1300, 'height':650})
    plotly.offline.plot(dendro, filename='simple_dendrogram')



def MyKMEANS(df, k):
    nump = np.array(df)
    columns = list(df.axes[1])
    # Number of training data
    n = df.shape[0]
    # Number of features in the data
    c = df.shape[1]
    centers = init_centers_kmeans(df, k)
    clusters = [[]]*k
    all_centers = []
    all_centers.append(centers)
    clusters = clustering_kmeans(list(df['Sequence']), centers, n, k)
    n_centers = []
    for i in range(100):
        print(">>>>>>>>>>>>>>>>>>>> Itération : ",i,"<<<<<<<<<<<<<<<<<<<")
        llll = [[]]*k
        for i in range(len(centers)):
            llll[i] = new_center(clusters[i])
        n_centers = llll
        clusters = clustering_kmeans(list(df['Sequence']), n_centers, n, k)    
        print("___________ NC : ", n_centers)
        
        if(n_centers in all_centers):
            print("1.   ", n_centers)
            print("2.   ", all_centers)
            print("!!!!!!!!!!!!!!!!! STOP !!!!!!!!!!!!!!")
            break        
        else:
            all_centers.append(n_centers)
            print("3.   ", n_centers)
            print("4.   ", all_centers)
        
        centers = n_centers
    for i in range(len(centers)):
        print("=============== Cluster : ",i," ===============")
        print("\tCentre : ", centers[i])
        print("\tNouveau Centre : ", n_centers[i])
        print(clusters[i])
        print("===============================================")

    intra = intra_class(clusters, centers)
    inter = inter_class(clusters, centre_global(df))
    inertie = intra/inter

    print(inertie)

    return (clusters, centers, intra, inter, inertie)


def MyKMEDOIDS(df, k):
    nump = np.array(df)
    columns = list(df.axes[1])
    # Number of training data
    n = df.shape[0]
    # Number of features in the data
    c = df.shape[1]
    centers = init_medoids(df, k)
    clusters = [[]]*k
    all_centers = []
    all_clusters = []
    inercie = []
    all_centers.append(centers)
    clusters = clustering_kmedoids(list(df['Sequence']), centers, n, k)
    all_clusters.append(clusters)
    n_centers = []
    intra = intra_class(clusters, centers)
    inter = inter_class(clusters, centre_global(df))
    val = intra/inter
    inn = val
    inercie.append(val)
    bool = 10
    for i in range(100):
        print(">>>>>>>>>>>>>>>>>>>> Itération : ",i,"<<<<<<<<<<<<<<<<<<<")        
        n_centers = init_medoids(df, k)
        clusters = clustering_kmedoids(list(df['Sequence']), n_centers, n, k)    

        intra = intra_class(clusters, centers)
        inter = inter_class(clusters, centre_global(df))
        val = intra/inter
        
            
        print("___________ NC : ", n_centers)
        
        if(val >= min(inercie) and bool <= 0):
            print("1.   ", n_centers)
            print("2.   ", all_centers)
            print("!!!!!!!!!!!!!!!!! STOP !!!!!!!!!!!!!!")
            break
                
        else:
            if(bool > 0 and val >= min(inercie)):
                bool -= 1
            elif(val < min(inercie)):
                all_clusters.append(clusters)
                inercie.append(val)
                inn = val
                bool = 10
                all_centers.append(n_centers)
                print("3.   ", n_centers)
                print("4.   ", all_centers)
        centers = n_centers
    for i in range(len(centers)):
        print("=============== Cluster : ",i," ===============")
        print("\tCentre : ", centers[i])
        print("\tNouveau Centre : ", n_centers[i])
        print(clusters[i])
        print("--- inertie intra/inter class ---->>>>>  ", inercie[len(inercie)-1] ,"  <<<<<")
        print("===============================================")

    print(all_clusters[len(all_clusters)-1])
    print(inercie[len(inercie)-1])

    return (clusters, centers, intra, inter, inn)


def MyDBSCAN(D, eps, MinPts):
    
    labels = [0]*len(D)
    print(len(labels))
    C = 0
    for P in range(0, len(D)):
        if not (labels[P] == 0):
           continue
        NeighborPts = regionQuery(D, P, eps)
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        else: 
           C += 1
           growCluster(D, labels, P, NeighborPts, C, eps, MinPts)

    noise = []
    clusters = [[]]*(max(labels))

    for i in range(len(clusters)):
        l = []
        for j in range(len(labels)):
            if((labels[j]-1) == i):
                l.append(j)
        clusters[i] = l

    for i in range(len(labels)):
        if(labels[i] == -1):
            noise.append(i)  

    centers = ['']*(len(clusters))

    print(df)

    clus = [[]]*(len(clusters))
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            clus[i].append(D[clusters[i][j]])

    noi = []
    for i in range(len(noise)):    
        noi.append(D[noise[i]])

    for i in range(len(clus)):
        matrix = matrix_distance(clus[i])
        somme = [[0]]*len(clus[i])
        s = matrix.sum(1)
        for j in range(len(clus[i])):
            somme[j] = s[j]
        val = s.min()
        for j in range(len(clus[i])):
            if(somme[i] == val):
                ind = j
                break
        
        centers[i] = clus[i][j]


    intra = intra_class(clus, centers)
    inter = inter_class(clus, centre_global(df))
    inertie = intra/inter


    return(clus, centers, noi, intra, inter, inertie)          


def MyAGNES(df):
    nump = np.array(df)
    columns = list(df.axes[1])
    n = df.shape[0]
    c = df.shape[1]

    print(n)
    new_dataset = list(df['Sequence'])
    index = [i for i in range(n)]
    
    sequence = list(df['Sequence'])
    for i in range(len(new_dataset)):
        new_dataset[i] = [new_dataset[i]]
        sequence[i] = [sequence[i]]
        index[i] = [index[i]]
    print("\n",index,"\n")

    liste, clusters, dataset, all_mat_dist, all_iteration = [] , [],[], [], []

    dataset.append(new_dataset)
    all_iteration.append(index)
    
    while(len(new_dataset) > 1):
        mat_dist = matrix_distance(new_dataset)
        all_mat_dist.append(mat_dist)
        minimum = min_matrix(mat_dist)
        (new_dataset, index) = group(new_dataset, index, minimum)
        print("\n",index,"\n")
        
        dataset.append(new_dataset)
        all_iteration.append(index)
        print("================================================")
        print(new_dataset)
        print("\n")

    print(all_iteration[len(all_iteration)-1])
    #dendrogram(df['Sequence'], matrix_distance(sequence))


    return (all_iteration, n)





# On va tout d'abord créer notre fenetre (window)
#-------------------------------------------------------
class window(QtWidgets.QMainWindow):
    def __init__(self):
        QWidget.__init__(self)
        #------ positionner la fenetre
        self.setGeometry(70, 50, 1200, 645)
        #----- titre de la fenetre
        self.setWindowTitle("Data Mining")
        #----- icon de la fenetre
        #self.setWindowIcon(QtGui.QIcon("icon.png"))
        #---- on va creer l'arriere plan
        
        oImage = QImage("adn2.jpg")
        sImage = oImage.scaled(QSize(950, 650))
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(sImage))
        self.setPalette(palette)
        
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        #------- le 1er GroupBox qui contient la zone d'affichage (INPUT)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(200, 5, 470, 580))
        self.groupBox.setObjectName("groupBox")
        self.edit = QtWidgets.QTextEdit(self.groupBox)
        self.edit.setGeometry(QtCore.QRect(10, 30, 445, 530))
        self.edit.setObjectName("textEdit")
        self.edit.setReadOnly(True)
        self.edit.setStyleSheet("QTextEdit {    background-color: #19232D;    color: #F0F0F0;    border: 1px solid #32414B;}")
        self.edit.setLineWrapColumnOrWidth(20000)
        self.edit.setLineWrapMode(QtWidgets.QTextEdit.FixedPixelWidth)

        #------- le 2eme GroupBox qui contient la zone d'affichage (OUTPUT)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(700, 5, 470, 580))
        self.groupBox_2.setObjectName("groupBox_2")
        self.edit2 = QtWidgets.QTextEdit(self.groupBox_2)
        self.edit2.setGeometry(QtCore.QRect(10, 30, 445, 530))
        self.edit2.setReadOnly(True)
        self.edit2.setObjectName("textEdit_2")
        self.edit2.setStyleSheet("QTextEdit {    background-color: #19232D;    color: #F0F0F0;    border: 1px solid #32414B;}")
        self.edit2.setLineWrapColumnOrWidth(20000)
        self.edit2.setLineWrapMode(QtWidgets.QTextEdit.FixedPixelWidth)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        #------- le 3eme GroupBox qui contient la zone d'affichage des calcules (%GC, A, T, G, C, Masse)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(30, 300, 150, 280))
        self.groupBox_3.setObjectName("groupBox_3")

        self.inertie_intra= QtWidgets.QGroupBox(self.groupBox_3)
        self.inertie_intra.setStyleSheet("QGroupBox { font-weight: bold;    border: 2px solid #32414B;    border-radius: 4px;    padding: 4px;margin-top: 16px;}")
        self.inertie_intra.setGeometry(QtCore.QRect(25, 15, 100, 60))
        self.intra = QtWidgets.QTextEdit(self.inertie_intra)
        self.intra.setDisabled(True)
        self.intra.setStyleSheet("QTextEdit {    background-color: #19232D;    color: #F0F0F0;    border: 1px solid #32414B;}")
        self.intra.setGeometry(QtCore.QRect(5, 22, 90, 30))

        self.inertie_inter = QtWidgets.QGroupBox(self.groupBox_3)
        self.inertie_inter.setStyleSheet("QGroupBox { font-weight: bold;    border: 2px solid #32414B;    border-radius: 4px;    padding: 4px;margin-top: 16px;}")
        self.inertie_inter.setGeometry(QtCore.QRect(25, 80, 100, 60))
        self.inter = QtWidgets.QTextEdit(self.inertie_inter)
        self.inter.setDisabled(True)
        self.inter.setStyleSheet("QTextEdit {    background-color: #19232D;    color: #F0F0F0;    border: 1px solid #32414B;}")
        self.inter.setGeometry(QtCore.QRect(5, 22, 90, 30))

        self.inertie = QtWidgets.QGroupBox(self.groupBox_3)
        self.inertie.setStyleSheet("QGroupBox { font-weight: bold;    border: 2px solid #32414B;    border-radius: 4px;    padding: 4px;margin-top: 16px;}")
        self.inertie.setGeometry(QtCore.QRect(25, 145, 100, 60))
        self.inn = QtWidgets.QTextEdit(self.inertie)
        self.inn.setDisabled(True)
        self.inn.setStyleSheet("QTextEdit {    background-color: #19232D;    color: #F0F0F0;    border: 1px solid #32414B;}")
        self.inn.setGeometry(QtCore.QRect(5, 22, 90, 30))

        self.temps = QtWidgets.QGroupBox(self.groupBox_3)
        self.temps.setStyleSheet("QGroupBox { font-weight: bold;    border: 2px solid #32414B;    border-radius: 4px;    padding: 4px;margin-top: 16px;}")
        self.temps.setGeometry(QtCore.QRect(25, 210, 100, 60))
        self.tt = QtWidgets.QTextEdit(self.temps)
        self.tt.setDisabled(True)
        self.tt.setStyleSheet("QTextEdit {    background-color: #19232D;    color: #F0F0F0;    border: 1px solid #32414B;}")
        self.tt.setGeometry(QtCore.QRect(5, 22, 90, 30))


        #------- le 5eme GroupBox qui regroupe les boutons (valide, arn, complement, ...)
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(30, 10, 150, 280))
        self.groupBox_5.setObjectName("groupBox_4")

        self.kmeans = QtWidgets.QPushButton(self.groupBox_5)
        self.kmeans.setGeometry(QtCore.QRect(25, 30, 100, 41))
        self.kmeans.setMouseTracking(True)
        self.kmeans.setTabletTracking(True)
        self.kmeans.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.kmeans.setStyleSheet("QPushButton {background-color: #AAAAAA; border-radius: 10px;} ; button:hover span:after {opacity: 1;right: 0;}")
        self.kmeans.setObjectName("kmeans")
        self.kmeans.setDisabled(True)
        self.kmeans.clicked.connect(self.kmeans_dialog)

        self.kmedoids = QtWidgets.QPushButton(self.groupBox_5)
        self.kmedoids.setGeometry(QtCore.QRect(25, 90, 100, 41))
        self.kmedoids.setMouseTracking(True)
        self.kmedoids.setTabletTracking(True)
        self.kmedoids.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.kmedoids.setStyleSheet("QPushButton {background-color: #AAAAAA; border-radius: 10px; border-color: #FFAAAA;} ; button:hover span:after {opacity: 1;right: 0;}")
        self.kmedoids.setObjectName("kmedoids")
        self.kmedoids.setDisabled(True)
        self.kmedoids.clicked.connect(self.kmedoids_dialog)

        self.dbscan = QtWidgets.QPushButton(self.groupBox_5)
        self.dbscan.setGeometry(QtCore.QRect(25, 150, 100, 41))
        self.dbscan.setMouseTracking(True)
        self.dbscan.setTabletTracking(True)
        self.dbscan.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.dbscan.setStyleSheet("QPushButton {background-color: #AAAAAA; border-radius: 10px;} ; button:hover span:after {opacity: 1;right: 0;}")
        self.dbscan.setObjectName("dbscan")
        self.dbscan.setDisabled(True)
        self.dbscan.clicked.connect(self.dbscan_dialog)

        self.agnes = QtWidgets.QPushButton(self.groupBox_5)
        self.agnes.setGeometry(QtCore.QRect(25, 210, 100, 41))
        self.agnes.setMouseTracking(True)
        self.agnes.setTabletTracking(True)
        self.agnes.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.agnes.setStyleSheet("QPushButton {background-color: #AAAAAA; border-radius: 10px;} ; button:hover span:after {opacity: 1;right: 0;}")
        self.agnes.setObjectName("agnes")
        self.agnes.setDisabled(True)
        self.agnes.clicked.connect(self.agnes_funct)


        
        
        
        

        self.left_spacer = QWidget()
        self.left_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_spacer = QWidget()
        self.right_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.save = QAction(QIcon('save.png'), '&Save', self)
        self.reset = QAction(QIcon('clear.png'), '&Clear', self)
        self.open = QAction(QIcon('open.png'), '&Open File (.Xlsx)', self)

        self.title = QLabel("DATA MINING")
        self.title.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Black))


        self.toolbar = self.addToolBar('toolbar')
        self.toolbar.addAction(self.open)
        self.toolbar.addWidget(self.right_spacer)
        self.toolbar.addWidget(self.title)
        self.toolbar.addWidget(self.left_spacer)
        self.toolbar.addAction(self.save)
        self.toolbar.addSeparator();
        self.toolbar.addAction(self.reset)
        
        self.save.triggered.connect(self.saved)
        self.reset.triggered.connect(self.resetall)
        self.open.triggered.connect(self.filechoose)


        """
        #------- le 7eme GroupBox qui contient la zone d'affichage de statut de l'application (operation en cours d'execution)
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(30, 10, 150, 60))
        self.groupBox_7.setObjectName("groupBox_7")

        self.stat = QtWidgets.QTextEdit(self.groupBox_7)
        self.stat.setGeometry(QtCore.QRect(10, 20, 130, 35))
        self.stat.setStyleSheet("QTextEdit {    background-color: #19232D;    color: #F0F0F0;    border: 1px solid #32414B;}")
        self.stat.setObjectName("Generate")
        self.stat.setDisabled(True)
        self.stat.setObjectName("textEdit_2")
        """

        #-------- appel du methode qui contient les raccourci clavier et les valeurs des composants de l'application
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        #-------- pour la visualisation de l'application
        self.show()
        return app.exec_()

    #-------- appel du methode qui contient les raccourci clavier et les valeurs des composants de l'application
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Data Mining"))
        self.shortcut = QShortcut(QKeySequence("Esc"), self)
        self.shortcut.activated.connect(self.on_close)

        
        #self.setStyleSheet("QGroupBox {background-image: url(adn2.jpg);}")

        self.toolbar.setStyleSheet("QGroupBox {font-weight: bold; font: bold; border: 2px solid #32414B; margin-top: 6px ;background-color: #ADFFFA; border-radius: 4px}")

        self.groupBox.setTitle(_translate("MainWindow", "DataSet"))
        self.groupBox.setStyleSheet("QGroupBox {font-weight: bold; font: bold; border: 2px solid #32414B; margin-top: 6px ;background-color: #FFFFFF; border-radius: 4px}")

        self.groupBox_2.setTitle(_translate("MainWindow", "OUTPUT"))
        self.groupBox_2.setStyleSheet("QGroupBox {font-weight: bold; font: bold; border: 2px solid #32414B; margin-top: 6px; background-color: #FFFFFF; border-radius: 4px}")

        self.groupBox_3.setTitle(_translate("MainWindow", "Performances"))
        self.groupBox_3.setStyleSheet("QGroupBox {font-weight: bold; font: bold; border: 2px solid #32414B; margin-top: 6px; background-color: #FFFFFF; border-radius: 4px}")

        self.groupBox_5.setTitle(_translate("MainWindow", "Clustering Methods"))
        self.groupBox_5.setStyleSheet("QGroupBox {font-weight: bold; font: bold; border: 2px solid #32414B; margin-top: 6px; background-color: #FFFFFF; border-radius: 4px;}")
        

        """
        self.groupBox_7.setTitle(_translate("MainWindow", "Method"))
        self.groupBox_7.setStyleSheet("QGroupBox {font-weight: bold; font: bold; border: 2px solid #32414B; margin-top: 6px; background-color: #FFFFFF; border-radius: 4px}")
        """

        self.save.setText("SAVE")
        self.reset.setText("RESET")

        self.kmeans.setText("K-Means")
        self.kmedoids.setText("K-Medoids")
        self.dbscan.setText("DB-Scan")        
        self.agnes.setText("AGNES")

        
        self.inertie_inter.setTitle(_translate("MainWindow", "Inter-Class"))
        self.inertie_intra.setTitle(_translate("MainWindow", "Intra-Class"))
        self.inertie.setTitle(_translate("MainWindow", "Inercie"))
        self.temps.setTitle(_translate("MainWindow", "Time"))

        self.open.setShortcut("Ctrl+o")
        self.save.setShortcut("Ctrl+s")
        self.reset.setShortcut("Ctrl+z")


#############################
# raccourci clavier ("Esc")
    @pyqtSlot()
    def on_close(self):
        sys.exit()
###############################

####################################
# la fonction qui rend les champs vides
    @pyqtSlot()
    def resetall(self):
        self.edit.setText("")
        self.edit2.setText("")
        #self.stat.setText("")
        self.inn.setText("")
        self.intra.setText("")
        self.inter.setText("")
        self.tt.setText("")
        file_data = []
        df = None
###################################
        

##############################################################################################################################
# pour ouvrir un fichier Fasta
    @pyqtSlot()
    def filechoose(self):
        global df

        #self.stat.setText("Open File ..")
        self.edit.setText("")
        self.edit2.setText("")
        #self.stat.setText("")
        self.inn.setText("")
        self.intra.setText("")
        self.inter.setText("")
        self.tt.setText("")
        file_data = []
        df = None

        self.kmeans.setDisabled(False)
        self.kmedoids.setDisabled(False)
        self.dbscan.setDisabled(False)
        self.agnes.setDisabled(False)

        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home/djalil/Bureau',
                                            filter='Excel Files(*.xlsx)')
        print("*** ", fname[0], " ***")
        
        df = import_data(str(fname[0]))
        print(df)
        print("-------------------")
        self.edit.setText(str(df))  
        #self.stat.setText("File opened ..")

        
##############################################################################################################################

##############################################################################################################################

    @pyqtSlot()
    def kmedoids_dialog(self):
       try:
            
            self.edit2.setText("")
            #self.stat.setText("")
            self.inn.setText("")
            self.intra.setText("")
            self.inter.setText("")
            self.tt.setText("")
            file_data = []
            df = None

            #self.stat.setText("K-Medoids ..")
            self.dialog = QDialog(self)

            self.dialog.setMinimumSize(200,150)
            self.dialog.close()
            self.dialog.setWindowTitle("K-Medoids")
            """
            oImage1 = QImage("adn3(1).jpg")
            sImage1 = oImage1.scaled(QSize(230, 150))
            palette1 = QPalette()
            palette1.setBrush(QPalette.Background, QBrush(sImage1))
            self.dialog.setPalette(palette1)
            """
            Hbox = QHBoxLayout()
            Hbox2 = QHBoxLayout()
            Hbox3 = QHBoxLayout()
            Vbox = QVBoxLayout()
            lab1 = QLabel("Partitioning with K-Medoids")
            lab2 = QLabel("Number of clusters : ")
            self.liste1 = QComboBox()
            self.liste1.setMinimumSize(30,30)
            for i in range(1,11):
                self.liste1.addItem(str(i))
            txt = QLineEdit()
            txt.setMaximumSize(100,30)
            btn = QPushButton("Start")
            btn.setMinimumSize(70,30)
            btn.setStyleSheet('QPushButton {background-color: #A3C1DA; border-radius: 4px}')
            btn.clicked.connect(self.kmedoids_funct)
            Hbox3.addStretch()
            Hbox3.addWidget(btn)
            Hbox2.addWidget(lab2)
            Hbox2.addStretch()
            Hbox2.addWidget(self.liste1)
            Hbox.addWidget(lab1)
            Vbox.addItem(Hbox)
            Vbox.addItem(Hbox2)
            Vbox.addItem(Hbox3)
            self.dialog.setLayout(Vbox)
            self.dialog.show()
            self.dialog.move(500, 300)

       except:
            print("---------------------------------------")


##############################################################################################################################

    @pyqtSlot()
    def dbscan_dialog(self):
       try:

            
            self.edit2.setText("")
            #self.stat.setText("")
            self.inn.setText("")
            self.intra.setText("")
            self.inter.setText("")
            self.tt.setText("")
            file_data = []
            df = None

            #self.stat.setText("DB-Scan ..")
            self.dialog = QDialog(self)
            self.dialog.setMinimumSize(200,150)
            self.dialog.close()
            self.dialog.setWindowTitle("K-Medoids")
            """
            oImage1 = QImage("adn3(1).jpg")
            sImage1 = oImage1.scaled(QSize(230, 150))
            palette1 = QPalette()
            palette1.setBrush(QPalette.Background, QBrush(sImage1))
            self.dialog.setPalette(palette1)
            """
            Hbox = QHBoxLayout()
            Hbox2 = QHBoxLayout()
            Hbox3 = QHBoxLayout()
            Hbox4 = QHBoxLayout()
            Vbox = QVBoxLayout()
            lab1 = QLabel("Clustering with DB-Scan")
            lab2 = QLabel("Min Points : ")
            lab3 = QLabel("Epsilon : ")
            self.eps = QComboBox()
            self.minpts = QComboBox()
            self.eps.setMinimumSize(30,30)
            self.minpts.setMinimumSize(30,30)
            for i in range(1,51):
                self.eps.addItem(str(i))
            for i in range(2,11):
                self.minpts.addItem(str(i))
            txt = QLineEdit()
            txt.setMaximumSize(100,30)
            btn = QPushButton("Start")
            btn.setMinimumSize(70,30)
            btn.setStyleSheet('QPushButton {background-color: #A3C1DA; border-radius: 4px}')
            btn.clicked.connect(self.dbscan_funct)
            Hbox3.addStretch()
            Hbox3.addWidget(btn)
            Hbox2.addWidget(lab2)
            Hbox2.addStretch()
            Hbox2.addWidget(self.minpts)
            Hbox4.addWidget(lab3)
            Hbox4.addStretch()
            Hbox4.addWidget(self.eps)
            Hbox.addWidget(lab1)
            Vbox.addItem(Hbox)
            Vbox.addItem(Hbox2)
            Vbox.addItem(Hbox4)
            Vbox.addItem(Hbox3)
            self.dialog.setLayout(Vbox)
            self.dialog.show()
            self.dialog.move(500, 300)

       except:
            print("---------------------------------------")
#################################################################################################################################

##############################################################################################################################

    @pyqtSlot()
    def kmeans_dialog(self):
       try:

            
            self.edit2.setText("")
            #self.stat.setText("")
            self.inn.setText("")
            self.intra.setText("")
            self.inter.setText("")
            self.tt.setText("")
            file_data = []
            df = None

            #self.stat.setText("K-Means ..")
            self.dialog = QDialog(self)
            self.dialog.setMinimumSize(200,150)
            self.dialog.close()
            self.dialog.setWindowTitle("K-Means")
            """
            oImage1 = QImage("adn3(1).jpg")
            sImage1 = oImage1.scaled(QSize(230, 150))
            palette1 = QPalette()
            palette1.setBrush(QPalette.Background, QBrush(sImage1))
            self.dialog.setPalette(palette1)
            """
            Hbox = QHBoxLayout()
            Hbox2 = QHBoxLayout()
            Hbox3 = QHBoxLayout()
            Vbox = QVBoxLayout()
            lab1 = QLabel("Partitioning with K-Means")
            lab2 = QLabel("Number of clusters : ")
            txt = QLineEdit()
            self.liste2 = QComboBox()
            self.liste2.setMinimumSize(30,30)
            for i in range(1,11):
                self.liste2.addItem(str(i))
            txt.setMaximumSize(100,30)
            btn = QPushButton("Start")
            btn.setMinimumSize(70,30)
            btn.setStyleSheet('QPushButton {background-color: #A3C1DA; border-radius: 4px}')
            btn.clicked.connect(self.kmeans_funct)
            Hbox3.addStretch()
            Hbox3.addWidget(btn)
            Hbox2.addWidget(lab2)
            Hbox2.addStretch()
            Hbox2.addWidget(self.liste2)
            Hbox.addWidget(lab1)
            Vbox.addItem(Hbox)
            Vbox.addItem(Hbox2)
            Vbox.addItem(Hbox3)
            self.dialog.setLayout(Vbox)
            self.dialog.show()
            self.dialog.move(500, 300)

       except:
            print("---------------------------------------")
#################################################################################################################################
    @pyqtSlot()
    def kmeans_funct(self):
        try:
            
            n = int(self.liste2.currentText())
            self.dialog.close()
            ch = ""
            #self.stat.setText("K-Means processing ..")
            self.dialog = QDialog(self)
            self.dialog.setMinimumSize(200,150)
            self.dialog.close()
            self.dialog.setWindowTitle("K-Means")
            Vbox = QVBoxLayout()
            Hbox = QHBoxLayout()
            lab = QLabel("Please waiting while K-MEANS finished")
            Hbox.addStretch()
            Hbox.addWidget(lab)
            Hbox.addStretch()
            Vbox.addItem(Hbox)
            self.dialog.setLayout(Vbox)
            self.dialog.show()
            self.dialog.move(500, 300)
            
            tmp = time.clock()
            (clusters, centers, intra, inter, inertie) = MyKMEANS(df ,n)
            tmp = time.clock() - tmp
            silh = silhouette(clusters)

            self.edit2.setText("\t\t K-MEANS's Clusters : \n")
            for i in range(len(clusters)):
                ch += "\n>>>>>>>>>>>>>>>>>>>>>>>  Cluster : " + str(i) + "  <<<<<<<<<<<<<<<<<<<<<<<<\n"
                ch += "-Center->  " + str(centers[i]) + "  <-Center-\n\n"
                for j in range(len(clusters[i])):
                    ch += str(j) + ". " + str(clusters[i][j]) + "\n"
                    ss = "%.4f" % silh[i][j]
                    ch += "         Silhouette : " + str(ss) + "\n"
                    
                ch + "\n"    
            
            self.dialog.close()

            self.edit2.append(ch)
            self.intra.setText(str(intra))
            self.inter.setText(str(inter))
            t = "%.4f" % tmp + " S"
            s = "%.4f" % inertie
            self.inn.setText(str(s))
            self.tt.setText(str(t))
            #self.stat.setText("K-Means finished ..")
        
        except:
            print("-----------------------------")
       
    
#################################################################################################################################
    @pyqtSlot()
    def kmedoids_funct(self):
        try:
            n = int(self.liste1.currentText())
            self.dialog.close()
            ch = ""
            #self.stat.setText("K-Means processing ..")
            self.dialog = QDialog(self)
            self.dialog.setMinimumSize(200,150)
            self.dialog.close()
            self.dialog.setWindowTitle("K-MEDOIDS")
            Vbox = QVBoxLayout()
            Hbox = QHBoxLayout()
            lab = QLabel("Please waiting while K-MEDOIDS finished")
            Hbox.addStretch()
            Hbox.addWidget(lab)
            Hbox.addStretch()
            Vbox.addItem(Hbox)
            self.dialog.setLayout(Vbox)
            self.dialog.show()
            self.dialog.move(500, 300)

            tmp = time.clock()
            (clusters, centers, intra, inter, inertie) = MyKMEDOIDS(df ,n)
            tmp = time.clock() - tmp
            
            silh = silhouette(clusters)
            
            self.edit2.setText("\t\t K-MEDOIDS's Clusters : \n")
            for i in range(len(clusters)):
                ch += "\n>>>>>>>>>>>>>>>>>>>>>>>  Cluster : " + str(i) + "  <<<<<<<<<<<<<<<<<<<<<<<<\n"
                ch += "-Center->  " + str(centers[i]) + "  <-Center-\n\n"
                for j in range(len(clusters[i])):
                    ch += str(j) + ". " + str(clusters[i][j]) + "\n"
                    ss = "%.4f" % silh[i][j]
                    ch += "         Silhouette : " + str(ss) + "\n"
                ch + "\n"    
            
            self.dialog.close()

            self.edit2.append(ch)
            self.intra.setText(str(intra))
            self.inter.setText(str(inter))
            t = "%.4f" % tmp + " S"
            s = "%.4f" % inertie
            self.inn.setText(str(s))
            self.tt.setText(str(t))
            #self.stat.setText("K-Means finished ..")
        except:
            print("-----------------------------")

#################################################################################################################################
    @pyqtSlot()
    def dbscan_funct(self):
        try:
            eps = int(self.eps.currentText())
            minpts = int(self.minpts.currentText())
            self.dialog.close()

            #self.stat.setText("DB-Scan finished ..")

            ch = ""
            #self.stat.setText("DB-SCAN processing ..")
            self.dialog = QDialog(self)
            self.dialog.setMinimumSize(200,150)
            self.dialog.close()
            self.dialog.setWindowTitle("DB-SCAN")
            Vbox = QVBoxLayout()
            Hbox = QHBoxLayout()
            lab = QLabel("Please waiting while DB-SCAN finished")
            Hbox.addStretch()
            Hbox.addWidget(lab)
            Hbox.addStretch()
            Vbox.addItem(Hbox)
            self.dialog.setLayout(Vbox)
            self.dialog.show()
            self.dialog.move(500, 300)
            
            tmp = time.clock()
            (clusters, centers, noise, intra, inter, inertie) = MyDBSCAN(list(df['Sequence']), eps, minpts)
            tmp = time.clock() - tmp
            silh = silhouette(clusters)

            self.edit2.setText("\t\t DB-SCAN's Clusters : \n")
            for i in range(len(clusters)):
                ch += "\n>>>>>>>>>>>>>>>>>>>>>>>  Cluster : " + str(i) + "  <<<<<<<<<<<<<<<<<<<<<<<<\n"
                ch += "-Center->  " + str(centers[i]) + "  <-Center-\n\n"
                for j in range(len(clusters[i])):
                    ch += str(j) + ". " + str(clusters[i][j]) + "\n"
                    ss = "%.4f" % silh[i][j]
                    ch += "         Silhouette : " + str(ss) + "\n"
                ch + "\n"

            ch += "\n -------------------->  Noise  <---------------------\n"
            for i in range(len(noise)):
                ch += str(i) + ". " + str(noise[i]) + "\n"
                   
            
            self.dialog.close()

            self.edit2.append(ch)
            self.intra.setText(str(intra))
            self.inter.setText(str(inter))
            t = "%.4f" % tmp + " S"
            s = "%.4f" % inertie
            self.inn.setText(str(s))
            self.tt.setText(str(t))
        except:
            print('------------------------------------------------')


#################################################################################################################################
    @pyqtSlot()
    def agnes_funct(self):
        global df
        #self.stat.setText("AGNES")
        if(True):
            self.edit2.setText("")
            #self.stat.setText("")
            self.inn.setText("")
            self.intra.setText("")
            self.inter.setText("")
            self.tt.setText("")
            file_data = []


            self.dialog = QDialog(self)
            self.dialog.setMinimumSize(200,150)
            self.dialog.close()
            self.dialog.setWindowTitle("AGNES")
            Vbox = QVBoxLayout()
            Hbox = QHBoxLayout()
            lab = QLabel("Please waiting while AGNES finished")
            Hbox.addStretch()
            Hbox.addWidget(lab)
            Hbox.addStretch()
            Vbox.addItem(Hbox)
            self.dialog.setLayout(Vbox)
            self.dialog.show()
            self.dialog.move(500, 300)

            tmp = time.clock()
            (all_iteration, n) = MyAGNES(df)
            

            self.edit2.setText("\t\t AGNES iterations : \n")
            ch = ""
            for i in range((len(all_iteration))):
                ch += "\n          Iteration  :  " + str(i) + "\n"
                for j in range(len(all_iteration[i])):
                    for k in range(len(all_iteration[i][j])):
                        ch += "--------------  Cluster : " + str(k) + "\n"
                        ch += "" + str(all_iteration[i][j][k])
                self.edit2.append(ch)

            tmp = time.clock() - tmp
            t = "%.4f" % tmp + " S"
            self.tt.setText(str(t))

            df = None

            self.dialog.close()
        #self.stat.setText("AGNES finished ..")
        



##########################################################################################################################
# pour enregistrer toutes les Operations effectuées par l'utilisateur (seulement les operations effectuées)
    @pyqtSlot()
    def saved(self):
        #self.stat.setText("Saving ...")

        try:
            save_dialog = QtWidgets.QFileDialog()
            save_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            file_path = save_dialog.getSaveFileName(self, 'Export PDF', None, 'PDF files (.pdf)')

            if file_path[0]:
                self.file_path = file_path
                file_open = open(self.file_path[0], 'w')
                self.file_name = (self.file_path[0].split('/'))[-1]

                printer = QPrinter(QPrinter.HighResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path[0])


                date = datetime.now()
                ttt = date.strftime('\t University of Science & Technology Houari Boumediene \n\n\n %d / %m / %Y  \t\t\t\t Bioinformatics \n %Hh : %Mm : %Ss \t\t\t       Projet Data Mining (Clustering)  \n \n \n')
                s = "" + ttt
                for i in range(len(file_data)):
                    s = s + str(file_data[i]) + "\n"

                s += "\n\n\n" + str(df) + "\n\n\n" + self.edit2.toPlainText()

                self.edit.setText(s)
                self.edit.document().print_(printer)
                self.edit.setText("")
                self.edit2.setText("")
                self.tt.setText("")
                self.inn.setText("")
                self.intra.setText("")
                self.inter.setText("")
                

        except FileNotFoundError as why:
            self.error_box(why)
            pass
##########################################################################################################################



#-------------------------- Variable global ---------------------------

global df
global s, ss
global data, file_data
global btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, btnG
btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, btnG = False, False, False, False, False, False, False, False, False
file_data = []
data = {'data': 'data'}
#----------------------------------------------------------------------



if __name__ == '__main__':
    app = QApplication(sys.argv)

# ------------- Splash screen -----------------------
    splash_pix = QPixmap('adn2.jpg')
    splash = QSplashScreen(splash_pix)
    splash.setWindowFlags(Qt.FramelessWindowHint)
    splash.setMaximumSize(600, 450)
    splash.setGeometry(300, 150, 600, 450)

    progressBar = QProgressBar(splash)
    progressBar.setMaximum(10)
    progressBar.setMaximumSize(1000, 15)
    progressBar.setMinimumSize(500, 10)
    progressBar.setGeometry(50, 200, 50, 200)
    progressBar.setFormat("Bioinformatics")

    splash.show()
    splash.showMessage("<h1><font color='black' > <br> <br> <br> Welcome ! </font></h1>", Qt.AlignCenter, Qt.black)

    for i in range(1, 11):
        progressBar.setValue(i)
        t = time.time()
        while time.time() < t + 0.1:
            app.processEvents()

    win = window()
    time.sleep(3)
    splash.finish(win)
    splash.close()
    splash.destroy()
# -------------------------------------------------------------

    win.self.show()
    sys.exit(app.exec_())
