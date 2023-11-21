import math
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from sklearn.cluster import KMeans


def euclidean_distance(x , y):
    
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))     
    #zip use to match x and y into tuple then bring the couple to a and b
    
    distance = round(distance, 2)     #set decimal to 2                                  
    
    return distance

def minkowski_distance(x, y, r):

    distance = sum([abs(a - b) ** r for a, b in zip(x, y)]) ** (1/r)
    distance = round(distance, 2)     #set decimal to 2

    return distance

def similarity_coefficient(list1, list2, n):
    
    intersection = len(set(list1).intersection(set(list2)))     #convert list to set then find num of intersect value
    coefficient = intersection / n                              #divide by all variable of the list
    return coefficient

def cosine_similarity(vector1, vector2):
    
    dot_product = sum(i * j for i, j in zip(vector1, vector2))  #sum of x * y from vector 1 and 2
    magnitude1 = math.sqrt(sum(i ** 2 for i in vector1))        #find ||X|| = sqrt(x1^2 + x2^2 + ...xn^2)
    magnitude2 = math.sqrt(sum(j ** 2 for j in vector2))        #find ||Y||
    similarity = dot_product / (magnitude1 * magnitude2)        #cos(x,y) = sum x*y / ||x|| * ||Y||
    return similarity

def pearson_correlation(list1, list2):
    
    n = len(list1)      #find num of variable

    #find x bar
    sum_x = sum(list1)
    xbar = (1/n)*sum_x
    xbar = round(xbar, 2)
    print(f'x bar : {xbar}')

    #find y bar
    sum_y = sum(list2)
    ybar = (1/n)*sum_y
    ybar = round(ybar, 2)
    print(f'y bar : {ybar}')

    #find x standard deviation(Sx)
    sdx_material = sum((x-xbar) ** 2 for x in list1)
    sd_x = math.sqrt((1/(n-1)) * sdx_material)
    sd_x = round(sd_x, 2)
    print(f'sdx : {sd_x}')

    #find y standard deviation(Sy)
    sdy_material = sum((y-ybar) ** 2 for y in list2)
    sd_y = math.sqrt((1/(n-1)) * sdy_material)
    sd_y = round(sd_y, 2)
    print(f'sdy : {sd_y}')

    #find covariance(Sxy)
    sxy_material = sum(((x-xbar) * (y-ybar)) for x, y in zip(list1, list2))
    covariance = (1/(n-1)) * sxy_material
    covariance = round(covariance, 2)
    print(f'covariance : {covariance}')

    #find cor(x,y)
    correlation = covariance/(sd_x * sd_y)
    correlation = round(correlation, 2)
    return correlation


'''-----------------Process is here---------------------'''
menu = "nn"

while(menu != "off"):
    
    print('-------Menu Name------')
    print('Euclidean Distance (ED)')
    print('Minkowski Distance (MD)')
    print('Similarity Coefficient (SMC)')
    print('Jaccard Similarity (JC)')
    print('Cosine Similarity (CS)')
    print('Pearson Correlation (PC)')
    print('K Nearest Neighbor (KNN)')
    print('K Mean (KM)')
    print('Decision Tree (DT)')
    print('Apriori (AP)')
    print('Exit --> (off)')

    menu = input('Enter Menu Code (xxx) : ')

    #Euclidean Distance menu
    if(menu == "ED"):

        ''' Test x = [1, 2, 3] y = [4, 5, 6] '''

        #input 
        point_x = list(map(int, input('(x) : ').strip().split()))
        point_y = list(map(int, input('(y): ').strip().split()))

        #sent array to function
        distance = euclidean_distance(point_x,point_y)

        #output
        print(f'Euclidean Distance : {distance}')
    
    elif(menu == "MD"):

        ''' Test x = [1, 2, 3] y = [4, 5, 6] r = 3 '''

        #input 
        point_x = list(map(int, input('(x) : ').strip().split()))
        point_y = list(map(int, input('(y): ').strip().split()))
        r = int(input('Enter r variable : '))

        #sent variable to function
        distance = minkowski_distance(point_x, point_y, r)

        #output
        print(f'Minkowski Distance : {distance}')

    elif(menu == "SMC"):

        ''' Test x = [0, 0, 1, 1] y = [0, 1, 0, 1] '''
        
        #input
        x = list(map(int, input('(x) : ').strip().split()))
        y = list(map(int, input('(y) : ').strip().split()))
        n = len(x)

        #sent variable to function
        coefficient = similarity_coefficient(x, y, n)

        #output
        print(f'Similar Coefficienct : {coefficient}')

    elif(menu == "JC"):

        ''' Test A = [1, 0, 0, 1, 1, 1] B = [0, 0, 1, 1, 1, 0] '''

        #input
        a = list(map(int, input('(a) : ').strip().split()))
        b = list(map(int, input('(b) : ').strip().split()))
        
        #library jaccard_score use function --> num of 1 intersect 1 / everything that isn't 0 intersect 0
        similarity = jaccard_score(a, b)

        #output
        print(f'Jaccard similarity : {similarity}')

    elif(menu == "CS"):

        ''' Test v1 = [1, 2, 4] v2 = [2, 4, 8] '''

        #input
        vector1 = list(map(int, input('Vector(1) : ').strip().split()))
        vector2 = list(map(int, input('Vector(2) : ').strip().split()))

        #sent variable to function
        similarity = cosine_similarity(vector1, vector2)

        #output
        print(f'Cosine Similarity : {similarity}')

    elif(menu == "PC"):

        ''' Test x = [1, 2, 4] y = [2, 4, 8] '''

        #input
        x = list(map(int, input('(x) : ').strip().split()))
        y = list(map(int, input('(y) : ').strip().split()))

        #sent variable to function
        correlation = pearson_correlation(x, y)

        #output
        print(f'Pearson Correlation : {correlation}')

    elif(menu == "KNN"):

        ''' Test x = [0, 0, 1, 4, 5, 5] y = [0, 1, 3, 5, 6, 8] c1 = [0, 0] c2 = [5, 8] '''

        #input
        x = list(map(int, input('Array of Point(x): ').strip().split()))
        y = list(map(int, input('Array of Point(y): ').strip().split()))
        c1 = list(map(int, input('C1(x,y) : ').strip().split()))
        c2 = list(map(int, input('C2(x,y) : ').strip().split()))

        #print input
        print(f'x = {x}')
        print(f'y = {y}')
        print(f'C1(x,y) = {c1}')
        print(f'C2(x,y) = {c2}')

        #distance (x,c1) and (x,c2)
        dx_c1 = []
        dx_c2 = []

        #first Cluster
        prev_cluster = []

        #variable to cal new centroid
        c1x_sum = 0
        c1y_sum = 0
        c2x_sum = 0
        c2y_sum = 0
        count1 = 0
        count2 = 0

        #variable for run knn utill prev_cluster equal to present_cluster
        status = True

        #cal prev_cluster
        for i in range (0,len(x)):

            #cal D(x,c1)    
            distance_c1 = math.sqrt(((x[i]-c1[0]) **2) + ((y[i]-c1[1]) **2))
            distance_c1 = round(distance_c1, 2)
            dx_c1.append(distance_c1)

            #cal D(x,c2)
            distance_c2 = math.sqrt(((x[i]-c2[0]) **2) + ((y[i]-c2[1]) **2))
            distance_c2 = round(distance_c2, 2)
            dx_c2.append(distance_c2)

            #add prev_cluster and cal material for cal new centroid
            if(distance_c1 < distance_c2):
                prev_cluster.append(1)
                c1x_sum = c1x_sum + x[i] #C1(x1 + x2 + ... + xn)
                c1y_sum = c1y_sum + y[i] #C1(y1 + y2 + ... + yn)
                count1 = count1 + 1      #num of cluster member for cal avg for new centroid
            else:
                prev_cluster.append(2)
                c2x_sum = c2x_sum + x[i] 
                c2y_sum = c2y_sum + y[i]
                count2 = count2 + 1
            
        print(f'D(x,c1) = {dx_c1}')
        print(f'D(x,c2) = {dx_c2}')
        print(f'Cluster = {prev_cluster}')
        print('\n-----------------------------------------------------\n')

        present_cluster = []

        #cal another new cluster
        while(status == True):

            #new centroid c1(x,y)
            new_c1_x = c1x_sum / count1
            new_c1_y = c1y_sum / count1
            c1[0] = new_c1_x
            c1[1] = new_c1_y

            #new centroid c2(x,y)
            new_c2_x = c2x_sum / count2
            new_c2_y = c2y_sum / count2
            c2[0] = new_c2_x
            c2[1] = new_c2_y

            for i in range (0,len(x)):
                
                distance_c1 = math.sqrt(((x[i]-c1[0]) **2) + ((y[i]-c1[1]) **2))
                distance_c1 = round(distance_c1, 2)
                dx_c1.append(distance_c1)

                distance_c2 = math.sqrt(((x[i]-c2[0]) **2) + ((y[i]-c2[1]) **2))
                distance_c2 = round(distance_c2, 2)
                dx_c2.append(distance_c2)

                if(distance_c1 < distance_c2):
                    present_cluster.append(1)
                    c1x_sum = c1x_sum + x[i] 
                    c1y_sum = c1y_sum + y[i]
                    count1 = count1 + 1
                else:
                    present_cluster.append(2)
                    c2x_sum = c2x_sum + x[i] 
                    c2y_sum = c2y_sum + y[i]
                    count2 = count2 + 1
            
            print(f'D(x,c1) = {dx_c1}')
            print(f'D(x,c2) = {dx_c2}')
            print(f'Cluster = {prev_cluster}')
            print('\n-----------------------------------------------------\n')

            #check cluster prev and present (true -> break loop) ,(false -> continue loop)
            if(prev_cluster == present_cluster):
                status = False
            else:
                status = True

    elif(menu == "KM"):

        ''' [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]] '''

        #input
        n = int(input('number of Cluster : '))
        nr = int(input('number of Record : '))
        dataset = []
        for i in range (0,nr):
            record = list((input('Enter point value(x,y) : ').strip().split()))
            dataset.append(record)

        print(dataset)

        # Initialize the KMeans model with n clusters
        kmeans = KMeans(n_clusters=n)

        # Fit the model to the data
        kmeans.fit(dataset)

        # Get the cluster labels for each data point
        clusters = kmeans.predict(dataset)

        # Get the coordinates of the cluster centroid
        centers = kmeans.cluster_centers_

        #Output --> the cluster centroid and 
        print("Cluster labels: ", clusters)
        print("Cluster centers: ", centers)


    elif(menu == "DT"):

        ''' attribute --> grade(0-4) , study day(0-7)
            x = [[4, 7], [2, 6], [1, 3], [2, 3]]

            result --> (not pass 0, pass 1)
            y = [1, 1, 0, 0]

            test x [3, 1] --> grade 3 study 1 day
        '''

        #input
        nr = int(input('number of Record : '))
        x = []
        for i in range (0,nr):
            record = list((input('Enter record value : ').strip().split()))
            x.append(record)

        y = list((input('Enter Result value : ').strip().split()))


        # Initialize the DecisionTreeClassifier with default parameters
        clf = DecisionTreeClassifier()

        # Fit the model to the data
        clf.fit(x, y)

        # Predict the class of a new datapoint
        X_data = list((input('Enter x value : ').strip().split()))
        new_X = [X_data]
        prediction = clf.predict(new_X)

        # Print the predicted class
        print(prediction)

        
    
    elif(menu == "AP"):

        ''' ehehe --> pip install mlxtend   '''
        
        #input
        nr = int(input('number of Record : '))
        dataset = []
        for i in range (0,nr):
            record = list((input('Enter record value : ').strip().split()))
            dataset.append(record)

        print(dataset)

        #use library to convert list to dataframe with true/false value
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        print(df)

        #input minsup to do apriori
        minsup = float(input('Min Support : '))
        apriori(df, min_support=minsup)
        apriori(df, min_support=minsup, use_colnames=True)

        #set length for itemset for filtering result
        frequent_itemsets = apriori(df, min_support=minsup, use_colnames=True)
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

        #output --> all itemset
        print(frequent_itemsets)

        #Filtering --> input length of itemset and support as you wish
        leng = int(input('Itemset Length : '))
        sup = float(input('Support Value : '))

        #output
        print(frequent_itemsets[ (frequent_itemsets['length'] == leng) & (frequent_itemsets['support'] >= sup) ])


    

    print('\n----------------------------------------------------\n')

