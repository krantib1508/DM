import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import CSVFile
from django.http import HttpResponse,JsonResponse
from rest_framework.parsers import FileUploadParser,MultiPartParser,FormParser
from .models import CSVFile
from .serializers import CSVFileSerializer
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from django.http import FileResponse
import uuid
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score



class CSVFileUploadView(APIView):
    parser_classes = ( MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = CSVFileSerializer(data=request.data)

        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)




def home(request):
    return HttpResponse("Hello homepage")


def calculate_mode(column):
    counts = {}
    for value in column:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    max_count = max(counts.values())
    mode_values = [key for key, value in counts.items() if value == max_count]
    return mode_values[0]


def assignment1(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    


    data = pd.read_csv(node[0].file)
    data=data.drop('variety',axis=1)
    mean_data = data.sum() / len(data)
    median_data = data.apply(sorted, axis=0).iloc[len(data) // 2]
    mode_data = data.apply(calculate_mode)
    squared_diff = (data - mean_data) ** 2
    var_data = squared_diff.sum() / (len(data) - 1)
    sd_data = var_data ** 0.5


    print("mean",mean_data[0])
    print("median",median_data[0])
    print("mode",mode_data[0])
    print("variance",var_data[0])
    print("standard deviation",sd_data[0])
    

    my_data={
        "name":node[0].name,
        "mean":mean_data.to_dict(),
        "median":median_data.to_dict(),
        "mode":mode_data.to_dict(),
        "variance":var_data.to_dict(),
        "std":sd_data.to_dict()
    }
    return JsonResponse(my_data)

from io import StringIO

def assignment1_que2(request):
        node=CSVFile.objects.all()
        print(node[0].name)
        if len(node)==0 :
            return HttpResponse("No csv file in database !!")
        

        csv_file=node[0].file

        print("boundary 1")

        df = pd.read_csv(csv_file)

        df = df.drop('variety', axis=1)
  
        # Calculate various dispersion measures
        data = df.values.flatten()
        data = np.sort(data)

        # Range
        data_range = np.ptp(data)

        # Quartiles
        quartiles = np.percentile(data, [25, 50, 75])

        # Interquartile Range (IQR)
        iqr = quartiles[2] - quartiles[0]

        # Five-Number Summary
        five_number_summary = {
            "Minimum": np.min(data),
            "Q1 (25th Percentile)": quartiles[0],
            "Median (50th Percentile)": quartiles[1],
            "Q3 (75th Percentile)": quartiles[2],
            "Maximum": np.max(data)
        }

        csv_file.seek(0)  # Ensure the file pointer is at the beginning
        csv_data = csv_file.read().decode('utf-8')
        csv_buffer = StringIO(csv_data)

        column_name="sepal.length"
        column_name2="sepal.width"

        column_values=[]
        column_values2=[]

        csv_reader = csv.DictReader(csv_buffer)
        for row in csv_reader:
            if column_name in row:
                   column_values.append(row[column_name])

        csv_file.seek(0)  # Ensure the file pointer is at the beginning
        csv_data = csv_file.read().decode('utf-8')
        csv_buffer = StringIO(csv_data)
        csv_reader = csv.DictReader(csv_buffer)

        for row in csv_reader:
            if column_name2 in row:
                   column_values2.append(row[column_name2])
    
        

        result = {
            "Range": data_range,
            "Quartiles": quartiles.tolist(),
            "Interquartile": iqr,
            "Five": five_number_summary,
            "values":column_values,
            "values2":column_values2
        }

        return JsonResponse(result)



def assignment2(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    Attr1="sepal.length"
    Attr2="sepal.width"

    print("boundary 1")

    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    contingency_table = pd.crosstab(df[Attr1], df[Attr2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    alpha=0.7  # we have set that.....
    fl=0

    if p <= alpha:
        print(f"The p-value ({p}) is less than or equal to the significance level ({alpha}).")
        print("The selected attributes are correlated.")
        fl=1
    else:
        print(f"The p-value ({p}) is greater than the significance level ({alpha}).")
        print("The selected attributes are not correlated.")
        fl=0
    
    print(expected)
    correlation_coefficient = df[Attr1].corr(df[Attr2])
    covariance = df[Attr1].cov(df[Attr2])

    min_value = df[Attr1].min()
    max_value = df[Attr1].max()

    df[Attr2] = (df[Attr1] - min_value) / (max_value - min_value)

    mean = df[Attr1].mean()
    std_dev = df[Attr1].std()
    df[Attr2] = (df[Attr1] - mean) / std_dev


    mean = df[Attr1].mean()
    std_dev = df[Attr1].std()
    df[Attr2] = (df[Attr1] - mean) / std_dev

    max_abs = df[Attr1].abs().max()
    df[Attr2] = df[Attr1] / (10 ** len(str(int(max_abs))))



    if fl:
       my_data={
        "name":node[0].name,
        "result":"correlated",
        "p":p,
        "chi2":chi2,
        "dof":dof,
        "a1":Attr1,
        "a2":Attr2
       }

       return JsonResponse(my_data)
    
    my_data={
        "name":node[0].name,
        "result":"not correlated",
        "p":p,
        "chi":chi2,
        "dof":dof,
        "a1":Attr1,
        "a2":Attr2
       }

    return JsonResponse(my_data)



def assignment3(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    strr="info"
    
    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

   
    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']

    if strr=="info":
       clf = DecisionTreeClassifier(criterion='entropy')
    elif strr=="gini":
       clf= DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=42)
    elif strr=="gain":
       clf = DecisionTreeClassifier(criterion='entropy', splitter='best')




    clf.fit(X, Y)
    decision_tree_text = export_text(clf, feature_names=X.columns.tolist())
    tree.plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, clf.classes_)))
    tree_image_path = 'static/plot/image.png'
    os.makedirs(os.path.dirname(tree_image_path), exist_ok=True)
    plt.savefig(tree_image_path)
    my_data={"name":file_name,"text":decision_tree_text}
    return JsonResponse(my_data)
               




def assignment3_confuse_matrix(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    misclassification_rate = 1 - accuracy
    sensitivity = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    response_data = {
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'misclassification_rate': misclassification_rate,
        'sensitivity': sensitivity,
        'precision': precision,
    }

    return JsonResponse(response_data, status=status.HTTP_200_OK)


def assignment4(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(criterion='entropy', splitter='best')
    clf.fit(X_train, y_train)


    rules = export_text(clf, feature_names=list(X.columns.tolist()))

    y_pred = clf.predict(X)
    accuracy = accuracy_score(Y, y_pred)

    coverage = len(y_pred) / len(Y) * 100

    rule_count = len(rules.split('\n'))

    my_data = {
        "name":file_name,
        'rules': rules,
        'accuracy': accuracy,
        'coverage': coverage,
        'toughness': rule_count,
    }
    return JsonResponse(my_data)


def assignment5(request):
    node=CSVFile.objects.all()
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")

    print("boundary 1")
    data = pd.read_csv(node[0].file)
    df = pd.DataFrame(data)
    file_name=node[0].name
    X = df.drop('variety', axis=1)
    Y = df['variety']
    my_data={
        "name":file_name
    }
    return JsonResponse(my_data)



###################################################################################################

# views.py

# Chi-Square Value: 1922.9347363945576

# P-Value: 2.6830523867648017e-17

from scipy.stats import chi2_contingency,zscore,pearsonr
import tempfile
from django.shortcuts import render
import json
# Create your views here.
from rest_framework.parsers import FileUploadParser
import csv
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.views import View
import csv
import math
from django.http import JsonResponse
from django.views import View
import csv
from django.http import HttpResponse
import json
from django.http import JsonResponse
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import csv
import statistics
import numpy as np
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views import View
import statistics
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency,zscore,pearsonr
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import datasets
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.tree import export_text
from django.views.decorators.csrf import csrf_exempt
import logging
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import tempfile
import shutil
import math
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse, FileResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

import pandas as pd
import os

import numpy as np
from django.http import JsonResponse
from scipy.stats import chi2_contingency
from scipy.stats import chi2

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import graphviz
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle


class RegressionClass(APIView):
    @method_decorator(csrf_exempt)
    def get(self, request, *args, **kwargs):
        if request.method == 'GET':
            try:
                
                node=CSVFile.objects.all()
                print(node[0].name)

                if len(node)==0 :
                    return HttpResponse("No csv file in database !!")

                print("boundary 1")

                algo="KNN"

                data = pd.read_csv(node[0].file)

                df = pd.DataFrame(data)
                df = shuffle(df, random_state=42)

                target_class = df.columns[-1]

                object_cols = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype=='object' and col != target_class]
                numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != target_class]

                X = df[numeric_cols+object_cols]
                y = df[target_class]

                # print(X.head())
                # print(y.head())
                

                X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
                ordinal_encoder = OrdinalEncoder()

                if target_class in object_cols :
                    object_cols = [col for col in object_cols if col != target_class]
                    y_train[target_class] = OrdinalEncoder.fit_transform(y_train[target_class])
                    y_test[target_class] = OrdinalEncoder.fit_transform(y_test[target_class])

                X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
                X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])


                if algo == "Linear" : 
                    cm = self.logistic_regression(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "Naive" : 
                    cm = self.naive_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "KNN" : 
                    cm = self.knn_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                elif algo == "ANN" : 
                    cm = self.ann_classifier(X, y, X_train, X_test, y_train, y_test)
                    return JsonResponse({"confusion_matrix": cm})
                
                return JsonResponse({"accuracy": "accuracy"})
            except Exception as e :
                print(e)
                return JsonResponse({"error => ": str(e)}, status=status.HTTP_200_OK, safe=False)

    def preprocess(self, df):
        numerical_columns = df.select_dtypes(include=[int, float])

        # Select columns with a number of unique values less than 4
        unique_threshold = 4
        selected_columns = []
        for column in df.columns:
            if len(df[column].unique()) < unique_threshold and df[column].dtype == 'object':
                selected_columns.append(column)

        # Combine the two sets of selected columns (numerical and unique value threshold)
        final_selected_columns = list(set(numerical_columns.columns).union(selected_columns))

        # Create a new DataFrame with only the selected columns
        filtered_df = df[final_selected_columns]

        from sklearn.preprocessing import LabelEncoder

        # Assuming 'filtered_df' is your DataFrame with object-type columns to be encoded
        encoder = LabelEncoder()

        for column in filtered_df.columns:
            if filtered_df[column].dtype == 'object':
                filtered_df[column] = encoder.fit_transform(filtered_df[column])
        
        return filtered_df


    def logistic_regression(self, X, y, X_train, X_test, y_train, y_test):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        # Load your dataset (e.g., Iris or Breast Cancer)
        # X, y, X_train, X_test, y_train, y_test = load_data()
        # Split the data into training and testing sets
        
        # Create and train the regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)

        print(y_pred.shape)
        print(y_test.shape)
        cm = confusion_matrix(y_test, y_pred).tolist()
        print(cm)

        
        return cm

    def naive_classifier(self, X, y, X_train, X_test, y_train, y_test):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score

        

        # Create and train the Naïve Bayes classifier
        nb_classifier = GaussianNB()
        nb_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = nb_classifier.predict(X_test)

        # Calculate accuracy
        cm = confusion_matrix(y_test, y_pred).tolist()

        return cm

    def knn_classifier(self, X, y, X_train, X_test, y_train, y_test):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        

        # Create and train the k-NN classifier with different values of k
        k_values = [1, 3, 5, 7]
        cm = []
        accuracy_scores = []


        for k in k_values:
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            knn_classifier.fit(X_train, y_train)
            
            # Make predictions
            y_pred = knn_classifier.predict(X_test)
            
            # Calculate accuracy
            metrix = confusion_matrix(y_test, y_pred).tolist()
            cm.append({'confusion_matrix': metrix})


            print(metrix)

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # Plot the error graph
        plt.figure(figsize=(8, 6))  # Adjust figure size as needed
        plt.plot(k_values, accuracy_scores)
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        # plt.show()

        plt.savefig("E:\sem7\DM lab\DM LA1\frontend\app\src\Components\static\KNNplot.png")
    
        return cm

    def ann_classifier(self, X, y, X_train, X_test, y_train, y_test):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import load_iris, load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier

        

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create and train the ANN classifier
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)

        # Plot the error graph (iteration vs error)
        plt.figure(figsize=(8, 6))
        plt.plot(mlp.loss_curve_)
        plt.title('Error Graph (Iteration vs Error)')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        # plt.show()

        plt.savefig("E:\sem7\DM lab\DM LA1\frontend\app\src\Components\static\ANNplot.png")

        # Evaluate the classifier
        y_pred = mlp.predict(X_test)

        cm = confusion_matrix(y_test, y_pred).tolist()

        return cm
    



    #######LA2

    
import json
import numpy as np
from django.http import JsonResponse
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

@csrf_exempt
def ann_classifier(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')
        
        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a three-layer ANN classifier
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Predict and calculate metrics
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        if len(np.unique(y_test)) == 2:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
        else:
            sensitivity = specificity = None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        response_data = {
            "confusionMatrix": cm.tolist(),
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "recall": recall,
        }
        
        return JsonResponse(response_data)

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn import datasets

def generate_dendrogram(dataset_name, method):
    if dataset_name == 'IRIS':
        data = datasets.load_iris().data
        title = "IRIS Dendrogram"
    elif dataset_name == 'BreastCancer':
        data = datasets.load_breast_cancer().data
        title = "Breast Cancer Dendrogram"
    else:
        return None  # Invalid dataset choice

    if method == 'agnes':
        linkage_method = 'ward'  # Use 'ward' method for AGNES
    elif method == 'diana':
        linkage_method = 'single'  # Use 'single' method for DIANA
    else:
        return None  # Invalid clustering method

    linkage_matrix = linkage(data, method=linkage_method)

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Distance")

    # Save the dendrogram to a file and close the plot
    img_path = os.path.join('client/src', 'dendrogram.png')
            # plt.savefig(img_path)
    plt.savefig(img_path)

    # Print the dendrogram
    plt.show()

    # Read the saved image and return it
    with open('dendrogram.png', 'rb') as image_file:
        dendrogram_image = image_file.read()

    return dendrogram_image



from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


@csrf_exempt
@require_POST
def dendrogram_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')
        clustering_method = data.get('clustering_method')
        
        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})
        
    
        
    if dataset_name and clustering_method:
        dendrogram_image = generate_dendrogram(dataset_name, clustering_method)

        if dendrogram_image:
            response = HttpResponse(dendrogram_image, content_type='image/png')
            response['Content-Disposition'] = 'attachment; filename="dendrogram.png"'
            return response
        else:
            return JsonResponse({'error': 'Invalid dataset or clustering method'}, status=400)
    else:
        return JsonResponse({'error': 'Missing dataset_name or clustering_method in POST data'}, status=400)




# views.py in your Django app

from django.http import JsonResponse
import networkx as nx
import pandas as pd

# Function to calculate PageRank and return the top 10 pages with their ranks
def calculate_pagerank(file_path):
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)
    pagerank = nx.pagerank(G)
    top_pages = sorted(pagerank, key=pagerank.get, reverse=True)[:10]
    results = pd.DataFrame(columns=['Page', 'PageRank'])
    results['Page'] = top_pages
    results['PageRank'] = [pagerank[page] for page in top_pages]
    return results.to_dict(orient='records')

# Django view function
def get_page_rank(request):
    # Define the file path (replace this with your file path)
    file_path = 'C:/Users/91902/OneDrive/Desktop/web-Stanford.txt'

    # Calculate PageRank and get the top 10 pages with their ranks
    top_pages_with_ranks = calculate_pagerank(file_path)

    # Return the top pages with their ranks in JSON format
    return JsonResponse({'top_pages': top_pages_with_ranks})


# views.py in your Django app

from django.http import JsonResponse
import networkx as nx
import pandas as pd

# Function to calculate HITS scores and return the top 10 authoritative and hub pages
def calculate_hits(file_path):
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)
    hits_scores = nx.hits(G)
    authority_scores = hits_scores[1]
    hub_scores = hits_scores[0]
    top_authority_pages = sorted(authority_scores, key=authority_scores.get, reverse=True)[:10]
    top_hub_pages = sorted(hub_scores, key=hub_scores.get, reverse=True)[:10]
    authority_results = pd.DataFrame(columns=['Page', 'AuthorityScore'])
    authority_results['Page'] = top_authority_pages
    authority_results['AuthorityScore'] = [authority_scores[page] for page in top_authority_pages]
    hub_results = pd.DataFrame(columns=['Page', 'HubScore'])
    hub_results['Page'] = top_hub_pages
    hub_results['HubScore'] = [hub_scores[page] for page in top_hub_pages]
    return authority_results.to_dict(orient='records'), hub_results.to_dict(orient='records')

# Django view function
def get_hits_scores(request):
    file_path = 'C:/Users/91902/OneDrive/Desktop/web-Stanford.txt'
    authority_pages, hub_pages = calculate_hits(file_path)
    return JsonResponse({'top_authority_pages': authority_pages, 'top_hub_pages': hub_pages})

import requests
from bs4 import BeautifulSoup
from collections import deque

def dfs_crawler(seed_url, max_pages=10):
    visited = set()
    stack = [(seed_url, 0)]
    result = []

    while stack and len(visited) < max_pages:
        url, depth = stack.pop()

        if url in visited:
            continue

        visited.add(url)
        result.append(url)

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            for link in soup.find_all('a', href=True):
                next_url = link['href']
                if next_url.startswith('http') and next_url not in visited:
                    stack.append((next_url, depth + 1))
        except Exception as e:
            print(f"Error: {e}")
    
    return result


def bfs_crawler(seed_url, max_pages=10):
    visited = set()
    queue = deque([(seed_url, 0)])
    result = []
    

    while queue and len(visited) < max_pages:
        url, depth = queue.popleft()
        
        if url in visited:
            continue
        
        visited.add(url)
        result.append(url)
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                next_url = link['href']
                if next_url.startswith('http') and next_url not in visited:
                    queue.append((next_url, depth + 1))
        except Exception as e:
            print(f"Error: {e}")
    
    return result


from django.http import JsonResponse

@csrf_exempt
def crawl_urls(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        url = data.get('url')  # Get the URL from the form input
        # Call BFS and DFS crawlers
        bfs_result = bfs_crawler(url)
        dfs_result = dfs_crawler(url)

        # Prepare the response data
        response_data = {
            'bfs_result': bfs_result,
            'dfs_result': dfs_result
        }

        return JsonResponse(response_data)


#assignment 6
# ==============kmeans==================
from io import BytesIO  # Add this import statement

import os

def generate_kmeans(dataset_name, k_value):
    if dataset_name == 'IRIS':
        data = datasets.load_iris()
        title = "IRIS K-Means Clustering"
    elif dataset_name == 'BreastCancer':
        data = datasets.load_breast_cancer()
        title = "Breast Cancer K-Means Clustering"
    else:
        return None  # Invalid dataset choice

    if k_value:
        kmeans = KMeans(n_clusters=k_value, random_state=42)
        kmeans.fit(data.data)
        y_pred = kmeans.predict(data.data)
        plt.figure(figsize=(10, 6))
        plt.scatter(data.data[:, 0], data.data[:, 1], c=y_pred)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        # Specify the directory where you want to save the image
        save_directory = 'client/src'

        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Construct the full path for the saved image with k_value in the filename
        img_filename = f'kmeansOf_{dataset_name}.png'
        img_path = os.path.join(save_directory, img_filename)

        # Save the image to the specified directory
        plt.savefig(img_path)
        plt.close()

        # Read the saved image and return it
        with open(img_path, 'rb') as image_file:
            kmeans_image = image_file.read()

        return kmeans_image
    else:
        return None  # Invalid k-value


@csrf_exempt
@require_POST
def kmeans_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print("Received data:", data)  # Print the received data for debugging
        dataset_name = data.get('dataset')
        k_value_str = data.get('k_value')
        
        print("dataset_name:", dataset_name)
        print("k_value_str:", k_value_str)
        if not k_value_str:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            k_value = int(k_value_str)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be an integer.'}, status=400)

        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})
    
        if dataset_name and k_value:
            kmeans_image = generate_kmeans(dataset_name, k_value)

            if kmeans_image:
                response = HttpResponse(kmeans_image, content_type='image/png')
                response['Content-Disposition'] = 'attachment; filename="kmeans.png"'
                return response
            else:
                return JsonResponse({'error': 'Invalid dataset or k-value'}, status=400)
        else:
            return JsonResponse({'error': 'Missing dataset_name or k_value in POST data'}, status=400)

# ================k-medoids================
import numpy as np
import matplotlib.pyplot as plt
import os
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from sklearn_extra.cluster import KMedoids
import numpy as np
import matplotlib.pyplot as plt
import os

# Sample data (Replace this with your actual dataset)
iris_data = np.random.rand(150, 4)  # Example for Iris dataset
breast_cancer_data = np.random.rand(569, 30)  # Example for Breast Cancer dataset

# Directory to save the generated images
save_directory = 'client/src'

# Ensure the directory exists
os.makedirs(save_directory, exist_ok=True)

# def generate_kmedoids(dataset_name, k_value):
#     if dataset_name == 'IRIS':
#         data = iris_data
#         title = "IRIS k-Medoids Clustering"
#     elif dataset_name == 'BreastCancer':
#         data = breast_cancer_data
#         title = "BREAST CANCER k-Medoids Clustering"
#     else:
#         return None  # Invalid dataset choice

#     if k_value:
#         # initial_medoids = np.random.choice(len(data), k_value, replace=False)
#         kmedoids_instance = KMedoids(n_clusters=k_value, random_state=0, init='k-medoids++')
#         kmedoids_instance.fit(data)
#         labels = kmedoids_instance.labels_
#         medoids = kmedoids_instance.medoid_indices_

#     # Plotting the clusters and medoids
#     plt.figure(figsize=(8, 6))
#     for cluster_index in range(3):
#         cluster_points = data[labels == cluster_index]
#         plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_index + 1}')
#         plt.scatter(data[medoids, 0], data[medoids, 1], c='red', marker='x', s=200, label='Medoids')
#         plt.title(title)
#         plt.xlabel("Feature 1")
#         plt.ylabel("Feature 2")
#         plt.legend()

#         # Construct the full path for the saved image with k_value in the filename
#         img_filename = f'kmedoidsOf_{dataset_name}.png'
#         img_path = os.path.join(save_directory, img_filename)

#         # Save the image to the specified directory
#         plt.savefig(img_path)
#         plt.close()

#         # Read the saved image and return it
#         with open(img_path, 'rb') as image_file:
#             kmedoids_image = image_file.read()

#         return kmedoids_image
#     else:
#         return None  # Invalid k-value


import numpy as np
import os
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer

def generate_kmedoids(dataset_name, k_value):
    if dataset_name == 'IRIS':
        data = load_iris().data[:, :2]  # Considering only the first two features for visualization purposes
        title = "IRIS k-Medoids Clustering"
    elif dataset_name == 'BreastCancer':
        data = load_breast_cancer().data[:, :2]  # Considering only the first two features for visualization purposes
        title = "BREAST CANCER k-Medoids Clustering"
    else:
        return None  # Invalid dataset choice

    if k_value:
        kmedoids_instance = KMedoids(n_clusters=k_value, random_state=0, init='k-medoids++')
        kmedoids_instance.fit(data)
        labels = kmedoids_instance.labels_
        medoids = kmedoids_instance.medoid_indices_

        # Plotting the clusters and medoids
        plt.figure(figsize=(8, 6))
        for cluster_index in range(k_value):
            cluster_points = data[labels == cluster_index]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_index + 1}')
        plt.scatter(data[medoids, 0], data[medoids, 1], c='red', marker='x', s=200, label='Medoids')
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

        # Construct the full path for the saved image with k_value in the filename
        save_directory = 'client/src'
        img_filename = f'kmedoidsOf_{dataset_name}.png'
        img_path = os.path.join(save_directory, img_filename)

        # Save the image to the specified directory
        plt.savefig(img_path)
        plt.close()

        # Read the saved image and return it
        with open(img_path, 'rb') as image_file:
            kmedoids_image = image_file.read()

        return kmedoids_image
    else:
        return None  # Invalid k-value

@csrf_exempt
@require_POST
def kmedoids_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        print("Received data:", data)  # Print the received data for debugging
        dataset_name = data.get('dataset')
        k_value_str = data.get('k_value')

        print("dataset_name:", dataset_name)
        print("k_value_str:", k_value_str)
        if not k_value_str:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            k_value = int(k_value_str)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be an integer.'}, status=400)

        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        if dataset_name and k_value:
            kmedoids_image = generate_kmedoids(dataset_name, k_value)

            if kmedoids_image:
                response = HttpResponse(kmedoids_image, content_type='image/png')
                response['Content-Disposition'] = 'attachment; filename="kmedoids.png"'
                return response
            else:
                return JsonResponse({'error': 'Invalid dataset or k-value'}, status=400)
        else:
            return JsonResponse({'error': 'Missing dataset_name or k_value in POST data'}, status=400)


# ==============
# Import necessary libraries
from django.http import JsonResponse, HttpResponse
from sklearn.cluster import Birch, DBSCAN
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import json
from io import BytesIO

# BIRCH Clustering
def generate_birch(dataset_name):
    if dataset_name == 'IRIS':
        data = load_iris()
        title = "IRIS BIRCH Clustering"
    elif dataset_name == 'BreastCancer':
        data = load_breast_cancer()
        title = "Breast Cancer BIRCH Clustering"
    else:
        return None  # Invalid dataset choice

    birch = Birch(threshold=0.5, branching_factor=50)
    birch.fit(data.data)
    labels = birch.predict(data.data)

    plt.figure(figsize=(10, 6))
    plt.scatter(data.data[:, 0], data.data[:, 1], c=labels)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    save_directory = 'client/src'
    os.makedirs(save_directory, exist_ok=True)
    img_filename = f'birchOf_{dataset_name}.png'
    img_path = os.path.join(save_directory, img_filename)

    plt.savefig(img_path)
    plt.close()

    with open(img_path, 'rb') as image_file:
        birch_image = image_file.read()

    return birch_image

# DBSCAN Clustering
def generate_dbscan(dataset_name):
    if dataset_name == 'IRIS':
        data = load_iris()
        title = "IRIS DBSCAN Clustering"
    elif dataset_name == 'BreastCancer':
        data = load_breast_cancer()
        title = "Breast Cancer DBSCAN Clustering"
    else:
        return None  # Invalid dataset choice

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data.data)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(data_normalized)

    plt.figure(figsize=(10, 6))
    plt.scatter(data.data[:, 0], data.data[:, 1], c=labels)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    save_directory = 'client/src'
    os.makedirs(save_directory, exist_ok=True)
    img_filename = f'dbscanOf_{dataset_name}.png'
    img_path = os.path.join(save_directory, img_filename)

    plt.savefig(img_path)
    plt.close()

    with open(img_path, 'rb') as image_file:
        dbscan_image = image_file.read()

    return dbscan_image

# View for BIRCH clustering
@csrf_exempt
@require_POST
def birch_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')

        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        birch_image = generate_birch(dataset_name)

        if birch_image:
            response = HttpResponse(birch_image, content_type='image/png')
            response['Content-Disposition'] = f'attachment; filename="birchOf_{dataset_name}.png"'
            return response
        else:
            return JsonResponse({'error': 'Invalid dataset for BIRCH clustering'}, status=400)

# View for DBSCAN clustering
@csrf_exempt
@require_POST
def dbscan_view(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        dataset_name = data.get('dataset')

        if dataset_name == 'IRIS':
            data = iris_data
        elif dataset_name == 'BreastCancer':
            data = breast_cancer_data
        else:
            return JsonResponse({'error': 'Invalid dataset name'})

        dbscan_image = generate_dbscan(dataset_name)

        if dbscan_image:
            response = HttpResponse(dbscan_image, content_type='image/png')
            response['Content-Disposition'] = f'attachment; filename="dbscanOf_{dataset_name}.png"'
            return response
        else:
            return JsonResponse({'error': 'Invalid dataset for DBSCAN clustering'}, status=400)


# ========
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.pyplot as plt
import io
import base64
import json
import numpy as np
import pandas as pd

results_df = pd.DataFrame(columns=['Algorithm', 'ARI Score', 'Silhouette Score'])

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris, load_breast_cancer

# Directory to save the generated images
save_directory = 'client/src'
os.makedirs(save_directory, exist_ok=True)

def hierarchical_clustering(dataset_name, algorithm_name):
    if dataset_name == 'iris':
        dataset = datasets.load_iris()
        title = "IRIS "
    elif dataset_name == 'breast_cancer':
        dataset = datasets.load_breast_cancer()
        title = "Breast Cancer "
    else:
        return None  # Invalid dataset choice
    if algorithm_name == 'agnes':
        linkage_matrix = linkage(dataset.data, method='single')  # 'single' linkage method for AGNES
        title += "AGNES Hierarchical Clustering"
    elif algorithm_name == 'diana':
        linkage_matrix = linkage(dataset.data, method='complete')  # 'complete' linkage method for DIANA
        title += "DIANA Hierarchical Clustering"
    else:
        return None  # Invalid algorithm choice

    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    
    # Save the dendrogram plot as an image
    img_filename = f'clustering_result.png'
    img_path = os.path.join(save_directory, img_filename)
    plt.savefig(img_path)
    plt.close()

    return img_path  # Return the path to the generated image


@csrf_exempt
@require_POST
def clustering_view(request):
    global results_df
    data = json.loads(request.body.decode('utf-8'))
    dataset_name = data.get('dataset')
    algorithm_name = data.get('algorithm')
    k_value = data.get('k_value')

    if dataset_name == 'iris':
        dataset = datasets.load_iris()
    elif dataset_name == 'breast_cancer':
        dataset = datasets.load_breast_cancer()
    else:
        return JsonResponse({'error': 'Invalid dataset name'})

    data_to_cluster = dataset.data
    true_labels = dataset.target

    print(dataset_name,algorithm_name)
    if algorithm_name == 'agnes' or algorithm_name == 'diana':
        img_path = hierarchical_clustering(dataset_name,algorithm_name)
        

    # Convert the plot to base64 for sending to the frontend
        image_base64 = plot_to_base64(img_path)
        return JsonResponse({'image': image_base64})

    elif algorithm_name == 'kmeans':
        # k-Means Clustering
        kmeans = KMeans(n_clusters=int(k_value), random_state=42)
        clusters = kmeans.fit_predict(data_to_cluster)

        ari_score, sil_score = evaluate_clusters(data_to_cluster, true_labels, clusters)
        results_df = update_results(results_df, algorithm_name, ari_score, sil_score)
        return JsonResponse({'ARI Score': ari_score, 'Silhouette Score': sil_score})

    elif algorithm_name == 'kmedoids':
        # k-Medoids Clustering (PAM)
        if not k_value:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            k_value = int(k_value)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be an integer.'}, status=400)

        # Use pyclustering's kmedoids for K-Medoids
        kmedoids_instance = kmedoids(data_to_cluster, initial_index_medoids=np.random.randint(0, len(data_to_cluster), int(k_value)))
        clusters = kmedoids_instance.process().get_clusters()

        ari_score, sil_score = evaluate_clusters(data_to_cluster, true_labels, clusters)
        results_df = update_results(results_df, algorithm_name, ari_score, sil_score)
        return JsonResponse({'ARI Score': ari_score, 'Silhouette Score': sil_score})
    
    elif algorithm_name == 'dbscan':
        # DBSCAN Clustering
        if not k_value:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            eps = float(k_value)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be a float.'}, status=400)

        dbscan = DBSCAN(eps=eps)
        clusters = dbscan.fit_predict(data_to_cluster)

        ari_score, sil_score = evaluate_clusters(data_to_cluster, true_labels, clusters)
        results_df = update_results(results_df, algorithm_name, ari_score, sil_score)
        return JsonResponse({'ARI Score': ari_score, 'Silhouette Score': sil_score})



    elif algorithm_name == 'birch':
        # BIRCH Clustering
        if not k_value:
            return JsonResponse({'error': 'k_value is missing or empty'}, status=400)

        try:
            k_value = int(k_value)
        except ValueError:
            return JsonResponse({'error': 'Invalid k_value. Must be an integer.'}, status=400)

        birch = Birch(n_clusters=k_value)
        clusters = birch.fit_predict(data_to_cluster)

        ari_score, sil_score = evaluate_clusters(data_to_cluster, true_labels, clusters)
        results_df = update_results(results_df, algorithm_name, ari_score, sil_score)
        return JsonResponse({'ARI Score': ari_score, 'Silhouette Score': sil_score})

    # Add other clustering algorithms as needed

    return JsonResponse({'error': 'Invalid algorithm name'})

def evaluate_clusters(data_to_cluster, true_labels, predicted_labels):
    unique_labels = np.unique(predicted_labels)

    # Check if there is only one label (e.g., all points assigned to the same cluster)
    if len(unique_labels) < 2:
        return 0.0, 0.0  # Return default scores for this case

    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    sil_score = silhouette_score(data_to_cluster, predicted_labels)
    return ari_score, sil_score


def plot_to_base64(plot_path):
    with open(plot_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

def update_results(results_df, algorithm_name, ari_score, sil_score):
    new_row = pd.DataFrame({
        'Algorithm': [algorithm_name],
        'ARI Score': [ari_score],
        'Silhouette Score': [sil_score]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df

#assign7
# ================ass7===============
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ucimlrepo import fetch_ucirepo
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import json


@csrf_exempt
def run_association_rules(request):
    if request.method == 'POST':
        try:
            print('Received POST request')
            
            # Load dataset
            dataset_id = 105  # Use the appropriate dataset ID
            dataset = fetch_ucirepo(id=dataset_id)
            X = dataset.data.features
            y = dataset.data.targets
            data = pd.concat([X, pd.DataFrame(y, columns=['Class'])], axis=1)

            # Convert categorical columns to boolean using one-hot encoding
            data = pd.get_dummies(data, drop_first=True)

            # Get parameters from the request
            data_json = json.loads(request.body.decode('utf-8'))
            support_values = data_json.get('support_values', [])
            confidence_values = data_json.get('confidence_values', [])
            
            results = []

            for support in support_values:
                for confidence in confidence_values:
                    support = float(support)
                    confidence = float(confidence)

                    # Find frequent itemsets
                    frequent_itemsets = apriori(data, min_support=support, use_colnames=True)
                    frequent_itemsets_list = frequent_itemsets['itemsets'].apply(list).tolist()
                    # print("frequent_itemsets_list:", frequent_itemsets_list)

                    # Generate association rules
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
                    rules_list = rules.to_dict(orient='records')
                    # print("rules_list:", rules.shape[0])

                    # Prepare results
                    

                    # for rule in rules_list:
                    #     lift_val = calculate_lift(data, rule)
                    #     chi_square_val = calculate_chi_square(data, rule)
                    #     all_confidence_val = calculate_all_confidence(data, rule)
                    #     max_confidence_val = calculate_max_confidence(data, rule)
                    #     kulczynski_val = calculate_kulczynski(data, rule)
                    #     cosine_val = calculate_cosine(data, rule)

                    # rule.update({
                    #     'lift': lift_val,
                    #     'chi_square': chi_square_val,
                    #     'all_confidence': all_confidence_val,
                    #     'max_confidence': max_confidence_val,
                    #     'kulczynski': kulczynski_val,
                    #     'cosine': cosine_val
                    # })
                    
                    
                    result = {
                        'support': support,
                        'confidence': confidence,
                        'frequent_itemsets': frequent_itemsets_list,
                        'total_rules': len(rules),
                        'rules': rules_list[0],
                    #     'data': {
                    #     'lift': lift_val,
                    #     'chi_square': chi_square_val,
                    #     'all_confidence': all_confidence_val,
                    #     'max_confidence': max_confidence_val,
                    #     'kulczynski': kulczynski_val,
                    #     'cosine': cosine_val
                    # }
                    }
                    results.append(result)

            if results:
                # Convert frozensets to lists before sending the response
                results_serializable = json.loads(json.dumps(results, default=list))
                return JsonResponse(results_serializable, safe=False)
            else:
                return JsonResponse({'error': 'No results found'})

        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'})


    import numpy as np
from mlxtend.frequent_patterns import apriori


def calculate_lift(rules, data):
    # Calculate Lift measure
    # Lift = P(A and B) / (P(A) * P(B))
    lift_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        support_B = np.sum(data.loc[:, consequent] == 1) / len(data)
        lift = support_A_B / (support_A * support_B)
        lift_values.append(lift)
    return lift_values

def calculate_chi_square(rules, data):
    # Calculate Chi-Square measure
    # Chi-Square = N * (ad - bc)^2 / [(a + b)(c + d)(a + c)(b + d)]
    chi_square_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        a = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1))
        b = np.sum((data.loc[:, antecedent] == 0) & (data.loc[:, consequent] == 1))
        c = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 0))
        d = np.sum((data.loc[:, antecedent] == 0) & (data.loc[:, consequent] == 0))
        chi_square = len(data) * ((a * d - b * c) ** 2) / ((a + b) * (c + d) * (a + c) * (b + d))
        chi_square_values.append(chi_square)
    return chi_square_values

def calculate_all_confidence(rules, data):
    # Calculate All Confidence measure
    # All Confidence = P(A and B) / max(P(A), P(B))
    all_confidence_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        support_B = np.sum(data.loc[:, consequent] == 1) / len(data)
        all_confidence = support_A_B / max(support_A, support_B)
        all_confidence_values.append(all_confidence)
    return all_confidence_values

def calculate_max_confidence(rules, data):
    # Calculate Max Confidence measure
    # Max Confidence = P(A and B) / P(A)
    max_confidence_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        max_confidence = support_A_B / support_A
        max_confidence_values.append(max_confidence)
    return max_confidence_values

def calculate_kulczynski(rules, data):
    # Calculate Kulczynski measure
    # Kulczynski = (P(A and B) / P(A)) + (P(A and B) / P(B)) / 2
    kulczynski_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        support_B = np.sum(data.loc[:, consequent] == 1) / len(data)
        kulczynski = (support_A_B / support_A + support_A_B / support_B) / 2
        kulczynski_values.append(kulczynski)
    return kulczynski_values

def calculate_cosine(rules, data):
    # Calculate Cosine measure
    # Cosine = P(A and B) / sqrt(P(A) * P(B))
    cosine_values = []
    for rule in rules:
        antecedent = rule[0]
        consequent = rule[1]
        support_A_B = np.sum((data.loc[:, antecedent] == 1) & (data.loc[:, consequent] == 1)) / len(data)
        support_A = np.sum(data.loc[:, antecedent] == 1) / len(data)
        support_B = np.sum(data.loc[:, consequent] == 1) / len(data)
        cosine = support_A_B / np.sqrt(support_A * support_B)
        cosine_values.append(cosine)
    return cosine_values








