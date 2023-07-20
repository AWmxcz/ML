
#Librerías a utilizar
import pandas as pd
import numpy as np

#Graficas 
import graphing
import missingno as ms
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import plotly.figure_factory as ff

#Modelos 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Metricas
from sklearn.metrics import balanced_accuracy_score, roc_curve, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest

#Transformaciones, separaciones  e imputaciones  
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold

def CurvaROC(model, X_test_roc, X_train_roc, y_train_roc, y_test_roc):
    """ Esta función hace la grafíca ROC """

    y_scores_train = model.predict_proba(X_train_roc)
    y_scores_test = model.predict_proba(X_test_roc)

    auc_train = roc_auc_score(y_train_roc,y_scores_train[:,1])
    auc_test = roc_auc_score(y_test_roc,y_scores_test[:,1])
    # calculate ROC curve
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_roc, y_scores_train[:,1])
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test_roc, y_scores_test[:,1])

    # plot ROC curve
    fig, ax = plt.subplots(figsize = (10,7))
    
    ax.set_title('Verdaderos positivos vs. falsos positivos')
    ax.plot(fpr_train,tpr_train, label = "Entrenamiento") # graficamos la curva ROC para el set de entrenamiento
    ax.plot(fpr_test,tpr_test, label = "Evaluacion") # graficamos la curva ROC para el set de evaluacion
    
    ax.set_xlabel('Tasa de falsos positivos') # Etiqueta del eje x
    ax.set_ylabel('Tasa de verdaderos positivos') # Etiqueta del eje y
    
    
    plt.legend()

    print('AUC entrenamiento: {}'.format(round(auc_train,4)))  
    print('AUC evaluacion: {}'.format(round(auc_test,4)))

    return plt.show()
    

def fit_and_test_model(model,X_train, X_valid, y_train, y_valid):
    '''
    Esta función valida los Modelos 
    '''  
    

    # Train the model
    model.fit(X_train, y_train)

    # Assess its performance
    # -- Train
    predictions = model.predict(X_train)
    train_accuracy = balanced_accuracy_score(y_train, predictions)

    # -- Test
    predictions = model.predict(X_valid)
    test_accuracy = balanced_accuracy_score(y_valid, predictions)

    return train_accuracy, test_accuracy
    # We use plotly to create plots and charts


    # Build and print our confusion matrix, using the actual values and predictions 
    # from the test set, calculated in previous cells

def matrix_confusion(model, X_train, X_valid, y_train, y_valid):

    model.fit(X_train, y_train)

        # Assess its performance
        

        # -- Test
    predictions = model.predict(X_valid)

    cm = confusion_matrix(y_valid, predictions, normalize=None)
    # Create the list of unique labels in the test set, to use in our plot
    # I.e., ['animal', 'hiker', 'rock', 'tree']
    x=y= sorted(list(y_valid.unique()))

    # Plot the matrix above as a heatmap with annotations (values) in its cells
    fig = ff.create_annotated_heatmap(cm, x, y)

    # Set titles and ordering
    fig.update_layout(  title_text="<b>Confusion matrix</b>", 
                        yaxis = dict(categoryorder = "category descending"))

    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted label",
                            xref="paper",
                            yref="paper"))

    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.15,
                            y=0.5,
                            showarrow=False,
                            text="Actual label",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # We need margins so the titles fit
    fig.update_layout(margin=dict(t=80, r=20, l=100, b=50))
    fig['data'][0]['showscale'] = True
    fig.show()
def matrix_confusion_Xgboost(model, X_train, X_valid, y_train, y_valid):

    model.fit(X_train, y_train, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)

        # Assess its performance
        

        # -- Test
    predictions = model.predict(X_valid)

    cm = confusion_matrix(y_valid, predictions, normalize=None)
    # Create the list of unique labels in the test set, to use in our plot
    # I.e., ['animal', 'hiker', 'rock', 'tree']
    x=y= sorted(list(y_valid.unique()))

    # Plot the matrix above as a heatmap with annotations (values) in its cells
    fig = ff.create_annotated_heatmap(cm, x, y)

    # Set titles and ordering
    fig.update_layout(  title_text="<b>Confusion matrix</b>", 
                        yaxis = dict(categoryorder = "category descending"))

    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted label",
                            xref="paper",
                            yref="paper"))

    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.15,
                            y=0.5,
                            showarrow=False,
                            text="Actual label",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # We need margins so the titles fit
    fig.update_layout(margin=dict(t=80, r=20, l=100, b=50))
    fig['data'][0]['showscale'] = True
    fig.show()


def FeatureBest(X,y,model, n):
    Kbest = n # los mejores K que voy a retener
    skf = StratifiedKFold(n_splits=5, shuffle=True) # 5 folds es un número típico si tenemos suficientes datos. Pedimos shuffle=True para que sea al azar la separación en subgrupos
    skf.get_n_splits(X, y.values) # arma los folds a partir de los datos

    auc_values_fs =  []  # aca es donde van a ir a parar los indices de los features seleccionados en cada fold
    selected_features= np.array([]).reshape(0,X.shape[1]) # aca es donde van a ir a parar los AUCs de cada fold. El reshape es para poder concatenar luego.

    for train_index, test_index in skf.split(X, y): # va generando los indices que corresponden a train y test en cada fold
        X_train, X_test = X[train_index], X[test_index] # arma que es dato de entrenamiento y qué es dato de evaluación
        y_train, y_test = y[train_index], y[test_index]     # idem con los targets

        # scaler = MinMaxScaler() # escaleo por separado ambos sets
        # scaler.fit(X_train) 
        # X_train = scaler.transform(X_train)

        # scaler = MinMaxScaler() # escaleo por separado ambos sets
        # scaler.fit(X_test) 
        # X_test = scaler.transform(X_test)

        selector = SelectKBest(k=Kbest) # por defecto, usa el F score de ANOVA y los Kbest features
        selector.fit(X_train, y_train) # encuentro los F scores 
        X_train_fs = selector.transform(X_train) # me quedo con los features mejor rankeados en el set de entrenamiento
        X_test_fs = selector.transform(X_test) # me quedo con los features mejor rankeados en el set de evaluacion
        features = np.array(selector.get_support()).reshape((1,-1)) # esto me pone True si la variable correspondiente fue seleccionada y False sino

        selected_features =  np.concatenate((selected_features,features),axis=0)

        # Inicializamos nuevamente el modelo. max_iter es la cantidad de iteraciones maximas del algoritmo de optimizacion de parametros antes de detenerse.
        model.fit(X_train_fs, y_train) # Ajustamos el modelo con los datos de entrenamiento
        probas_test = model.predict_proba(X_test_fs)  # probabilidades con datos de evaluación
        #fpr_test, tpr_test, thresholds_test = roc_curve(y_test, probas_test[:,1]) # para plotear curva ROC con datos de entrenamiento
        auc_test = roc_auc_score(y_test, probas_test[:,1]) #  AUC con datos de evaluación
        auc_values_fs.append(auc_test)

    print('El AUC promedio es:')
    print(np.mean(auc_values_fs))



    plt.bar(np.arange(0,X.shape[1]),np.sum(selected_features,axis=0))
    plt.title('Seleccion de features')
    plt.xlabel('Feature')
    plt.ylabel('Folds')
    return plt.show()


    