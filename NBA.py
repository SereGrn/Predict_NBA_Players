# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:17:50 2022

@author: seren
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns 
from xgboost import XGBClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from dash import Dash, html, dcc, Output, Input, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px 
import plotly.figure_factory as ff
from sklearn.decomposition import PCA 


def score_classifier(dataset,labels,classifier): 
    
    """ 
    Train a classifier by maximizing precision, using a cross-validation by 3Fold.
    Displays the confusion matrix
    @param dataset : Array type, dataset without the target 
    @param labels : Array type, list of target per observation
    @param classifier : Function type, classifier to use 
    
    """
    
    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    precision = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        precision += precision_score(test_labels, predicted_labels)
        
    precision/=3

    print(f"precision : {precision}")
    
    sns.heatmap(confusion_mat, annot=True) 
    plt.show()
    print(confusion_mat)
    return precision


df = pd.read_csv("\nba_logreg.csv")

# =============================================================================
# Pre-processing
# =============================================================================

# Types of variables 
df.dtypes

# Missing values 
df.isna().sum()

## 11 missing values for 3points Attempts % 
df_na = df[df['3P%'].isna() == True]
df_na.head(11)
df_na['3P Made']
df_na['3PA']

df['3P%'].fillna(0, inplace = True) # replacing by 0 because na occurs when 3pts made & 3pts attemps are 0. 

# Outliers 
df_boxplot = df.drop(columns = ['Name','TARGET_5Yrs'])
green_diamond = dict(markerfacecolor='g', marker='D')

for i in df_boxplot.columns:
    plt.boxplot(df[i],flierprops = green_diamond)
    plt.title("BoxPlot de " + i)
    plt.show()

# Many extreme values in the data, a robust method for normalization should be used

outliers_3pt = df[df['3P%'] == df['3P%'].max() ]
outliers_3pt
# % of 3 points attempted at 100% when 0 3 points attempted or made: we think of an input error 
df['3P%'] = np.where( df['3P%'] == df['3P%'].max(), 0,df['3P%'])


df_corr = df_boxplot.corr()
# Correlation between variables 
sns.heatmap(df_corr)

# MIN & PTS & FGM & FGA & FTM & FTA & TOV 
# OREB & DREB & REB  

# Perfom a PCA on variables that are more than 80% correlated with each other 
df_corr = df_corr[df_corr.iloc[:,:] >= 0.8]

# 1st group of variables
df_PCA1 =  df[['MIN','PTS','FGM','FGA','FTM','FTA','TOV']]
scaler = RobustScaler() 
df_PCA1 = scaler.fit_transform(df_PCA1)

pca = PCA()
pca = pca.fit(df_PCA1)
pca.explained_variance_ratio_ # 1st component explains 88% of the inertia 

pca = PCA(1)
df_pca1 = pd.DataFrame(pca.fit_transform(df_PCA1), columns = ['Component_1'])

# 2nd group of variables
df_PCA2 =  df[['OREB','DREB','REB']]
df_PCA2 = scaler.fit_transform(df_PCA2)

pca = PCA()
pca = pca.fit(df_PCA2)
pca.explained_variance_ratio_ # 1st component explains 94.6% of the inertia 

pca = PCA(1)
df_pca2 = pd.DataFrame(pca.fit_transform(df_PCA2), columns = ['Component_2'])

# drop correlated variables and add components 
df_model = df.drop(columns = ['MIN','PTS','FGM','FGA','FTM','FTA','TOV','OREB','DREB','REB'])
df_model = pd.concat([df_model,df_pca1,df_pca2], axis = 1)

# check target balance 
count_classes = pd.value_counts(df['TARGET_5Yrs'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(len(df['TARGET_5Yrs'].unique())), df.TARGET_5Yrs.unique())
plt.title("Count of the modalities of the variable TARGET_5Yrs")
plt.xlabel("Duration of the career (1 = 5 years and more)")
plt.ylabel("Number of observations");

(df['TARGET_5Yrs'].value_counts()/len(df))*100

# Duplicate check 
duplicate = df_model[df_model.duplicated(keep = False)] 
duplicate

# Keep first line of duplicates
df_model.drop_duplicates(keep = 'first', inplace=True)
duplicate = df_model[df_model.duplicated(keep = False)] 
duplicate

# =============================================================================
# Modeling
# =============================================================================

# extract names, labels, features names and values
names = df_model['Name'].values #.tolist() # players names
labels = df_model['TARGET_5Yrs'].values # labels
paramset = df_model.drop(['TARGET_5Yrs','Name'],axis=1).columns.values
df_vals = df_model.drop(['TARGET_5Yrs','Name'],axis=1)

# normalize dataset
X = RobustScaler().fit_transform(df_vals.values) # robust outlier standardization


models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVC', SVC()),
          ('XGBM', XGBClassifier()),
          ('GB',GradientBoostingClassifier())]
          

modele = []
precision = []
          
for i, j in models:
    precision.append(score_classifier(X,labels,j))
    modele.append(i)
    
dict_from_list = dict(zip(modele, precision))
sorted(dict_from_list.items(), key=lambda t: t[1])

# SVC, LR and GB best precision 

# SVC tuning 
model = SVC()

parametres = {"C" : np.arange(1,3,0.1),
              "kernel" : ['linear', 'poly', 'rbf', 'sigmoid']
             }
                
best_model_SVC = GridSearchCV(model,
                            parametres,
                            cv=10,
                            n_jobs=-1, scoring = "precision",
                            verbose=2).fit(X, labels)
best_model_SVC.best_params_

score_classifier(X,labels, SVC(C = best_model_SVC.best_params_['C'],
                        kernel = best_model_SVC.best_params_['kernel']))

# 0.747 precision and 229 false positives & 678 true positives 

# LR tuning
model = LogisticRegression()

parametres = {'penalty' : ['l1','l2'], #['l1','l2','elasticnet'],
              "C" : np.logspace(-4, 4, 20)
              
             }
                
best_model_LR = GridSearchCV(model,
                            parametres,
                            cv=5,
                            n_jobs=-1, scoring = "precision",
                            verbose=True).fit(X, labels)
best_model_LR.best_params_

score_classifier(X, labels, LogisticRegression( C = best_model_LR.best_params_['C'],
                                      penalty = best_model_LR.best_params_['penalty']))

# 0.747 preicision and 232 false positives & 685 true positives 


# RF tuning
model = RandomForestClassifier()

parametres = {'n_estimators' : range(10,100,10),
              'criterion' : ['gini','entropy','log_loss']
           
             }
                
best_model_RF = GridSearchCV(model,
                            parametres,
                            cv=10,
                            n_jobs=-1, scoring = "precision",
                            verbose=2).fit(X, labels)
best_model_RF.best_params_

score_classifier(X, labels, RandomForestClassifier(n_estimators = best_model_RF.best_params_['n_estimators'], 
                                                   criterion = best_model_RF.best_params_['criterion']))

# 0.736 precision and 227 false positives & 633 true positives

# best precision is for SVC and LR model, but with RF we have less false positive 

# =============================================================================
# API 
# =============================================================================

# pre-processing before building the API

# Elements layout 1 
# Number of observations and variables 
n = len(df)
d = len(df.columns) - 1 

# df without Name et target 
df_num = df.drop(columns = ['Name','TARGET_5Yrs'])

# Mean and sd of each columns
mean_columns = []
sd_columns = []
for i in df_num.columns:
    mean_columns.append(round(df_num[i].mean(),2))
    sd_columns.append(round(df_num[i].std(),2))
    
# insert "/" for variables names and target 
mean_columns.insert(0," / ")
mean_columns.insert(len(mean_columns), " / ")

sd_columns.insert(0," / ")
sd_columns.insert(len(sd_columns), " / ")


# description of each columns in a df : 
Descript = pd.DataFrame({
    "Variable": df.columns,
    "Description": ["Name", "Game Played","Minutes Played","Points per game","Field goals attempts",
                    "Field goal Attempts","Field goald percent","3 points Made","3 points attemps",
                    "3 points percent","Free throw made","Free throw attempts","Free throw percent",
                    "Offensive Rebounds","Defensive rebounds","Rebounds","Assists","Steals","Blocks",
                    "Turnovers","Target : 1 if career length > 5 years, O otherwise"],
    "Mean" : mean_columns,
    "Standard Deviation" : sd_columns
})


## Elements layout 3

## new scoring function to keep predicted labels and names of players  
def score_classifier(dataset,labels,classifier):
    
    kf = KFold(n_splits=3,random_state=50,shuffle=True)
    confusion_mat = np.zeros((2,2))
    precision = 0
    recall = 0
    accuracy = 0
    index = []
    true_label = []
    label_pred = []
    importances = []
    
    for training_ids,test_ids in kf.split(dataset):
        index.append(names[test_ids])
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        
        confusion_mat += confusion_matrix(test_labels,predicted_labels)
        precision += precision_score(test_labels, predicted_labels)
        accuracy += accuracy_score(test_labels, predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
        
       
        label_pred.append(predicted_labels)
        true_label.append(test_labels)
        importances.append(classifier.feature_importances_)
    precision/=3
    accuracy /= 3
    recall /= 3

    return confusion_mat, precision, accuracy, recall, index, label_pred, true_label, importances


mat, precision, accuracy, recall, index, label_pred, label, importances = score_classifier(X,
                labels,RandomForestClassifier(n_estimators = best_model_RF.best_params_['n_estimators'] , criterion = best_model_RF.best_params_['criterion']))

# metrics of the model in a table
Score = pd.DataFrame({
    "Metrics" : ["Accuracy","Recall","Precision"],
    "Score" : [round(accuracy,2),round(recall,2),round(precision,2)]
    })


# confusion mat
confusion_mat = px.imshow(mat,text_auto=True,
                          labels=dict(x="Predicted label", y="True label", color="Numbers of players"),
                          x = ['0','1'],
                          y = ['0','1']
                          )

# labels, names of each fold 
list_names1 = pd.DataFrame(index[0], columns = ['Name'])
list_names2 =  pd.DataFrame(index[1], columns = ['Name'])
list_names3 = pd.DataFrame(index[2], columns = ['Name'])

names = pd.concat([list_names1, list_names2, list_names3])

list_label1 = pd.DataFrame(label[0], columns = ['True label'])
list_label2 = pd.DataFrame(label[1], columns = ['True label'])
list_label3 = pd.DataFrame(label[2], columns = ['True label'])

true_label = pd.concat([list_label1, list_label2, list_label3])

list_label_pred1 = pd.DataFrame(label_pred[0], columns = ['Predicted label'])
list_label_pred2 = pd.DataFrame(label_pred[1], columns = ['Predicted label'])
list_label_pred3 = pd.DataFrame(label_pred[2], columns = ['Predicted label'])

predicted_label = pd.concat([list_label_pred1, list_label_pred2, list_label_pred3])


list_players = pd.concat([names, true_label, predicted_label], axis = 1)

# list of players with predicted label of 1
players = list_players[ list_players['Predicted label'] == 1]

players = pd.DataFrame({
    "Player's name" : players['Name']
    })


# feature importances for each fold
feat1 = pd.DataFrame(importances[0])
feat2 = pd.DataFrame(importances[1])
feat3 = pd.DataFrame(importances[2])

feat_importances = pd.concat([feat1, feat2, feat3], axis = 1)
feat_importances = pd.DataFrame(feat_importances.mean(axis = 1))
feat_importances.set_index(df_vals.columns, inplace = True)

importances_features = px.bar( x=feat_importances.index, y=feat_importances[0],
                              labels=dict(x="Variables", y="Importances"))


# Creation of the API 
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Style and sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# style of the application
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.H2("Menu", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Presentation of the data", href="/", active="exact"),
                dbc.NavLink("Data visualization", href="/page-1", active="exact"),
                dbc.NavLink("Summary", href="/page-2", active="exact"),
                dbc.NavLink("Customizable request", href = "/page-3", active = "exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(id="page-content", style=CONTENT_STYLE)


app.layout = html.Div([
    dcc.Location(id="url"), 
    sidebar,
    content])

# page 1
page_1_layout = html.Div(
    [
    html.Div(id = "page1"),
    html.H1('Description of the data',
            style={
            'textAlign': 'center'}),
    html.Br(),
    html.Div(
    html.H6('The database has ' + str(n) + " NBA players and " + str(d) + " variables to describe them.")
    ),
    html.Br(),
    html.Br(),
    html.H6("Here is the list and a short description of the variables : "),
    html.Br(),

    html.Div(
        dash_table.DataTable(Descript.to_dict('records'),
                         style_cell={'textAlign': 'left',
                                     'border': '1px solid grey' },
                         style_as_list_view=True,
                         style_header = { 'fontWeight': 'bold' },
                         style_data={'whiteSpace': 'normal',
                                     'width': 'auto'}
    )  )
    
    ])

# page 2
page_2_layout = html.Div([
    html.H1("Data visualization",
            style={
            'textAlign': 'center'}),
    
    dbc.Row(
        [
            dbc.Col(
                html.Div( children = [
                html.Label("Selecting a variable"),
                        dcc.Dropdown(id ="var",
                            options = df_num.columns,
                            value = "GP"
 
                ) ]) ),
            
            dbc.Col(
                html.Div( children = [
                html.Label("Choosing a graph"),
                
                        dcc.Dropdown(id ="graph",
                            options = ['BoxPlot', 'kdeplot'],
                            value = "BoxPlot"

                )
               ] )
            
             )
        
    ] ),
    
    html.Div(
        dcc.Graph(id = "graphic")
        
        ),
    
    dbc.Row(
        [
            dbc.Col(
                html.Div( children = [
                    html.Label("Selecting two variables"),
                    dcc.Dropdown(id ="var1",
                        options = df_num.columns,
                        value = "GP"),
                    
                    dcc.Dropdown(id ="var2",
                        options = df_num.columns,
                        value = "MIN"),
                    
                    html.Div(dcc.Graph(id = "graph3"))
            
            
                    ]),
                
                ),
            
            dbc.Col(
                html.Div(
                    dcc.Graph(id = "graph2")))
            
            ]
        )
   ] )
            
                            
                               

@callback(
    Output("graphic","figure"),
    Output("graph2","figure"),
    [
    Input("var","value"),
    Input("graph","value"),
    Input("var1","value"),
    Input("var2","value"),
    ],
    )

def make_graph1(var,graph,var1,var2):
    if graph == "BoxPlot":
          fig = px.box(y=df[var], width=500, height=400)
          fig.update_layout(title_text="BoxPlot of " + var)
          
          fig2 = px.scatter(df, x = var1, y = var2) 
          fig2.update_layout(title_text="Scatter plot of variables " + var1 + " and " + var2)

          return fig, fig2
    else:
        colors = ['slategray', 'magenta']
        
        df_0 = df[df['TARGET_5Yrs'] == 0]
        df_1 = df[df['TARGET_5Yrs'] == 1]
        
        group_data = [df_0[var], df_1[var]]
        
        fig = ff.create_distplot(group_data, ['Target = 0','Target = 1'],
                                  show_hist=False,
                                  colors=colors)
        
        # Add title
        fig.update_layout(title_text="Distribution of " + var)
        
        fig2 = px.scatter(df, x = var1, y = var2) 
        fig2.update_layout(title_text="Scatter plot of variables " + var1 + " and " + var2)
        return fig, fig2



@callback(
    Output("graph3","figure"),
    [
      Input("var1","value"),
      Input("var2","value"),
      ]
    )

def make_graph2(var1,var2):
    df_corr = df[[var1,var2]]
    fig = px.imshow(df_corr.corr(), text_auto=True)
    return fig 


# page 3
page_3_layout = html.Div(
    [
     html.Div(id = "page3"),
     html.H1("Summary of the model",
             style={
             'textAlign': 'center'}),
     
     html.Br(),
     html.H6("We therefore tried to classify NBA players according to the expected length of their career in order to help investors."),
     html.H6("The best classifier for our problem, i.e. to classify a maximum of players while minimizing the false positive rate, is the Random Forest type classifier"),
     
     html.Br(),
     dbc.Row(
         [
             
             dbc.Col(
                 html.Div( children = [
                     html.H2("Synthesis of the classifier performances",
                     style={
                     'textAlign': 'center'}),
                     html.Br(),
                     dash_table.DataTable(Score.to_dict('records'),
                                      style_cell={'textAlign': 'left',
                                                  'border': '1px solid grey' },
                                      style_as_list_view=True,
                                      style_header = { 'fontWeight': 'bold' },
                                      style_data={'whiteSpace': 'normal',
                                                  'width': 5}
                 ),
                     html.Br(),
                     html.Label("Confusion matrix :",
                                style={
                                'textAlign': 'center',
                                'fontWeight': 'bold'}),
                     dcc.Graph(
                                id='example-graph',
                                figure = confusion_mat
                            ),
                     
                     html.Br(),
                     html.Label("Features importances :",
                                style={
                                'textAlign': 'center',
                                'fontWeight': 'bold'}),
                     dcc.Graph(
                         id = 'importances',
                         figure = importances_features
                         ),
                     html.Br(),
                     html.H6("Note that component_1 is the first principal component obtained after a PCA performed on the following highly correlated variables : MIN, PTS, FGM, FGA, FTM, FTA, TOV."),
                     html.H6("Component_2 is also the first principal component of the PCA on OREB, DREB, REB")
             ])
             ),
             
             dbc.Col( 
                 html.Div( children = [
                     html.H2('List of players to invest in',
                    style={
                    'textAlign': 'center'}),
                     html.H6("There are a total of " + str(len(players)) + " players whose careers are expected to last more than 5 years."),
                     html.Button("Download the list", id ="dowload_but"),
                     dcc.Download(id = "dowload_button"),
                     html.Br(),
                     dash_table.DataTable(players.to_dict('records'),
                                      style_cell={'textAlign': 'left',
                                                  'border': '1px solid grey' },
                                      style_as_list_view=True,
                                      style_header = { 'fontWeight': 'bold' },
                                      style_data={'whiteSpace': 'normal',
                                                  'width': 'auto'}
                 )
                     
                     
          ])
            )

         
       ]  )
    

   ])

@callback(
    Output("dowload_button","data"),
    Input("dowload_but","n_clicks"),
    prevent_initial_call = True,
    
    )

def download(n_clicks):
    return dcc.send_data_frame(players.to_excel, "List_investment_players.xlsx", sheet_name="Sheet_name_1")

# page 4
page_4_layout = html.Div(
    [
     html.Div(id = "page4"),
     html.H1("Customizable query",
             style={
             'textAlign': 'center'}),
     
     html.Br(), 
     html.H6("Search for a player's name to see their statistics :"),
     html.Br(), 
     dcc.Dropdown ( id ="player",
                  options = df['Name'].unique(),
                  value = 'Tony Bennett'),
     
     html.Br(),
     html.Div(id='info'),
     html.Br(),
     html.Div(id = 'info2'),
     html.Br(),
     html.Div(id = 'info3')

     
     ])


@callback(
    Output("info","children"),
    [
     Input("player","value"),
     ]
    
    )

def information(player):
    df_filter = df[df['Name'] == player]
    filter_player = list_players[list_players['Name'] == player]
    if len(df_filter) > 1:
        return "Attention, there are several players named like this, impossible to dissociate them! Please enter another player!"
    if (len(df_filter) == 1) & (filter_player.iloc[0,2] == 1): 
        return "We must invest in this player"
    elif (len(df_filter) == 1) & (filter_player.iloc[0,2] == 0): 
            return "Don't invest in this player"
    
    
@callback(
    Output("info2","children"),
    [
      Input("player","value"),
      ]
    
    )

def info2(player):
    df_filter = df[df['Name'] == player]
    df_filter.drop(columns = ['TARGET_5Yrs'], inplace = True)
    return  dash_table.DataTable(df_filter.to_dict('records'),
                          style_cell={'textAlign': 'left',
                                      'border': '1px solid grey' },
                          style_as_list_view=True,
                          style_header = { 'fontWeight': 'bold' },
                          style_data={'whiteSpace': 'normal',
                                      'width': 'auto'}
     )

        
@callback(
    Output("info3","children"),
    [
      Input("player","value"),
      ]
    
    )

def info3(player):
    df_filter = df[df['Name'] == player]
    df_filter.drop(columns = ['TARGET_5Yrs'], inplace = True)
    NB = df_filter.iloc[0,1]
    PTS = df_filter.iloc[0,3]
    texte = "" + player + " played " + str(NB) + " games, and scored an average of " + str(PTS) + " points per game."
    if len(df_filter) == 1:
        return  texte
    else:
        return "No information!"


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return page_1_layout
       
          
    elif pathname == "/page-1":
        return page_2_layout
    
    elif pathname == "/page-2":
        return page_3_layout
        
    elif pathname == "/page-3":
        return page_4_layout 
    
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == '__main__':
    app.run_server(debug=False)