import dash
from dash import html, dcc , callback , Output, Input
from flask_login import current_user
from utils.login_handler import require_login
import time
import importlib
import dash_auth
import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import io
import base64
from flask import url_for 
from sklearn.ensemble import RandomForestClassifier


import utils.dash_reusable_components as drc
import utils.figures as figs

dash.register_page(__name__ )
require_login(__name__)




SIDEBAR_STYLE = {
    "left": 0,
   "position": "fixed",
    "top": "25rem",
    #"bottom": "50%",
    "width": "16rem",
    "height":"30rem",
    "padding": "2rem 1rem",
    "background-color": "#2f3445",
}
sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        
        dbc.Nav(
            [
                             dbc.NavLink("Pandas profiling", href="/", active="exact"),
                dbc.NavLink("PCA", href="/page-1", active="exact"),
                dbc.NavLink("SVM", href="/page-2", active="exact"),
                dbc.NavLink("random forest",href="/page-3", active="exact"),
                dbc.NavLink("Xgboost", href="/page-4", active="exact"),
                
                dbc.NavLink("logout",id="statenav",href="/logout", active="exact"),
                #dbc.NavItem(id="statenav"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}



def layout():
    if not current_user.is_authenticated:
        return html.Div(["Please ", dcc.Link("login", href="/login"), " to continue"])

    return  html.Div(
    children=[sidebar,html.Div(style=CONTENT_STYLE,children=[
       
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[dcc.Upload(
                                    id='upload-data',
                                    children=html.Div([
                                        'Drag and Drop or ',
                                        html.A('Select Files')
                                    ]),)
                                        
                                    ],
                                ),
                           
                                drc.Card(
                                    id="last-card",
                                    children=[
                                         drc.NamedSlider(
                                             name="threshold",
                                              id="slider-threshold",
                                              min=0,
                                              max=1,
                                              value=0.5,
                                             step=0.01,
                                             marks= None ,
                                             
                                           tooltip={"placement": "bottom", "always_visible": True}
                                          ),
                                        drc.NamedSlider(
                                             name="min_samples_split",
                                              id="slider-min-samples-split",
                                              min=2,
                                              max=10,
                                              value=5,
                                             step=1,
                                             marks= None ,
                                             
                                           tooltip={"placement": "bottom", "always_visible": True}
                                          ),

                                        
                                        drc.NamedSlider(
                                            name="n-estimator",
                                            id="slider-nestimator",
                                            min=50,
                                            max=500,
                                            value=100,
                                            step=1,
                                             marks= None ,
                                             
                                           tooltip={"placement": "bottom", "always_visible": True}
                                          
                                        ),
                                        drc.FormattedSlider(
                                            id="max-depth",
                                            min=3,
                                            max=20,
                                            value=6,
                                             step=1,
                                             marks= None ,
                                             
                                           tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                        
                                        drc.NamedSlider(
                                            name="min_samples_leaf",
                                            id="slider-min-samples-leaf",
                                            min=1,
                                            max=5,
                                            value=3,
                                            step=1,
                                             marks= None ,
                                             
                                           tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                        # drc.NamedSlider(
                                        #     name="max_features",
                                        #     id="slider-max-features",
                                        #     min=0.1,
                                        #     max=1,
                                        #     value=1,
                                        #      step=0.01,
                                        #      marks= None ,
                                             
                                        #    tooltip={"placement": "bottom", "always_visible": True}
                                        # ),
                                        
                                        
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-RFgraphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)])



@callback(
    Output("div-RFgraphs", "children"),
    [
        
        Input("slider-min-samples-split", "value"),
        Input("slider-min-samples-leaf", "value"),
        Input("max-depth", "value"),
        Input('upload-data', 'contents'),
    
        Input("slider-nestimator", "value"),
        
        Input("slider-threshold", "value"),
        
    ],
    [dash.dependencies.State('upload-data', 'filename')]
)

def update_RF_graph(
    min_samples_split,
    min_samples_leaf,
    max_depth,
   
    contents,
    
    n_estimators,
    
    threshold ,
    filename
    
):
    content_type, content_string = contents.split(',')
    df = pd.read_csv(
         io.StringIO(base64.b64decode(content_string).decode('utf-8'))
     )
    
    
    # Perform PCA on the data
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.iloc[:, :-1])
    pca_df = pd.DataFrame({
        'x': pca_result[:, 0],
        'y': pca_result[:, 1],
        #'z': pca_result[:, 2],
        'class': df[df.columns[-1]]
        
    })
    t_start = time.time()
    h = 0.3  # step size in the mesh

    # Data Pre-processing
    #X, y = generate_data( dataset=pca_df)
    X  = pca_df.iloc[:, :-1].values
    y  = pca_df.iloc[:, -1:].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5
    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

 

    # Train SVM
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features='sqrt')

    clf.fit(X_train, y_train)

 
    if hasattr(clf, "decision_function"):
         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
         Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

 

    roc_figure = figs.serve_xgboost_roc_curve(model=clf, X_test=X_test, y_test=y_test)

    confusion_figure = figs.serve_xgboost_pie_confusion_matrix(
        model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
    )

    return [
      
        html.Div(
            id="graphs-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-pie-confusion-matrix", figure=confusion_figure
                    ),
                ),
            ],
        ),
    ]
