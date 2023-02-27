import dash
from dash import html, dcc , callback , Output, Input
from flask_login import current_user
from utils.login_handler import require_login

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash.dependencies import Input, Output, State

from sklearn import datasets

import pandas as pd

import plotly.express as px
import io
import base64
from flask import url_for 
from pandas_profiling import ProfileReport
import utils.dash_reusable_components as drc
import utils.figures as figs
import dash_dangerously_set_inner_html

dash.register_page(__name__, path="/")
require_login(__name__)


data=None

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
        #html.Hr(),
        # html.P(
        #     "Number of students per education level", className="lead"
        # ),
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

    return   html.Div(children=[sidebar,html.Div(style=CONTENT_STYLE, children=[
    html.H2('Upload a file',style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and drop or click to select a file'
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow only CSV, XLS, and XLSX files to be uploaded
        accept='.csv,.xls,.xlsx'
    ),
    html.Div(id='output-data',style={"text-align":"center"}) ,
    
        ])]) 

@callback(Output('output-data', 'children'),
              Input('upload-data', 'contents'))
def update_output(contents):
    global data 
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Try to read the file as CSV
            data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except:
            try:
                # Try to read the file as XLS or XLSX
                data = pd.read_excel(io.BytesIO(decoded))
            except:
                return html.Div([
                    'The file you have uploaded is not a CSV, XLS, or XLSX file.'
                ])
        
        profile = ProfileReport(data, title="Pandas Profiling Report")
        profile.to_file("./assets/your_report.html")

        # Display the dataframe as a table
        return (html.Div([html.Iframe(src='/assets/your_report.html',height=800 ,width=1200)]))
print (type(data))
    

