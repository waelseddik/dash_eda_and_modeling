import dash
from dash import html, dcc, Output, Input, callback
import dash_bootstrap_components as dbc
from flask_login import current_user
from utils.login_handler import require_login
import base64
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import io

dash.register_page(__name__)
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

    return  html.Div(children=[sidebar,
    html.Div(style=CONTENT_STYLE , children=
    [   
        html.P("Please choose the components number"),
        dcc.Dropdown(
            
            id="component-number",
            options=[{"label": i, "value": i} for i in [2, 3]],
            value="2",
        ),

        dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
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
        }
    ),

    html.Div(id='output-data-upload', style={"height":"800" }),
    ]
)])


def parse_contents(contents, filename,n):
    content_type, content_string = contents.split(',')
    df = pd.read_csv(
        io.StringIO(base64.b64decode(content_string).decode('utf-8'))
    )
    X  = df.iloc[:, :-1]
    
    
    # Perform PCA on the data
    pca = PCA(n)
    pca_result = pca.fit_transform(X)
    if n==3:
         pca_df = pd.DataFrame({
        'x': pca_result[:, 0],
        'y': pca_result[:, 1],
        'z': pca_result[:, 2],
        'class': df[df.columns[-1]]})
         fig = px.scatter_3d(
        pca_df,
        x='x',
        y='y',
        z='z',
        height=800,
        color='class')
        
        
    else :
        pca_df = pd.DataFrame({
        'x': pca_result[:, 0],
        'y': pca_result[:, 1],
        
        'class': df[df.columns[-1]]})
        fig = px.scatter(
        pca_df,
        x='x',
        y='y',
        height=800,
        
        color='class')
    return dcc.Graph(figure=fig)


@callback(
    dash.dependencies.Output('output-data-upload', 'children'),
    [dash.dependencies.Input('upload-data', 'contents')],
    [dash.dependencies.State('upload-data', 'filename')],
    [dash.dependencies.Input('component-number', 'value')])


def update_output(contents, filename,n):
    if contents is None:
        return html.Div([
            'No data uploaded yet.'
        ])

    return parse_contents(contents, filename,n)





   