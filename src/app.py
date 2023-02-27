


import os
from flask import Flask, request, redirect, session
from flask_login import login_user, LoginManager, UserMixin, logout_user, current_user
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from utils.login_handler import restricted_page
import dash_bootstrap_components as dbc

import secrets

from flask import url_for 
os.environ["SECRET_KEY"] = secrets.token_hex(16)

# Exposing the Flask Server to enable configuring it for logging in
server = Flask(__name__)


@server.route('/login', methods=['POST'])
def login_button_click():
    if request.form:
        username = request.form['username']
        password = request.form['password']
        if VALID_USERNAME_PASSWORD.get(username) is None:
            return """invalid username and/or password <a href='/login'>login here</a>"""
        if VALID_USERNAME_PASSWORD.get(username) == password:
            login_user(User(username))
            if 'url' in session:
                if session['url']:
                    url = session['url']
                    session['url'] = None
                    return redirect(url) ## redirect to target url
            
            return redirect('/') ## redirect to home
        return """invalid username and/or password <a href='/login'>login here</a>"""


app = dash.Dash(
    __name__, server=server, use_pages=True, suppress_callback_exceptions=True ,meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server=app.server 
app._favicon = ("assets/favicon.ico")
app.title = "First dash app"
# Keep this out of source code repository - save in a file or a database
#  passwords should be encrypted
VALID_USERNAME_PASSWORD = {"test": "test", "hello": "world"}


# Updating the Flask Server configuration with Secret Key to encrypt the user session cookie
server.config.update(SECRET_KEY=os.getenv("SECRET_KEY"))

# Login manager object will be used to login / logout users
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = "/login"


class User(UserMixin):
    # User data model. It has to have at least self.id as a minimum
    def __init__(self, username):
        self.id = username


@login_manager.user_loader
def load_user(username):
    """This function loads the user by user id. Typically this looks up the user from a user database.
    We won't be registering or looking up users in this example, since we'll just login using LDAP server.
    So we'll simply return a User object with the passed in username.
    """
    return User(username)




# styling the sidebar
SIDEBAR_STYLE = {
     "left": 0,
   "position": "fixed",
    "top": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "Number of students per education level", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("SVM", href="/page-2", active="exact"),
                dbc.NavLink("Random forest", href="/page-3", active="exact"),
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
#content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

banner=html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Support Vector Machine (SVM) Explorer",
                                    href="https://github.com/plotly/dash-svm",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                             id="banner-logo",
                             children=[
                                 html.Img(src=app.get_asset_url("dash-logo-new.png"))
                             ],
                             href="https://www.value.com.tn/",
                         )
                    ]
                )
            ]
        )







app.layout = html.Div(
    [
        dcc.Location(id="url"),
        #html.Div(id="user-status-header"),
        #sidebar,
        banner,
        # html.Hr(),
        dash.page_container,
    ]
)








@app.callback(
    #Output("statenav", "children"),
    Output('url','pathname'),
    Input("url", "pathname"),
    Input({'index': ALL, 'type':'redirect'}, 'n_intervals')
)
def update_authentication_status(path, n):
    ### logout redirect
    if n:
        if not n[0]:
            return  dash.no_update
        else:
            return  '/login'

    ### test if user is logged in
    if current_user.is_authenticated:
        if path == '/login':
            return  '/'
        return  dash.no_update
    else:
        ### if page is restricted, redirect to login and save path
        if path in restricted_page:
            session['url'] = path
            return  '/login'

    ### if path not login and logout display login link
    if current_user and path not in ['/login', '/logout']:
        return  dash.no_update

    ### if path login and logout hide links
    if path in ['/login', '/logout']:
        return  dash.no_update



if __name__ == "__main__":
    app.run_server()