# dash_eda_and_modeling

This repository contains the code for an interactive data analysis dashboard built using Dash, Flask, Pandas, Scikit-learn, and Plotly. The app has 6 pages that allow users to perform exploratory data analysis, principal component analysis (PCA), and various supervised learning techniques such as Support Vector Machines (SVM), Random Forest, and XGBoost.

## Pages

1. **Login**: Users can log in to the app using their credentials, which are handled by Flask's authentication system.
2. **EDA**: Users can perform exploratory data analysis (EDA) using Pandas profiling, which generates interactive HTML reports with visualizations and summary statistics.
3. **PCA**: Users can upload their data and perform PCA analysis to reduce the dimensionality of the data and visualize it in 2D or 3D using Plotly.
4. **SVM**: Users can apply Support Vector Machines (SVM) to their data, choose the hyperparameters, and visualize the results using Plotly.
5. **Random Forest**: Users can apply Random Forest to their data, choose the hyperparameters, and visualize the results using Plotly.
6. **XGBoost**: Users can apply XGBoost to their data, choose the hyperparameters, and visualize the results using Plotly.

## Usage

To run the app locally, follow these steps:

1. Clone this repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Set up the database by running `python setup.py`.
4. Run the app using `python app.py`.
5. Open your web browser and go to `http://localhost:8050/` to access the app.

## Contributing

Contributions are welcome! If you find any bugs or have any suggestions for improvement, please create a new issue or submit a pull request.
