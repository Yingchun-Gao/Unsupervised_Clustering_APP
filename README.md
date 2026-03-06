# Mall Customer Segmentation (Unsupervised Clustering)

This project is deployed as an interactive Streamlit application:

https://yingchun-gao-unsupervised-clustering-app.streamlit.app/

This application segments mall customers using **KMeans clustering**.  
The goal is to identify groups of customers with similar purchasing behavior.  
These segments can help businesses better understand customer profiles and target marketing strategies.

## Features

- Complete data preprocessing pipeline
- Feature scaling using StandardScaler
- KMeans clustering model
- Cluster evaluation using Elbow Method and Silhouette Score
- Visualization of customer segments
- Interactive Streamlit dashboard

## Dataset

The dataset contains mall customer information.

Features used for clustering:

- **Age** – Age of the customer
- **Annual Income** – Annual income (in thousands)
- **Spending Score** – Score assigned based on spending behavior

The **Customer_ID** column is removed during preprocessing because it has no analytical value.

## Technologies Used

- **Streamlit** – Web application interface
- **Scikit-learn** – Clustering algorithm and evaluation
- **Pandas** – Data preprocessing
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization