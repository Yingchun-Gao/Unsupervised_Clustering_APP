# unsupervised_clustering
This app has been built using Streamlit and deployed with Streamlit community cloud

https://yingchun-gao-real-estate-price-prediction.streamlit.app/

This project applies KMeans clustering to segment mall customers based on their behavior. The goal is to identify groups of customers with similar characteristics so businesses can better understand purchasing patterns.

## Features
- Data preprocessing pipeline
- Feature scaling for clustering
- KMeans clustering model
- Elbow method and Silhouette score for cluster evaluation
- Visualization of customer segments
- Interactive Streamlit application

## Dataset
Features used for clustering:
- Age - Age of the customer
- Annual Income - Annual income in thousands
- Spending Score - Score assigned based on spending behavior

The Customer_ID column is removed during preprocessing because it has no analytical value.

## Technologies Used
- **Streamlit** - Web application interface
- **Scikit-learn** - Model training and evaluation
- **Pandas** - Data preprocessing 
- **NumPy** - Numerical operations
- **Matplotlib** - Feature importance visualization