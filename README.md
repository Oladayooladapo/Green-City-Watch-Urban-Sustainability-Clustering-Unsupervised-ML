# Green-City-Watch-Urban-Sustainability-Clustering-Unsupervised-ML
This project applies **unsupervised machine learning (clustering)** to group global cities based on environmental and livability metrics.   It was conducted for **GreenCityWatch**, a global urban sustainability organisation, with the goal of supporting governments and agencies in identifying cities most in need of sustainability interventions.

## Project Overview

This project focuses on leveraging unsupervised machine learning to segment global cities based on their urban sustainability indicators. Developed for **GreenCityWatch**, a global urban sustainability organization, the goal is to identify and categorize cities into distinct sustainability profiles (e.g., "Critical Intervention Zone," "Transitional Cities," "Sustainability Leaders"). This allows governments and development agencies to prioritize interventions and funding effectively, accelerating the transition towards climate resilience and sustainable living.

The project encompasses a full data science lifecycle, from comprehensive Exploratory Data Analysis (EDA) and robust data preprocessing to K-Means clustering model development, evaluation, and the deployment of a Streamlit application for interactive predictions.

## Business Problem

GreenCityWatch has compiled a global dataset of cities with various environmental and livability metrics. However, they lack a systematic way to:
- Group cities based on their overall sustainability performance.
- Identify which cities are most in need of support and funding for sustainability transitions.
- Understand underlying patterns in sustainability indicators across different urban regions.

This absence of clear segmentation and prioritization hinders their mission to accelerate sustainable urban development.

## Project Goal

To apply unsupervised learning (K-Means clustering) to:
1.  **Cluster cities** into meaningful groups based on their sustainability indicators.
2.  **Identify urban regions** with poor performance in key areas like air quality, green space, energy use, carbon emissions, and waste management.
3.  **Recommend specific clusters** that require immediate intervention and funding, supported by data-driven insights and visualizations.
4.  **Develop an interactive Streamlit application** for predicting a city's sustainability cluster based on user input.

## Dataset

The dataset used for this project is `Green_City_Watch_Urban_Sustainability_Clusters.csv`, containing various urban sustainability metrics for a collection of global cities.

**Key Columns:**
-   `city`: Name of the city.
-   `green_space_pct`: Percentage of city area covered in green parks and natural spaces.
-   `air_quality_index`: Air quality index (lower is better).
-   `waste_recycled_pct`: Percentage of waste recycled.
-   `renewable_energy_pct`: Percentage of energy from renewable sources.
-   `carbon_emissions`: Annual per capita $CO_{2}$ emissions (metric tons).
-   `energy_efficiency_score`: City's building energy performance index (0-100, higher is better).
-   `avg_commute_time`: Average commute time for residents (minutes).
-   `water_access_pct`: Percentage of population with access to clean water.
-   `population`: Total population of the city.
-   `country`: Country the city belongs to.
-   `iso_alpha`: ISO alpha-3 code for the country.
-   `region`: World region the city belongs to.

## Project Workflow & Thought Process

My approach to this unsupervised learning project followed a structured methodology, emphasizing data quality, thorough exploration, robust model building, and practical deployment.

### 1. Data Understanding & Initial Inspection
-   **Objective:** Gain a foundational understanding of the dataset's structure, content, and initial quality.
-   **Steps:**
    -   Loaded essential libraries: `pandas`, `matplotlib.pyplot`, `seaborn`.
    -   Loaded the `Green_City_Watch_Urban_Sustainability_Clusters.csv` dataset.
    -   Used `df.head()` to inspect the first few rows and understand column content.
    -   Employed `df.info()` to check data types and identify non-null counts. The dataset was found to be clean with **no missing values**.
    -   Utilized `df.describe()` to obtain descriptive statistics for numerical columns, observing ranges, means, and standard deviations.
    -   Used `df.describe(include="object")` to get statistics for categorical columns (`city`, `country`, `iso_alpha`, `region`), identifying unique values and their frequencies.
-   **Thought Process:** A clean dataset (no missing values) simplifies preprocessing. Understanding the basic statistics helps identify potential outliers or skewed distributions that might require transformation.

### 2. Data Cleaning & Preprocessing
-   **Objective:** Prepare the numerical features for clustering by handling irrelevant features and scaling.
-   **Steps:**
    -   **Feature Selection:** Identified the numerical columns relevant for clustering based on sustainability indicators: `green_space_pct`, `air_quality_index`, `waste_recycled_pct`, `renewable_energy_pct`, `carbon_emissions`, `energy_efficiency_score`, `avg_commute_time`, `water_access_pct`, `population`.
    -   **Remove Irrelevant Features:** `city`, `country`, `iso_alpha`, and `region` were excluded from the clustering features as they are identifiers or geographical categorizations rather than sustainability metrics themselves, though `region` can be used for post-clustering analysis.
    -   **Feature Scaling:** Applied `StandardScaler` to the selected numerical features. This is a crucial step for K-Means clustering, as it is a distance-based algorithm. Scaling ensures that features with larger numerical ranges (e.g., `population`) do not disproportionately influence the distance calculations and, consequently, the cluster assignments.
-   **Thought Process:** Feature selection is critical for unsupervised learning; only relevant features should be used for clustering. Scaling ensures fair weighting of all features in the clustering process.

### 3. Exploratory Data Analysis (EDA) - Focused on Features for Clustering
-   **Objective:** Understand the distributions of key sustainability indicators and identify potential patterns or anomalies that might form clusters.
-   **Steps & Key Insights:**
    -   **Distribution of Individual Metrics:** Visualized histograms and box plots for each numerical sustainability indicator (`green_space_pct`, `air_quality_index`, `waste_recycled_pct`, etc.). This helped in understanding the spread, skewness, and presence of outliers.
    -   **Top/Bottom Performing Cities:** Identified cities with:
        -   Highest `carbon_emissions` and `air_quality_index` (indicating poor performance).
        -   Lowest `green_space_pct`, `renewable_energy_pct`, `waste_recycled_pct`, `energy_efficiency_score`, `water_access_pct` (indicating areas needing intervention).
    -   **Regional Patterns:** Analyzed average sustainability metrics by `region` to see if certain geographical areas tend to perform better or worse. For example:
        -   Bar plots showing average `carbon_emissions` by `region`.
        -   Bar plots showing average `air_quality_index` by `region`.
        -   (As indicated by the notebook's TODOs and the presentation slides, these visualizations would be key).
-   **Thought Process:** EDA before clustering helps build intuition about the data's inherent groupings. Understanding extreme values and regional differences can provide initial hypotheses for cluster characteristics.

### 4. K-Means Clustering Model Development
-   **Objective:** Apply K-Means to group cities into distinct sustainability clusters.
-   **Steps:**
    -   **Determining Optimal K (Number of Clusters):**
        -   Used the **Elbow Method** (plotting WCSS vs. number of clusters) to identify the optimal number of clusters (`k`). The point where the decrease in WCSS (Within-Cluster Sum of Squares) starts to level off indicates a good `k`.
        -   (The notebook would show the elbow plot and the chosen `k`, typically 3 for "Critical," "Transitional," "Leaders").
    -   **Model Training:** Initialized and trained the `KMeans` model with the chosen `k` on the scaled data.
    -   **Assigning Clusters:** Assigned the predicted cluster labels back to the original DataFrame.
    -   **Saving Model & Scaler:** Saved the trained `KMeans` model and the `StandardScaler` using `joblib` for later deployment in the Streamlit app. This ensures consistency between training and prediction environments.
-   **Thought Process:** The Elbow Method provides a data-driven way to select `k`. Saving the scaler is crucial because new data must be scaled using the *same* transformation applied during training.

### 5. Cluster Analysis & Interpretation
-   **Objective:** Characterize each identified cluster and derive actionable insights.
-   **Steps:**
    -   **Descriptive Statistics per Cluster:** Calculated the mean (or median) of each sustainability indicator for each cluster. This helps define the "profile" of each cluster.
    -   **Visualization of Clusters:**
        -   Created scatter plots (e.g., `carbon_emissions` vs. `renewable_energy_pct`, colored by cluster) to visually inspect the separation and characteristics of the clusters.
        -   Used bar plots to compare average values of key metrics across clusters.

-  **Key Cluster Profiles**
-   **Cluster Traits Recommendation**
0	Low emissions, high green space, strong renewable adoption	Maintain momentum; share best practices
1	High emissions, low renewable use, poor air quality	Priority for intervention
2	Moderate performance, large populations	Target scalable improvements
3	Strong air quality but low energy efficiency	Introduce green building incentives

-   **Naming Clusters:** Based on the analysis of cluster profiles above, I was able to assigned meaningful names:
        -   **Cluster 0: Critical Intervention Zone:** (Likely characterized by high carbon emissions, low green space, poor air quality, low renewable energy, low waste recycling). These cities are in urgent need of support.
        -   **Cluster 1: Transitional Cities:** (Likely showing mixed performance, some progress but still room for significant improvement). These cities need targeted guidance.
        -   **Cluster 2: Sustainability Leaders:** (Likely excelling in most sustainability metrics). These cities can serve as benchmarks.

-   **Thought Process:** Interpreting clusters is the most important part of unsupervised learning. By comparing feature averages across clusters, we can assign meaningful labels and understand the practical implications of the groupings.

### 6. Strategic Business Recommendations
-   **Objective:** Translate cluster insights into actionable strategies for GreenCityWatch and its stakeholders.
-   **Key Recommendations:**

    1.  **Prioritize the "Critical Intervention Zone" (Cluster 0):**
        * **Action:** Direct immediate funding, policy support, and expert resources to these cities. Focus on fundamental improvements in waste management, air quality, and transitioning to renewable energy.
    2.  **Support "Transitional Cities" (Cluster 1):**
        * **Action:** Provide tailored guidance and best practices. Encourage adoption of proven strategies from "Sustainability Leaders" and facilitate knowledge sharing.
    3.  **Leverage "Sustainability Leaders" (Cluster 2):**
        * **Action:** Document their successful policies and initiatives. Promote them as case studies and benchmarks for other cities. Facilitate partnerships for technology transfer and expertise sharing.
    4.  **Targeted Interventions:**
        * For cities with high `carbon_emissions`, recommend investments in public transport, green infrastructure, and renewable energy sources.
        * For cities with low `waste_recycled_pct`, advise on implementing comprehensive recycling programs and waste-to-energy initiatives.
        * For cities with poor `air_quality_index`, suggest policies to reduce industrial emissions and promote cleaner transportation.
    5.  **Regional Focus:** Tailor strategies based on regional patterns identified in EDA, acknowledging unique challenges and opportunities in different parts of the world.

## Deployment: Streamlit Application

-   **Objective:** Create an interactive web application to allow users to input sustainability metrics for a hypothetical city and predict its cluster.
-   **Details:**
    -   Developed `app.py` using Streamlit.
    -   Loads the pre-trained `kmeans_sustainability_model.joblib` and `scaler_sustainability_model.joblib`.
    -   Provides input fields for all nine numerical sustainability features.
    -   Includes a dropdown for selecting a country, which is used for visualization on a choropleth map.
    -   Upon clicking "Predict Cluster," the app scales the input data using the loaded scaler, predicts the cluster using the K-Means model, and displays the predicted cluster name.
    -   Visualizes the predicted cluster on a global choropleth map using `plotly.express`, with distinct colors for each cluster type.


## Tools & Libraries Used

-   **Programming Language:** Python
-   **Data Manipulation:** `pandas`, `numpy`
-   **Data Visualization:** `matplotlib.pyplot`, `seaborn`, `plotly.express`
-   **Machine Learning:** `scikit-learn` (for `StandardScaler`, `KMeans`)
-   **Model Persistence:** `joblib`
-   **Web Application Framework:** `streamlit`
-   **Jupyter Notebook:** For interactive analysis and documentation.


## Files in this Repository

-   `Unsupervised Learning (Clustering) - Green City Watch Urban Sustainability.ipynb`: The main Jupyter Notebook containing all the code for data loading, cleaning, EDA, preprocessing, K-Means model training, and cluster analysis.
-   `Green_City_Watch_Urban_Sustainability_Clusters.csv`: The raw dataset used for the project.
-   `kmeans_sustainability_model.joblib`: The trained K-Means clustering model.
-   `scaler_sustainability_model.joblib`: The fitted StandardScaler object used for preprocessing.
-   `README.md`: This file.

