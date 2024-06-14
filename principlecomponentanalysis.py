import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from IPython.display import display

plt.style.use('ggplot')

#Store and load data
abalone_filepath = r'course data\abalone.csv'
auto_filepath = r'course data\autos.csv'
abalone_dataset = pd.read_csv(abalone_filepath)
auto_dataset = pd.read_csv(auto_filepath)

#Define plot_var and get_mi_score functions

def plot_var(pca, width=8, dpi=100):
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n+1)
#Explained Var plot
    exp_var_ratio = pca.explained_variance_ratio_
    axs[0].bar(grid,exp_var_ratio)
    
#Cumlative Var plot
    cum_var = np.cumsum(exp_var_ratio)
    axs[1].plot(np.r_[0, grid], np.r_[0,cum_var], 'o-')
    
#plot set up
    fig.set(figwidth=8, dpi=100)
    return axs

def get_mi_score(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]

X = auto_dataset.copy()
y = X.pop('price')
X = X.loc[:,features]

#Standardizing data
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

#Import PCA
from sklearn.decomposition import PCA

#creating PCs
pca = PCA()
prin_comp = pca.fit_transform(X_scaled)

#convert components to DF
comp_names = [f'PC{i+1}' for i in range (prin_comp.shape[1])]
prin_comp = pd.DataFrame(prin_comp,columns=comp_names)

prin_comp.head(20)

#####loadings of the components are stored in component_ attribute of PCA after fitting
#Accessing and converting the loadings

loadings = pd.DataFrame(
    pca.components_.T , # transpose the matrix of loadings
    columns=comp_names, # so the columns are the principal components
    index=X.columns, # and the rows are the original features
)
loadings


#plot var vizze
plot_var(pca);

#mutual info scores
mi_scores = get_mi_score(prin_comp,y, discrete_features=False)
mi_scores

#DF sorted by PC3
idx = prin_comp["PC3"].sort_values(ascending=False).index
cols = ["make", "body_style", "horsepower", "curb_weight"]
auto_dataset.loc[idx, cols]

#creating new feature for ratio between curb_weight and horsepower
auto_dataset['performance_or_wagon'] = X.curb_weight / X.horsepower
sns.regplot(data=auto_dataset, x='performance_or_wagon', y='price', order=2)