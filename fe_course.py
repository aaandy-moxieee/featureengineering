import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.cluster import KMeans

#graph style formatting with matplotlib

plt.style.use('ggplot')

#store course data file paths in variables
abalone_filepath = r'course data\abalone.csv'
accidents_filepath = r'course data\accidents.csv'
customer_filepath = r'course data\customer.csv'
concrete_filepath = r'course data\concrete.csv'
automobile_filepath = r'course data\autos.csv'

#read stored data files and store into dfs
accidents_dataset = pd.read_csv(accidents_filepath)
customer_dataset = pd.read_csv(customer_filepath)
concrete_dataset = pd.read_csv(concrete_filepath)
autos_dataset = pd.read_csv(automobile_filepath)

#showing heads (5) of all datasets
accidents_dataset.head(5)
customer_dataset.head(5)
concrete_dataset.head(5)
autos_dataset.head(5)

#Performing Mathematical transformations on datasets
#1st Transform - Stroke ratio (Simple numerical feature combination)
autos_dataset['stroke_ratio'] = autos_dataset.stroke / autos_dataset.bore
#More complex combination
autos_dataset["displacement"] = (
    np.pi * ((0.5 * autos_dataset.bore) ** 2) * autos_dataset.stroke * autos_dataset.num_of_cylinders
)

autos_dataset.head(5)

#2nd Transform - Log transformations on accidents daTaset
accidents_dataset['logWindSpeed'] = accidents_dataset.WindSpeed.apply(np.log1p)

accidents_dataset.head(5)

sns.kdeplot(data=accidents_dataset.logWindSpeed, shade =True)

accidents_dataset.columns

roadway_Features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]

accidents_dataset['roadwayFeatures'] = accidents_dataset[roadway_Features].sum(axis=1)

accidents_dataset.head(10)

#3rd Transform - create boolean to count presence of component in formulation

components = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]

concrete_dataset['ComponentsInForm'] = concrete_dataset[components].gt(0).sum(axis=1)

concrete_dataset.head(20)
concrete_dataset.tail(20)

#Building-up and Breaking down Structured Features.

customer_dataset[['Type', 'Level']] = (
    customer_dataset['Policy'].str.split(' ', expand=True)
)

customer_dataset[['Policy','Type','Level']].head(20)

#Making a more complex feature from simple features

autos_dataset['make_model'] = autos_dataset['make'] + ' ' + autos_dataset['body_style']
autos_dataset.head(20)

#Group transforms

customer_dataset['avgIncome'] = (
    customer_dataset.groupby('State')
    ['Income'].transform('mean')
)

customer_dataset.head(20)

#DF built-in methods passed as strings to Transform function

customer_dataset['StateFreq'] = (
    customer_dataset.groupby('State')
    ['State'].transform('count') / customer_dataset.State.count()
)
###########################################################################

#'Frequency encoding' on categorical variables
#
#create training and validation sets (no training,just for feature engineering purposes)
df_train = customer_dataset.sample(frac=0.7)
df_valid = customer_dataset.drop(df_train.index)

df_train['avgClaim'] = (
    df_train.groupby('Coverage') #Select the column to group by
    ['ClaimAmount'].transform('mean') #Select column to transform and apply mean transformation
)

df_valid = df_valid.merge(
    df_train[['Coverage','avgClaim']].drop_duplicates(),
    on= 'Coverage',
    how= 'left',
)

df_valid.head(20)

############################################################################

