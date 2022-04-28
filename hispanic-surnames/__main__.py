import pandas as pd
import os
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

def rename_index(f_country):
    """
    Returns corret country names
    """
    country = f_country[2:]
    country = country.replace('_',' ')
    country = country.title()
    
    return country

def calculate_densities(col, countries):
    """
    Returns relative densities using variation of tf-idf
    """
    res = col.copy()
    for country in countries:
        x = res[country] * math.log((len(col) / sum(col.mask(col != 0, other = 1))))
        res[country] = x
    
    return res

def get_inertia_values(features):
    """
    Iterates and finds the intertias for 1-11 clusters
    """  
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
    }

    sse = []
    
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)
        
    return sse

def clean_and_transform_dataframe(df):
    """
    Fixes column names, removes unecessary columns and pivots dataframe
    """  
    df.fillna(0, inplace = True)
    frequency_columns = [col for col in df if col.startswith('f_') or col == 'Surname']
    df_frequency = df[frequency_columns]
    df_frequency = df_frequency.transpose()
    df_frequency = df_frequency.rename(columns = df_frequency.iloc[0])
    df_frequency = df_frequency.iloc[1: , :]
    df_frequency.rename(mapper = rename_index, inplace = True)
    countries = df_frequency.transpose().columns
    df = df_frequency.apply(calculate_densities, countries = countries)

    return (df, df_frequency)

def load_spain_dataset(path, filename):
    """
    Loads Spain.xlsx separately
    """      
    df_spain = pd.read_excel(path + '/data' + filename)
    df_spain = df_spain.rename(columns = {"f_españa" : "spain_frequency"})
    df_spain.drop('i_españa', axis = 1, inplace = True)

    return df_spain

def get_rank(df, colname):
    """
    Adds rank column of last name by frequency
    """  
    df['Rank'] = df.sort_values(by = [colname], ascending = False) \
            .reset_index() \
            .sort_values('index') \
            .index + 1

    return df

def validate(features):
    """
    Implements "elbow test" to validate hyperparameter k
    """  
    sse = get_inertia_values(features)
    elbow = pd.DataFrame(range(1,11),sse)

    return elbow

def calculate_clusters(df):
    """
    Implements K means on frequencies dataframe
    """      
    features = df.iloc[:,1:] 
    kmeans = KMeans(3) 

    identified_clusters = kmeans.fit_predict(features)

    df_clusters = df.copy()
    df_clusters['Cluster'] = identified_clusters 
    df_clusters = df_clusters[['index', 'Cluster']]

    df.set_index('index', inplace=True) # Set index

    return (df, df_clusters, features)


def get_densities_by_country(df):
    """
    Transforms the dataset for analysis using country as category
    Also fixes some spelling issues
    """  
    df = pd.melt(df.reset_index(), id_vars = 'index', var_name = 'Apellido', value_name = 'y')
    df.rename({'index' : 'País'}, axis = 'columns', inplace = True)
    df['y'] = df['y'].astype(float)
    df['y'] = df['y'] / df['y'].max() # normalize
    df['c'] = pd.Categorical(df['País'], ["Mexico", "Guatemala", "El Salvador", "Honduras", "Nicaragua", "Costa Rica",
                                        "Panama", "Colombia", "Venezuela", "Ecuador", "Peru", "Bolivia",
                                        "Chile", "Argentina", "Paraguay", "Uruguay", "Cuba", "Dominicana", "Puerto Rico"])
    df.sort_values(by = ['Apellido', 'c'], inplace = True)

    # Add graphic accents
    df.loc[df["País"] == "Mexico", "País"] = "México"
    df.loc[df["País"] == "Dominicana", "País"] = "República Dominicana"
    df.loc[df["País"] == "Panama", "País"] = "Panamá"
    df.loc[df["País"] == "Peru", "País"] = "Perú"

    return df

def get_top_5(df):
    """
    Gets top 5 surnames by relative density per country
    """  
    top5 = df.groupby(['País'])['Apellido','y'].apply(lambda x: x.nlargest(5, columns=['y']))

    return top5
    

def main():

    outdir = './clean_data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Load data
    path = os.getcwd()
    files = os.listdir(path + '/data')
    df = pd.DataFrame(columns = ['Surname'])

    # Iterate through source data, except for Spain
    for f in files:
        if f != 'españa.xlsx':
            f = 'data/' + f
            data = pd.read_excel(f)
            df = pd.merge(df,data, on = 'Surname', how = 'outer')

    df_frequency = clean_and_transform_dataframe(df)[1]
    df = clean_and_transform_dataframe(df)[0]
    
    # Aggregate frequencies
    df_agg = df_frequency.sum(axis = 0).sort_values(ascending = False).head(10)

    # Create dataframe for Spain
    df_spain = load_spain_dataset(path, '/españa.xlsx')

    # Create dataframe for the rest of LatAm
    df_latam = pd.DataFrame(df_agg).reset_index()
    df_latam.rename(columns = {"index" : "Surname", 0 : "latam_frequency"}, inplace = True)

    # Merge Latin America and Spain dataframes
    df_latam = get_rank(df_latam, 'latam_frequency')
    df_spain = get_rank(df_spain, 'spain_frequency')
    df_rank = df_latam.merge(df_spain, on = 'Surname', how='left')
    df_rank.rename(columns = {"Rank_x" : "latam_rank", "Rank_y" : "spain_rank"}, inplace = True)
    df_rank = df_rank[['Surname', 'latam_rank', 'spain_rank']]
    df_rank.to_csv(outdir + "/ranking.csv", mode='w+')

    # Reset index
    df_frequency.reset_index(drop=False, inplace=True) 

    # Calculate clusters using KMeans
    df, df_clusters, features = calculate_clusters(df_frequency)
    df_clusters.to_csv(outdir + "/clusters.csv", mode='w+')

    # Validate with elbow test
    elbow = validate(features)
    elbow.to_csv("clean_data/elbow.csv", mode='w+')

    # Get last name densities per country
    df = get_densities_by_country(df)
    df.to_csv(outdir + "/densities.csv", mode='w+')

    # Get top 5 most dense last names per country
    top5 = get_top_5(df)
    top5.to_csv(outdir + "/top_5.csv", mode='w+')

    print("Success! Check out clean_data on the parent directory")

if __name__=="__main__":
    main()
    