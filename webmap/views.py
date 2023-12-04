from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import sqlite3
import random
import pickle

base_dir = settings.BASE_DIR

con = sqlite3.connect(base_dir/'db.sqlite3')

df = pd.read_sql_query("SELECT * from sa_final", con).reset_index(drop=True)#db.sql lite

geom = gpd.points_from_xy(df['lon'], df['lat'], crs='EPSG:4326')
gdf = gpd.GeoDataFrame(df, geometry=geom, crs=geom.crs).to_crs('EPSG:4087')

scaler_uri = base_dir/'utils/raghad_scaler.pkl' 
kmeans_uri = base_dir/'utils/raghad_kmeans_20.pkl' #pkl file used for saved training model (saved models)

with open(scaler_uri, 'rb') as file:  
    scaler = pickle.load(file)

with open(kmeans_uri, 'rb') as file:  
    kmeans = pickle.load(file)

gdf['labels'] = kmeans.labels_


def recommend_locations(input_lat, input_long, num_recommendations, max_distance_km):
       input_loc = gpd.GeoSeries(Point(input_long,input_lat), crs='EPSG:4326').to_crs(gdf.crs)[0]
       
       index_of_nearest = gdf.distance(input_loc).idxmin()#geopandas - nearest point to the input at that point
       
       nearest_loc = gdf.iloc[index_of_nearest, :]#get nearest point
       cols = ['PVOUT_csi', 'DNI', 'GHI', 'DIF','GTI_opta', 'OPTA', 'TEMP','ELE']#extract att

       nearest_scaled = scaler.transform([nearest_loc[cols].values])#normalization
       
       cluster = kmeans.predict(nearest_scaled)#pass normalized data to ml to predict cluster it belongs too
       target = gdf[gdf['labels'] == cluster[0]]#get all the points that belong to the predicted cluster
       
    #    target = target[(target.distance(input_loc)/1000) > max_distance_km]
       s = random.choices(target.index, k=num_recommendations)#get randomly 5 of those locations
       
       return target.loc[s, :]

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def recommend(request):
       input_lat, input_long = request.GET.get('lat'), request.GET.get('lon')
       num_recommendations = request.GET.get('num_recommendations', 5)
       minimum_distance = request.GET.get('minimum_distance', 50)
       
       recommendations = recommend_locations(
          input_lat, input_long, num_recommendations, minimum_distance)
       
       return HttpResponse(recommendations.loc[:, recommendations.columns != 'geometry'].to_json(orient='index'))