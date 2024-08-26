
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from math import sqrt

# Load the TensorFlow model
loaded_model = tf.keras.models.load_model(
    'my_model.h5',
    custom_objects={'InputLayer': InputLayer}
)

# Company logo and title
st.image('switcher-CS-2x.png', width=50)
st.markdown('<h1 style="color:#004085;">CrimeStoppers Trust</h1>', unsafe_allow_html=True)

# Main app section
st.markdown('<h2 style="color:#0056b3;">Crime Prediction Near Infrastructure</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:#383d41;">Providing safety insights near key infrastructure assets</h3>', unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header('Input Features')

# Updated list of key infrastructure assets
infrastructure_data = {
    "Belfast Port": (-5.90612312450567, 54.6179059307711),
    "Dover Port": (1.31348435794760, 51.12109445788082),
    "FelixTowe Port": (1.3236771534242822, 51.94958820033655),
    "Grimsby Port": (-79.55138841046319, 43.20462435449015),
    "Hartlepool Port": (-1.202566617251045, 54.70521093133299),
    "Immingham Port": (-0.1880165733876612, 53.62977848111598),
    "Liverpool Port": (-3.0287171849617938, 53.45761189538581),
    "Milford Haven Port": (-5.052638188821906, 51.707818587603086),
    "Port of London Authority": (0.395638869708862, 51.443084214693286),
    "Southampton Port": (-1.4354268597772608, 50.91057383287762),
    "Tees Port": (-1.1517549388603077, 54.61380384046637),
    "London Heathrow Airport (LHR)": (-0.4550832014594401, 51.468158387474894),
    "London Gatwick Airport (LGW)": (-0.18201998700055266, 51.15379666132887),
    "London Stansted Airport (STN)": (0.26171922751194737, 51.890090695059165),
    "London Luton Airport (LTN)": (-0.37265844278940274, 51.87559444554112),
    "London Southend Airport (SEN)": (0.6924988427502838, 51.57033980739883),
    "London City Airport (LCY)": (0.049593099522232384, 51.5050173033172),
    "Manchester Airport": (-2.27696889530223, 53.35548487081846),
    "Newcastle Airport": (-1.7099805040012814, 55.03723022362747),
    "Leeds Bradford Airport": (-1.6586384430052505, 53.86685683537879),
    "Liverpool Airport": (-2.8522708003897668, 53.33525090959623),
    "Birmingham International": (-1.7424353187424722, 52.45249156262602),
    "East Midlands Airport": (-1.3321661887680822, 52.8294387947214),
    "Bristol Airport": (-2.7172436041788157, 51.383166403331295),
    "Exeter Airport": (-3.4148794912759435, 50.735128946252516),
    "Edinburgh Airport": (-3.36309345608355, 55.94872109698716),
    "Glasgow Airport": (-4.433851052952626, 55.870257804876495),
    "Aberdeen Airport": (-2.2000647093447077, 57.203807886631495),
    "Cardiff Airport": (-3.373672161735299, 51.50322412663002),
    "Belfast International": (-6.2065672978064415, 54.660242505925495),
    "Birmingham New Street": (52.477910802309424, -1.8988614427604904),
    "Bristol Temple Meads": (51.449658482890825, -2.580510715821907),
    "Clapham Junction": (51.46410784952202, -0.17011407533962045),
    "Edinburgh Waverley": (55.95210024651832, -3.1898635732725173),
    "Glasgow Central": (55.85924424242692, -4.258140988618711),
    "Guildford": (51.23743771724084, -0.5825180446673522),
    "Leeds City": (53.79605514354437, -1.5484177445441423),
    "Liverpool Lime Street": (51.51889844597763, -0.08087950288514983),
    "London Bridge": (51.50440271448311, -0.08475948698393773),
    "London Cannon Street": (51.51141112969868, -0.09035961766660003),
    "London Charing Cross": (51.50829085784499, -0.12477313116063984),
    "London Euston": (51.52819429074685, -0.1332734581474344),
    "London King’s Cross": (51.530809607565246, -0.1232309753364447),
    "London Liverpool Street": (51.518871741619826, -0.0810404354157968),
    "London Paddington": (51.516854634623506, -0.1769328023248663),
    "London St Pancras International": (51.531242983274, -0.12585517348879188),
    "London Victoria": (51.49539729908572, -0.1437476986306047),
    "London Waterloo": (51.50340017432952, -0.11222862931320994),
    "Manchester Piccadilly": (53.477595356288624, -2.2311988887366083),
    "Reading": (51.45849777074971, -0.9714906311630112),
    "Drax Power Station": (53.7415891908775, -0.9981828391845192),
    "Heysham Nuclear Power Station": (54.02954067292075, -2.9149192941483553),
    "Pembroke Power Station": (51.683051420083224, -4.994804786975456),
    "Peterhead Power Station": (57.4781174435026, -1.7880362676522126),
    "Ratcliffe Power Station": (52.86543221506888, -1.254833424298609),
    "Dinorwig Power Station": (53.118705686774675, -4.103428532930965),
    "Staythorpe C Power Station": (53.07492138370051, -0.8552876028104598),
    "Grain CHP": (51.443268376918965, 0.7077631553424332),
    "Hornsea 2 Offshore Wind Farm": (53.6569432974762, 1.8177003096746216),
    "Wilton International Power Station": (54.584102969101226, -1.043699388243283),
    "Connahs Quay Power Station": (53.231641958260205, -3.0812983721198703),
    "South Humber Bank Power Station": (53.6021924216449, -0.1449392877235352),
    "Torness Nuclear Power Station": (55.969270928603, -2.4061145315028756),
    "Ballylumford Power Station": (54.843992939064684, -5.786601708266597),
    "Dogger Bank A Wind Farm": (51.45783338140948, -0.965415327467728),
    "Saltend Power Station": (53.73521883010514, -0.24335360406553688),
    "Sizewell B Nuclear Power Station": (52.21534973645022, 1.6202163799589062),
    "Hartlepool Nuclear Power Station": (54.63566514095673, -1.1809248119721796),
    "Immingham Power Station": (53.638813385120145, -0.23337494141698245),
    "Seabank Power Station": (51.53948600649043, -2.669680904731861),
    "Seagreen Wind Farm": (51.45793365682278, -0.9648788856989046),
    "Walney Wind Farm": (54.10929774899547, -3.2260521367279744)
}

# Function to get user input
def get_user_input():
    crime_latitude = st.sidebar.number_input('Crime Latitude', min_value=-90.0, max_value=90.0, value=0.0)
    crime_longitude = st.sidebar.number_input('Crime Longitude', min_value=-180.0, max_value=180.0, value=0.0)
    infrastructure_type = st.sidebar.selectbox('Select Infrastructure', list(infrastructure_data.keys()))
    
    infrastructure_latitude, infrastructure_longitude = infrastructure_data[infrastructure_type]
    
    # Calculate Euclidean distance between crime and infrastructure
    distance = sqrt((crime_latitude - infrastructure_latitude)**2 + (crime_longitude - infrastructure_longitude)**2)
    
    data = {
        'latitude': crime_latitude,
        'longitude': crime_longitude,
        'distance': distance
    }
    features = pd.DataFrame(data, index=[0])
    return features, infrastructure_type

# Preprocess the input data
def preprocess_input(input_df):
    # Placeholder: load pre-trained scaler if used
    # scaler = joblib.load('scaler.joblib')
    # input_data = scaler.transform(input_df)
    
    # In this version, we're bypassing scaling
    return input_df.values

# Predict function
def predict_crime_tensorflow(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0][0]

# Get user input
input_df, selected_infrastructure = get_user_input()

# Predict and display result
if st.sidebar.button('Predict'):
    input_features = preprocess_input(input_df)
    prediction = predict_crime_tensorflow(loaded_model, input_features)
    
    st.markdown(f'<h2 style="color:#0056b3;">Prediction</h2>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:#383d41;">Predicted Crime Count Near {selected_infrastructure}: {prediction:.0f}</h3>', unsafe_allow_html=True)
    
    # Highlight potential threat
    if prediction >= 10:
        st.markdown('<div style="background-color:#ffc107;color:#212529;font-weight:bold;padding:10px;border-radius:10px;">Warning: High potential threat to infrastructure!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<h3 style="color:#383d41;">No significant threat detected.</h3>', unsafe_allow_html=True)

