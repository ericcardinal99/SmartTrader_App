from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import datetime
import random #temp
import pickle
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def process_data(input):
    date_results = np.empty(5, dtype='U10') #empty string array for the dates
    date_results = date_results.reshape(5, 1)
    decision_results = np.empty(5, dtype='U10') #empty string array for the action taken 
    decision_results = decision_results.reshape(5, 1)
    
    #increment the date, save next 7 days into an array
    date = datetime.datetime.strptime(input, "%Y-%m-%d")
    i = 0
    while i < 5:
        date = date + datetime.timedelta(days=(1)) #increment to next day
        if date.weekday() == 5 or date.weekday() == 6:
            continue
        date_results[i] = date.strftime("%Y-%m-%d")
        i += 1
    
    
    #TEMP for the model, which should return the prices in an array
    #print(input)
    matrix = generate_data(input)
    #print("Matrix",matrix)
    
    answer = np.array([[np.max(matrix[:, 1])], [np.min(matrix[:, 2])], [np.mean(matrix[:, 3])]]) # High, low and close
    answer = np.round(answer, 2)
    #print(answer)
    
    names = np.array([["Highest NVDA Price"], ["Lowest NVDA Price"], ["Average NVDA Closing Price"]])
    organized_matrix = np.hstack((names, answer))
    
    #choose an action to take
    #NVDA_shares = 10000
    #NVDQ_shares = 100000

    final_money = 0
    shares = 'NONE'

    for i in range(len(matrix)-1):
       
        #find out which stock has the higher open price
        NVDA_open_price = matrix[i][0]
        NVDQ_open_price = matrix[i][4]
        #print("NVDA Open Price:",NVDA_open_price)
        #print("NVDQ Open Price:",NVDQ_open_price)

        NVDA_next_day_open = matrix[i+1][0]
        NVDQ_next_day_open = matrix[i+1][4]

        NVDA_percent_change = (NVDA_next_day_open - NVDA_open_price) / NVDA_open_price
        NVDQ_percent_change = (NVDQ_next_day_open - NVDQ_open_price) / NVDQ_open_price

        print("Day",i+1," NVDA percent change:",NVDA_percent_change)
        print("Day",i+1," NVDQ percent change:",NVDQ_percent_change)
        print('---')
       
        if NVDA_percent_change > NVDQ_percent_change:
            decision_results[i] = "BULLISH"
        elif NVDA_percent_change < NVDQ_percent_change:
            decision_results[i] = "BEARISH"
        else:
            decision_results[i] = "IDLE" 


    #decision_results[:] = "IDLE"
    decision_results = np.hstack((date_results, decision_results))
    
    return organized_matrix, decision_results
    
def generate_data(input_date):

    #with open('mlp_model.pkl', 'rb') as file:
       #mlp_model = pickle.load(file)
        
    NVDA_url = "data_set/NVIDIApricehistory.csv"
    NVDQ_url = "data_set/NVDQpricehistory.csv"
    NVDA_data = pd.read_csv(NVDA_url)
    NVDQ_data = pd.read_csv(NVDQ_url)
    
    NVDA_data.dropna(inplace=True)
    NVDQ_data.dropna(inplace=True)
    NVDA_data['Date'] = pd.to_datetime(NVDA_data['Date'], format='%d-%b-%y')
    NVDQ_data['Date'] = pd.to_datetime(NVDQ_data['Date'], format='%d-%b-%y')
    
    NVDA_data.columns = NVDA_data.columns.str.strip()
    NVDQ_data.columns = NVDQ_data.columns.str.strip()
    NVDA_data['Open'] = NVDA_data['Open'].str.replace(',', '').astype(float) #some of the data set includes commas in the numbers, stripped it and coverted to float
    NVDA_data['Volume'] = NVDA_data['Volume'].str.replace(',', '').astype(float)
    NVDQ_data['Open'] = NVDQ_data['Open'].str.replace(',', '').astype(float)
    NVDQ_data['Volume'] = NVDQ_data['Volume'].str.replace(',', '').astype(float)
    
    NVDA_data.dropna(inplace=True)
    NVDA_data = NVDA_data.reset_index(drop=True)
    NVDQ_data.dropna(inplace=True)
    NVDQ_data = NVDQ_data.reset_index(drop=True)

    start_date = '2024-08-01' #for calculation use just 2 year data, maybe try iterating over this to see which produces best mse
    NVDA_data_reduced = NVDA_data[NVDA_data['Date'] >= pd.to_datetime(start_date)]
    NVDA_data_reduced = NVDA_data_reduced.copy() #excluding causes some warning about splice
    NVDQ_data_reduced = NVDQ_data[NVDQ_data['Date'] >= pd.to_datetime(start_date)]
    NVDQ_data_reduced = NVDQ_data_reduced.copy() #excluding causes some warning about splice

    days_to_predict = 1
    for day in range(1, days_to_predict + 1):
        NVDA_data_reduced.loc[:, f'high_price_day{day}_NVDA'] = NVDA_data_reduced['High'].shift(day)
        NVDA_data_reduced.loc[:, f'low_price_day{day}_NVDA'] = NVDA_data_reduced['Low'].shift(day)
        NVDA_data_reduced.loc[:, f'close_price_day{day}_NVDA'] = NVDA_data_reduced['Close'].shift(day)
        NVDA_data_reduced.loc[:, f'open_price_day{day}_NVDA'] = NVDA_data_reduced['Open'].shift(day)
        NVDA_data_reduced.loc[:, f'volume_price_day{day}_NVDA'] = NVDA_data_reduced['Volume'].shift(day)
        NVDA_data_reduced.loc[:, f'adj_close_price_day{day}_NVDA'] = NVDA_data_reduced['Adj Close'].shift(day)
        
    for day in range(1, days_to_predict + 1):
        NVDA_data_reduced.loc[:, f'high_price_day{day}_NVDQ'] = NVDQ_data_reduced['High'].shift(day)
        NVDA_data_reduced.loc[:, f'low_price_day{day}_NVDQ'] = NVDQ_data_reduced['Low'].shift(day)
        NVDA_data_reduced.loc[:, f'close_price_day{day}_NVDQ'] = NVDQ_data_reduced['Close'].shift(day)
        NVDA_data_reduced.loc[:, f'open_price_day{day}_NVDQ'] = NVDQ_data_reduced['Open'].shift(day)
        NVDA_data_reduced.loc[:, f'volume_price_day{day}_NVDQ'] = NVDQ_data_reduced['Volume'].shift(day)
        NVDA_data_reduced.loc[:, f'adj_close_price_day{day}_NVDQ'] = NVDQ_data_reduced['Adj Close'].shift(day)

    NVDA_data_reduced.dropna(inplace=True)
    NVDA_data_reduced = NVDA_data_reduced.reset_index(drop=True)
    NVDQ_data_reduced = NVDQ_data_reduced[1:]
    NVDQ_data_reduced = NVDQ_data_reduced.reset_index(drop=True)

    start_date = '2024-11-01'
    NVDA_test_data = NVDA_data_reduced[NVDA_data_reduced['Date'] >= pd.to_datetime(start_date)]
    NVDA_train_data = NVDA_data_reduced[NVDA_data_reduced['Date'] < pd.to_datetime(start_date)]
    NVDQ_test_data = NVDQ_data_reduced[NVDQ_data_reduced['Date'] >= pd.to_datetime(start_date)]
    NVDQ_train_data = NVDQ_data_reduced[NVDQ_data_reduced['Date'] < pd.to_datetime(start_date)]
    
    
    NVDA_data = NVDA_data.rename(columns= {'Date': 'Date', 'High': 'High_NVIDIA', 'Low':'Low_NVIDIA', 'Close':'Close_NVIDIA', 'Open':'Open_NVIDIA', 'Volume': 'Volume_NVIDIA', 'Adj Close': 'Adj Close_NVIDIA'})

    input_query_data_database = pd.concat([NVDA_data[['Date', 'High_NVIDIA', 'Low_NVIDIA', 'Close_NVIDIA', 'Open_NVIDIA', 'Volume_NVIDIA', 'Adj Close_NVIDIA']],
                                   NVDQ_data[['High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']]], axis=1)

    input_query_data_database.dropna(inplace=True)
    input_query_data_database = input_query_data_database.reset_index(drop=True)
    
    NVDA_train_data = NVDA_train_data.rename(columns= {'Date': 'Date_NVIDIA', 'High': 'High_NVIDIA', 'Low':'Low_NVIDIA', 'Close':'Close_NVIDIA', 'Open':'Open_NVIDIA', 'Volume': 'Volume_NVIDIA', 'Adj Close': 'Adj Close_NVIDIA'})
    NVDA_test_data = NVDA_test_data.rename(columns= {'Date': 'Date_NVIDIA', 'High': 'High_NVIDIA', 'Low':'Low_NVIDIA', 'Close':'Close_NVIDIA', 'Open':'Open_NVIDIA', 'Volume': 'Volume_NVIDIA', 'Adj Close': 'Adj Close_NVIDIA'})


    x_NVDA_train_data = pd.concat([NVDA_train_data[['Date_NVIDIA', 'High_NVIDIA', 'Low_NVIDIA', 'Close_NVIDIA', 'Open_NVIDIA', 'Volume_NVIDIA', 'Adj Close_NVIDIA']],
                               NVDQ_train_data[['High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']]], axis=1)

    #x_NVDA_train_data = pd.concat([NVDA_train_data[['Date', 'High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']],
    #                               NVDQ_train_data[['High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']]], axis=1)

    y_NVDA_train_data = NVDA_train_data[[
            'high_price_day1_NVDA', 
            'low_price_day1_NVDA', 
            'close_price_day1_NVDA',
            'open_price_day1_NVDA',
            'volume_price_day1_NVDA',
            'adj_close_price_day1_NVDA',
            'high_price_day1_NVDQ', 
            'low_price_day1_NVDQ', 
            'close_price_day1_NVDQ',
            'open_price_day1_NVDQ',
            'volume_price_day1_NVDQ',
            'adj_close_price_day1_NVDQ',
            ]]

    x_NVDA_test_data = pd.concat([NVDA_test_data[['Date_NVIDIA', 'High_NVIDIA', 'Low_NVIDIA', 'Close_NVIDIA', 'Open_NVIDIA', 'Volume_NVIDIA', 'Adj Close_NVIDIA']],
                               NVDQ_test_data[['High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']]], axis=1)

    #x_NVDA_test_data = pd.concat([NVDA_test_data[['Date', 'High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']],
                                  # NVDQ_test_data[['High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']]], axis=1)
    
    y_NVDA_test_data = NVDA_test_data[[
            'high_price_day1_NVDA', 
            'low_price_day1_NVDA', 
            'close_price_day1_NVDA',
            'open_price_day1_NVDA',
            'volume_price_day1_NVDA',
            'adj_close_price_day1_NVDA',
            'high_price_day1_NVDQ', 
            'low_price_day1_NVDQ', 
            'close_price_day1_NVDQ',
            'open_price_day1_NVDQ',
            'volume_price_day1_NVDQ',
            'adj_close_price_day1_NVDQ'
            ]]
    
    data = x_NVDA_train_data[['High_NVIDIA','Low_NVIDIA', 'Close_NVIDIA', 'Open_NVIDIA', 'High','Low',	'Close',	'Open']]
    
    def createTimeSeriesData(df, timesteps=1):
        columns = []
        for col in df.columns:
            for i in range(1, timesteps+1):
                columns.append(df[col].shift(i).rename(f"{col}_lag_{i}"))
        target = df.rename(columns=lambda col: f"{col}_target")
        ts_df = pd.concat(columns + [target], axis=1)
        ts_df.dropna(inplace=True)
        return ts_df
    
    timesteps = 10
    ts_data = createTimeSeriesData(data, timesteps)
    
    # Split the data into features and targets (the last 4 columns of the dataframe are the targets)
    X = ts_data.iloc[:, :-8].values  
    y = ts_data.iloc[:, -8:].values  

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define and train the MLP model (In this case the model has 4 outputs, Open, High, Low, Close)
    mlp_model = MLPRegressor(hidden_layer_sizes=(50,10), activation='relu', solver='adam', max_iter=1000)
    mlp_model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = mlp_model.predict(X_train)
    y_pred_test = mlp_model.predict(X_test)

    # Evaluate the model
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train, multioutput='raw_values'))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test, multioutput='raw_values'))
    
    latest_date_avaiable = input_query_data_database[input_query_data_database['Date'] == input_date]
    
    
    if latest_date_avaiable.empty:
      latest_input_query = input_query_data_database.sort_values(by='Date', ascending=False).head(timesteps+1) #replaced tail with head, as tails picked the first date instead of the latest, this method may have issues if user selects weekend
      latest_date_avaiable = latest_input_query['Date']
      latest_date_avaiable = pd.to_datetime(latest_date_avaiable).iloc[0]
      input_date = pd.to_datetime(input_date) 
    else:
      latest_input_query = input_query_data_database[input_query_data_database['Date'] <= input_date]
      latest_input_query = latest_input_query.sort_values(by='Date', ascending=False).head(timesteps+1)
      latest_date_avaiable = input_date
      latest_date_avaiable = pd.to_datetime(latest_date_avaiable)
      input_date = pd.to_datetime(input_date)

    latest_input_query = latest_input_query.copy()
    latest_input_query['Date'] = pd.to_datetime(latest_input_query['Date'])
    latest_input_query['Date'] = latest_input_query['Date'].astype('int64')

    #print(x_NVDA_test_data[0]) #comparing format
    #print(latest_input_query)
    
    print(f"input_date: {input_date}, latest_date_avaiable: {latest_date_avaiable}")

    latest_input_query_initial = latest_input_query[['High_NVIDIA',  'Low_NVIDIA',  'Close_NVIDIA', 'Open_NVIDIA','High', 'Low', 'Close', 'Open']]
    latest_input_query = createTimeSeriesData(latest_input_query_initial, timesteps)
    latest_input_query = latest_input_query.iloc[:, :-8].values  
    latest_input_query = pd.DataFrame(latest_input_query)
    
    if (input_date != latest_date_avaiable):
      while(latest_date_avaiable < input_date):
        latest_date_avaiable = latest_date_avaiable + timedelta(days=1)
        if latest_date_avaiable.weekday() == 5 or latest_date_avaiable.weekday() == 6: #skip saturday and sunday
            continue
        predictions = mlp_model.predict(latest_input_query)    
        latest_input_query_initial.iloc[0:10, :] = latest_input_query_initial.iloc[1:11, :]
        latest_input_query_initial.iloc[11:12, :] = predictions
        latest_input_query = createTimeSeriesData(latest_input_query_initial, timesteps)
        latest_input_query = latest_input_query.iloc[:, :-8].values 

    #print(latest_date_avaiable)

    results = np.zeros((6, 8))
    i = 0
    
    #print('latest_input_query', latest_input_query.shape)
    
    while(i < 6):
      latest_date_avaiable = latest_date_avaiable + timedelta(days=1)
      if latest_date_avaiable.weekday() == 5 or latest_date_avaiable.weekday() == 6: #skip saturday and sunday
          continue
      predictions = mlp_model.predict(latest_input_query)
    
      #predictions = best_model.predict(latest_input_query)
      results[i, :] = predictions
      latest_input_query_initial.iloc[0:10, :] = latest_input_query_initial.iloc[1:11, :]
      latest_input_query_initial.iloc[11:12, :] = predictions
      latest_input_query = createTimeSeriesData(latest_input_query_initial, timesteps)
      latest_input_query = latest_input_query.iloc[:, :-8].values  
      i += 1

    return results


@app.route('/', methods=['GET', 'POST'])
def process_user_request():

    if request.method == 'POST':
        user_input = request.form['date']
        returned_array, returned_actions = process_data(user_input)
        return render_template('front_end.html', user_input=user_input, result = returned_array, action = returned_actions)
    
    else:
        return render_template('front_end.html', user_input=None)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
