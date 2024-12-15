from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import datetime
import random #temp
import pickle
import os
from datetime import timedelta

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
    
    answer = np.array([[np.max(matrix[:, 0])], [np.min(matrix[:, 1])], [np.mean(matrix[:, 2])]])
    answer = np.round(answer, 2)
    #print(answer)
    
    names = np.array([["Highest NVDA Price"], ["Lowest NVDA Price"], ["Average NVDA Closing Price"]])
    organized_matrix = np.hstack((names, answer))
    
    #choose an action to take
    #NVDA_shares = 10000
    #NVDQ_shares = 100000

    final_money = 0
    shares = 'NONE'

    for i in range(len(matrix)):
       
       #find out which stock has the higher open price
       NVDA_open_price = matrix[i][3]
       NVDQ_open_price = matrix[i][9]
       #print("NVDA Open Price:",NVDA_open_price)
       #print("NVDQ Open Price:",NVDQ_open_price)

       
       if NVDA_open_price > NVDQ_open_price and shares != 'NVDA': #NVDA is higher + it's not alr in nvda
          decision_results[i] = "BULLISH"
          shares = 'NVDA'
       elif NVDA_open_price < NVDQ_open_price and shares != 'NVDQ': #NVDQ is higher + not already NVDQ
          decision_results[i] = "BEARISH"
          shares = 'NVDQ'
       else:
          decision_results[i] = "IDLE"
       

    #decision_results[:] = "IDLE"
    decision_results = np.hstack((date_results, decision_results))
    
    return organized_matrix, decision_results
    
def generate_data(input_date):

    with open('lr_model.pkl', 'rb') as file:
        lr_model = pickle.load(file)
        
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

    input_query_data_database = pd.concat([NVDA_data[['Date', 'High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']],
                                   NVDQ_data[['High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']]], axis=1)

    input_query_data_database.dropna(inplace=True)
    input_query_data_database = input_query_data_database.reset_index(drop=True)


    x_NVDA_train_data = pd.concat([NVDA_train_data[['Date', 'High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']],
                                   NVDQ_train_data[['High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']]], axis=1)

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

    x_NVDA_test_data = pd.concat([NVDA_test_data[['Date', 'High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']],
                                   NVDQ_test_data[['High', 'Low', 'Close', 'Open', 'Volume', 'Adj Close']]], axis=1)
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
    
    latest_date_avaiable = input_query_data_database[input_query_data_database['Date'] == input_date]
    if latest_date_avaiable.empty:
      latest_input_query = input_query_data_database.sort_values(by='Date', ascending=False).head(1) #replaced tail with head, as tails picked the first date instead of the latest, this method may have issues if user selects weekend
      latest_date_avaiable = latest_input_query['Date']
      latest_date_avaiable = pd.to_datetime(latest_date_avaiable).iloc[0]
      input_date = pd.to_datetime(input_date) 
    else:
      latest_input_query = input_query_data_database[input_query_data_database['Date'] == input_date]
      latest_date_avaiable = input_date
      latest_date_avaiable = pd.to_datetime(latest_date_avaiable)
      input_date = pd.to_datetime(input_date)

    latest_input_query = latest_input_query.copy()
    latest_input_query['Date'] = pd.to_datetime(latest_input_query['Date'])
    latest_input_query['Date'] = latest_input_query['Date'].astype('int64')

    #print(x_NVDA_test_data[0]) #comparing format
    #print(latest_input_query)

    if (input_date != latest_date_avaiable):
      while(latest_date_avaiable < input_date):
        latest_date_avaiable = latest_date_avaiable + timedelta(days=1)
        if latest_date_avaiable.weekday() == 5 or latest_date_avaiable.weekday() == 6: #skip saturday and sunday
            continue
        predictions = lr_model.predict(latest_input_query)
        #predictions = best_model.predict(latest_input_query)
        latest_input_query['Date'] = int(latest_date_avaiable.timestamp()) * 10**9
        latest_input_query.iloc[:, 1:] = predictions

    #print(latest_date_avaiable)
    #print(latest_input_query)

    results = np.zeros((5, 12))
    i = 0
    while(i < 5):
      latest_date_avaiable = latest_date_avaiable + timedelta(days=1)
      if latest_date_avaiable.weekday() == 5 or latest_date_avaiable.weekday() == 6: #skip saturday and sunday
          continue
      predictions = lr_model.predict(latest_input_query)

      
      #predictions = best_model.predict(latest_input_query)
      results[i, :] = predictions
      latest_input_query['Date'] = int(latest_date_avaiable.timestamp()) * 10**9
      latest_input_query.iloc[:, 1:] = predictions
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
