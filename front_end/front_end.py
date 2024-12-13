from flask import Flask, render_template, request
import numpy as np
import datetime
import random #temp
import os

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
    matrix = np.random.randint(1, 101, size=(5, 3)) #TEMP REPLACE WITH THE MODEL(S) RETURN TABLE, elements 0-4 are for NVDA, elements 5-8 are for NVDQ
    print(matrix)
    
    answer = np.array([[np.max(matrix[:, 0])], [np.min(matrix[:, 1])], [np.mean(matrix[:, 2])]])
    print(answer)
    
    names = np.array([["Highest NVDA Price"], ["Lowest NVDA Price"], ["Average NVDA Price"]])
    organized_matrix = np.hstack((names, answer))
 
   # action_matrix = np.random.randint(1, 101, size=(5, 3)) #element 0 is for NVDA stock #, element 1 is for NVDQ stock #, element 2 is net value 
    
   # action_matrix[0][0] = 10000 #NVDA init
   # action_matrix[0][1] = 100000 #NVDQ init
    
    
    # for i in range(0,5):
        # #calculate net price of the day MODIFY MATRIX INDEX TO OPEN PRICE
        # action_matrix[i][2] = action_matrix[i][0] * matrix[i][0] + action_matrix[i][1] * matrix[i][4] 
        
        
        # if(matrix[i][0] == matrix[i][4]):
            # decision_results[i] = 'IDLE'
        # elif(matrix[i][0] > matrix[i][4]):
           # action_matrix[i][0] = 110000
           # action_matrix[i][1] = 0
           # if (i != 4):
               # action_matrix[i+1][0] = 110000
               # action_matrix[i+1][1] = 0
           # decision_results[i] = 'BULLISH'
        # else:
           # action_matrix[i][0] = 0
           # action_matrix[i][1] = 110000
           # if (i != 4):
               # action_matrix[i+1][0] = 0
               # action_matrix[i+1][1] = 110000
           # decision_results[i] = 'BEARISH'
        
        # if (i != 0):
            # if (action_matrix[i][0] == action_matrix[i-1][0]):
                # decision_results[i] = 'IDLE'
           
        # action_matrix[i][2] = action_matrix[i][0] * matrix[i][0] + action_matrix[i][1] * matrix[i][4]   #update after comparsion 
    
    # result_array = np.hstack((result_array, decision_results))   
    # result_array = np.hstack((result_array, action_matrix))    
    
    # print(action_matrix)
    # final_nvda_stock_shares = action_matrix[4][0]
    # final_nvdq_stock_shares = action_matrix[4][1]
    # final_price = action_matrix[4][0] * matrix[4][3] + action_matrix[4][1] * matrix[4][7] 
    
    decision_results[:] = "IDLE"
    decision_results = np.hstack((date_results, decision_results))
    
    return organized_matrix, decision_results
    
    
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
