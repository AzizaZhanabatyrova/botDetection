#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:31:57 2017
@author: Clayton and Aziza
"""

import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
import time
from math import log10

start = time.time()
print('\n===== Processing CSV file =====\n')

with open('GrazynaOnlineAll.csv', 'r') as csvfile:
    # Read the dataset contained in CSV file
    reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
    data = np.array(list(reader)).astype("float")
    # Shuffle dataset
    shuffled_data = np.copy(data)
    np.random.shuffle(shuffled_data)

    x = data[:, :-2] # Input 
    y = data[:, -1] # Output 
    
    shuffled_x = shuffled_data[:, :-2] # Shuffled input 
    shuffled_y = shuffled_data[:, -1] # Shuffled output 
    
    print('===== Building and training neural network =====\n')
    # Define parameters of the neural network
    clf = MLPClassifier(solver='adam', alpha=1e-3, learning_rate_init=0.001, max_iter=200, hidden_layer_sizes=(10, 10))
    print(clf)
    # Train the neural network with defined inputs and outputs
    clf.fit(shuffled_x, shuffled_y) 
    # Calculate accuracy of the neura network
    accuracy = clf.score(shuffled_x, shuffled_y)
    print("\nAccuracy of neural network: .....", accuracy*100, "%\n")
    
    # Obtain the outputs of the neural network for a determined input
    predicted_y = clf.predict_proba(x)  
    
    m = data[:, -2] # Session ID's
    count = h = t = false_pos = false_neg = undecided = decided_on_req = divide = positive = negative = 0
    session = []
    
    # Thresholds 
    threshold1 = 10
    threshold2 = -10
    
    # Requests to accumulate before starting to decide something
    n_req_acc = 1
   
    print('===== Applying sequential classification =====\n')
    
    for a in range(13587): # for every session
        session.append([])
        
        for i in range(h,len(m)): # calculate requests in a session
            if m[i] == (a+1):
                count += 1
            elif m[i] > (a+1):
                break
        h = i
        
        # sum_of_log1 contains the sum of the logarithms of the output
        # sum_of_log2 contains the sum of the logarithms of 1 - output
        sum_of_log1 = sum_of_log2 = 0
        
        # predicted_y[] has two values, first one is the probability of NOT being a bot
        # and the second one is the probability of being a bot. It is enough to use the 
        # second value.
        
        for b in range(count): # for every request
            session[a].append([])
            session[a][b].append(predicted_y[t][1])
            session[a][b].append(predicted_y[t][0])
            log1 = log10(predicted_y[t][1])
            session[a][b].append(log1)
            log2 = log10(predicted_y[t][0])
            session[a][b].append(log2)
            ratio = (log1 + sum_of_log1) - (log2 + sum_of_log2) # Ratio score
            session[a][b].append(ratio)
            
            if (b >= n_req_acc - 1): # Decides things at least after some requests
                if (ratio > threshold1) : # Decides it is a bot
                    session[a][b].append(1)
                    decided_on_req += b + 1
                    divide += 1
                    positive += 1
                    if (y[t] == 0): # In fact, it is not a bot
                       false_pos += 1
                       #print('False positive on: ', a)
                    t = h
                    break
                elif (ratio < threshold2): # Decides it is not a bot
                    session[a][b].append(0)
                    decided_on_req += b + 1
                    divide += 1
                    negative += 1
                    if (y[t] == 1): # In fact, it is a bot
                       false_neg += 1
                    t = h
                    break
                else: # Doesn't decide anything
                    session[a][b].append(-1)
                    if (b == count - 1):
                        undecided += 1
            else: # Doesn't decide anything
                session[a][b].append(-1)
                if (b == count - 1):
                        undecided += 1
                

            sum_of_log1 += log1
            sum_of_log2 += log2
            
            t += 1
            
        count = 0 
        
print('Thresholds: ................................', threshold1, ',', threshold2)

print('False positives: ...........................', false_pos)
print('Total positives: ...........................', positive)
print('Correct positive rate: .....................', (positive - false_pos)*100/positive, '%')

print('False negatives: ...........................', false_neg)
print('Total negatives: ...........................', negative)
print('Correct negative rate: .....................', (negative - false_neg)*100/negative, '%')

print('Total errors: ..............................', false_neg + false_pos)

print('Undecided sessions: ........................', undecided)
print('Undecided rate: ............................', undecided*100/13587, '%')
print('Wrong decision rate: .......................', (false_neg + false_pos)*100/13587, '%') 
print('Correct decision rate: .....................', (13587 - undecided - false_neg - false_pos)*100/(13587 - undecided),'%')
print('Average number of requests needed to decide:', decided_on_req/divide)
print('Number of requests to accumulate:...........', n_req_acc)
end = time.time()
print("Total Time: ................................", end-start)  