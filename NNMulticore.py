import csv
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from math import floor
from random import shuffle
import time

x = []
y = []
indices_list = []

# Trains the neural network for a specific training and test set
def train(x, y, indices_list, i, q, avg_q):
    
    # Obtains input and outputs of training set and test set
    x_current_train, x_current_test = x[indices_list[i][0]], x[indices_list[i][1]]
    y_current_train, y_current_test = y[indices_list[i][0]], y[indices_list[i][1]]
    clf = MLPClassifier(solver='adam', alpha = 0.01, learning_rate_init=0.001, max_iter=200, hidden_layer_sizes=(50, 50))  # Creates a classifier 
    clf.fit(x_current_train, y_current_train) # Train/fit using current set
    current_accuracy = clf.score(x_current_test,y_current_test) # Obtain accuracy from the trained neural network
    
    # Write the accuracy to the 'q' queue
    q.put(current_accuracy)
    # Write the weighted accuracy to the 'avg_q' queue
    avg_q.put(current_accuracy*len(indices_list[i][0]))

# Reads the CSV file and store its data in variables named x and y
def readCSV(filename):
    
    global x, y
    print('Processing CSV file')
    
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        data = np.array(list(reader)).astype("float")
        np.random.shuffle(data)
    
        x = data[:, :-2] #input 
        y = data[:, -1] #output 
        
# Use KFold splitting method
def useKFold(k):
    global indices_list
    
    print('Starting KFold')
    kf = KFold(n_splits = k, random_state=None, shuffle=False) # Create sets
    indices_list = [(train_indices, test_indices) for train_indices, test_indices in kf.split(x)]

def main():
    # Read the CSV File
    readCSV('GrazynaOnlineAll.csv')
    # Use KFold splitting
    k = 10
    useKFold(k)
    # Start measuring time
    start_time = time.time()
    
    # Create FIFO queues for interprocess communication
    q = mp.Queue()
    avg_q = mp.Queue()
    
    # Processes list, accuracies list and weighted accuracies list
    procs, accuracies, weighted_accuracies = [], [], []

    # Define maximum number of simultaneous processes
    nprocs = 2
    
    # Parameters for process creation loop
    m = 0
    n = nprocs

    # Process creation loop
    for j in range(floor(k/nprocs) + 1):
        # Creates 'nprocs' simultaneous processes and start them
        for i in range(m, n):
            p = mp.Process(target=train, args=(x, y, indices_list, i, q, avg_q))
            procs.append(p)
            p.start()
            print('Training ', i, ' started.')

        # Communicate with the previously created processes to obtain the accuracy
        # calculated for the training
        for i in range(m, n):
            r = q.get()
            accuracies.append(r)
            s = avg_q.get()
            weighted_accuracies.append(s)
            print('Partial accuracy obtained from training: ', r*100, '%')
        
        # Join all processes
        for p in procs:
            p.join()
            
        # Empty the process list
        procs = []
        
        # Define new values for the parameters for process creation loop
        m = n 
        n = m + nprocs
        
        if (n > k):
            n = k
 
    # Finishes measuring time
    end_time = time.time()
    
    # Calculates weighted average accuracy
    avg_accuracy = 0
    avg_accuracy = sum(weighted_accuracies)/(sum([len(indices_list[i][0]) for i in range(0,k)]))
    
    print('Total time (in seconds): ', end_time - start_time)
    print('Training finished. Average accuracy: ', avg_accuracy*100, '%')

if __name__=='__main__':
    main()