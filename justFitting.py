#from sklearn import cross_validation, linear_model
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def getLastValidValue(row):
    lastValidIndex = row.last_valid_index()
    return row.iloc(lastValidIndex)

def exportDataFrame(filename, df):
    # For .read_csv, always use header=0 when you know row 0 is the header row
    print('Reading csv file : ' , filename)
              
    df = pd.read_csv(filename, index_col="Id")
    train_seqs = df['Sequence'].to_dict()
    
    for key in train_seqs:
        seq = train_seqs[key]
        seq = [int(x) for x in seq.split(',')]
        train_seqs[key] = seq

    return train_seqs
        
def main():

    df = None
    train_dict = exportDataFrame('csv/train.csv', df)
    test_dict  = exportDataFrame('csv/test.csv' , df)

    #print(train_dict)
    #print(test_dict)
    
    
    degrees = range(1,11)
    scores = []

    counter = 0.
    closeCounter = 0.

    for seqId, sequence in train_dict.items():
        for degree in degrees :
            print('seqId  : ', seqId )
            print('degree : ', degree )
            model = make_pipeline(PolynomialFeatures(degree), Ridge())
            array = np.array(range(0,len(sequence) - 1))
            re = array.reshape(-1, 1)

            model.fit( re, sequence[:-1])
            y_plot = model.predict(len(sequence))
            print('y_plot ' , y_plot)
            print('next sequence value ' , sequence[-1])
            print('diff in value : ' , y_plot - sequence[-1])
            if( y_plot - sequence[-1] < .5 ) : closeCounter == closeCounter + 1
            
        counter = counter + 1
        if counter > 25 : break

    print('score on close counter : ' , closeCounter/counter)
        
#    scores.append(lasso_alpha.fit(X,y).score(X_test, y_test))
# plt.plot(alphas, scores)
# plt.xlabel('alpha  ')
# plt.ylabel('score  ')
# plt.grid(True)

# plt.show()



if __name__== '__main__':
    main()
