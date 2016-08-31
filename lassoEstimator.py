from sklearn import cross_validation, linear_model
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

def exportDataFrame(filename):
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv(filename , header=0)

    #turn into list of ints
    #    df['Sequence'] = df['Sequence'].astype(np.int64)

    df['Sequence' ] = df['Sequence'].apply(ast.literal_eval)
    df              = df['Sequence'].apply(lambda row: pd.Series(row[:]))

    print(df.head(10))
    print(df.info())
    return df

def main():

    train_df = exportDataFrame('csv/train.csv')
    
    lasso = linear_model.Lasso()
    alphas = np.logspace(-4, -.5, 30)

    scores = []

    for alpha in alphas : 
#        print(alpha)
        lasso_alpha = copy.deepcopy(lasso)
        lasso_alpha.alpha = alpha
#        lasso_alpha.fit(train_X[]
    
        
#    scores.append(lasso_alpha.fit(X,y).score(X_test, y_test))
# plt.plot(alphas, scores)
# plt.xlabel('alpha  ')
# plt.ylabel('score  ')
# plt.grid(True)

# plt.show()



if __name__== '__main__':
    main()
