import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.impute import KNNImputer


class Imputer():
    """
    :param nan_data: data where the values of detected outliers are NaN
    :type: dataFrame 

    :param method: method for data imputation which is selected in ['KNN', 'Stats']
    :type method: string
    
    :param args: config for data imputation method
    :type args: dictionary
    
    """
    def __init__(self, nan_data, method, args):
        self.method = method
        self.args = args
        
        self.data = nan_data.copy()
        self.columns_list = list(self.data.columns)
        
    def getResult(self):
        """
        :return replaced_data: data where the values of detected outliers are imputed using selected data imputation method
        :type: dataFrame
        
        """
        if self.method == 'KNN':
            replaced_data = self.KNN_imputer()
        elif self.method == 'Stats':
            replaced_data = self.Stats_imputer()
        return replaced_data
    
    def KNN_imputer(self):
        """
        :return replaced_data: data where the values of detected outliers are imputed using KNN method
        :type: dataFrame
        
        """
        replaced_data = pd.DataFrame()
        for col in tqdm(range(self.data.shape[1])):      
            data_col = self.data.iloc[:,col]
            data_col = data_col.values.reshape(-1,1)

            imputer = KNNImputer(missing_values=self.args['KNN_missing_values'], 
                                 n_neighbors=self.args['KNN_neighbors'],
                                 weights=self.args['KNN_weights'],
                                 metric=self.args['KNN_metric'])
            data_col = imputer.fit_transform(data_col)
            
            data_col = pd.DataFrame(data_col, columns=[self.columns_list[col]])       
            replaced_data = pd.concat([replaced_data, data_col], axis=1)
        return replaced_data

    def Stats_imputer(self):
        """
        :return replaced_data: data where the values of detected outliers are imputed using selected statistics
        :type: dataFrame
        
        """
        replaced_data = pd.DataFrame()
        for col in tqdm(range(self.data.shape[1])):      
            data_col = self.data.iloc[:,col]
            data_col = data_col.values.reshape(-1,1)

            if self.args['Stats_strategy'] == 'mean':
                num = np.mean(data_col)
            elif self.args['Stats_strategy'] == 'median':
                num = np.median(data_col)
            elif self.args['Stats_strategy'] == 'most_frequent':
                count = np.bincount(data_col)
                num = np.argmax(count)
            elif self.args['Stats_strategy'] == 'zero':
                num = 0
            elif self.args['Stats_strategy'] == 'one':
                num = 1

            data_col[np.isnan(data_col)] = num
            data_col = pd.DataFrame(data_col, columns=[self.columns_list[col]])       
            replaced_data = pd.concat([replaced_data, data_col], axis=1)
        return replaced_data
