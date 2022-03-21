import numpy as np
import pandas as pd
from tqdm import tqdm

import sranodec as anom
from sklearn.ensemble import IsolationForest   
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from sklearn.mixture import GaussianMixture


class OutlierDetector():
    """
    :param raw_data: train data whose shape is (num_index x num_variable)
    :type raw_data: dataframe

    :param algorithm: method for outlier detection which is selected in ['SR', 'LOF', 'MoG', 'KDE', 'IF']
    :type algorithm: string
    
    :param imputation: method for data imputation which is selected in ['KNN', 'Stats']
    :type imputation: string
    
    :param args: config for outlier detection method
    :type args: dictionary
    
    """
    def __init__(self, raw_data, algorithm, imputation, args):
        self.algorithm = algorithm
        self.imputation = imputation
        self.args = args
        
        self.data = raw_data.copy()
        self.columns_list = list(self.data.columns)
        
    def getResult(self):
        """
        :return nan_data: data where the values of detected outliers are NaN
        :type: dataFrame 
       
        :return index_list: indices of detected outliers
        :type: list
        
        """
        nan_data = pd.DataFrame()
        col_index, row_index = [], []
        for col in tqdm(range(self.data.shape[1])):
            if self.algorithm == "SR":
                data_col = self.data.iloc[:, col].values
            else:
                data_col = self.data.iloc[:, col].values.reshape(-1, 1)

            model = self.getModel(data_col)
            indexes = self.getIndexList(data_col, model)
            
            if self.imputation == None or indexes == []:
                pass
            else:
                data_col[indexes] = np.nan

            data_col = pd.DataFrame(data_col, columns=[self.columns_list[col]])       
            nan_data = pd.concat([nan_data, data_col], axis=1)

            row_index.extend(indexes)
            col_index.extend([col] * len(indexes))

        index_list = [[idx, col] for idx, col in zip(row_index, col_index)]
        return nan_data, index_list
    
    def getModel(self, data_col):
        """
        :param data_col: data for each column
        :type: np.array
        
        :return model: fitted model of selected outlier detection algorithm
        :type: model
        
        """
        if self.algorithm == 'SR':
            model = anom.Silency(self.args['SR_spectral_window_size'], self.args['SR_series_window_size'],
                                 self.args['SR_score_window_size'])
        elif self.algorithm == 'LOF':
            model = LocalOutlierFactor(n_neighbors=self.args['LOF_neighbors'], novelty=True, 
                                       algorithm=self.args['LOF_algorithm'], leaf_size=self.args['LOF_leaf_size'], 
                                       metric=self.args['LOF_metric']).fit(data_col)
        elif self.algorithm == 'MoG':
            model =  GaussianMixture(n_components=self.args['MoG_components'], covariance_type=self.args['MoG_covariance'], 
                                     max_iter=self.args['MoG_max_iter'], random_state=0).fit(data_col)
        elif self.algorithm == 'KDE':
            model = KernelDensity(kernel=self.args['KDE_kernel'], bandwidth=self.args['KDE_bandwidth'], 
                                  algorithm=self.args['KDE_algorithm'], metric=self.args['KDE_metric'], 
                                  breadth_first=self.args['KDE_breadth_first'], 
                                  leaf_size=self.args['KDE_leaf_size']).fit(data_col)
        elif self.algorithm == 'IF':
            model = IsolationForest(n_estimators=self.args['IF_estimators'], max_samples=self.args['IF_max_samples'], 
                                    contamination=self.args['IF_contamination'], max_features=self.args['IF_max_features'], 
                                    bootstrap=self.args['IF_bootstrap']).fit(data_col)
        return model
    
    def getIndexList(self, data_col, model):
        """
        :param data_col: data for each column
        :type: np.array
        
        :param model: fitted model of selected outlier detection algorithm
        :type: model
        
        :return indexes: indices of detected outliers
        :type: list
        
        """
        if self.algorithm == 'SR':
            score = model.generate_anomaly_score(data_col)
        elif self.algorithm == 'MoG':
            score = - 1.0* model.predict_proba(data_col)
        else:
            score = - 1.0 * model.score_samples(data_col)
        
        if self.algorithm == 'MoG':
            indexes = np.where(score[:, 0] > np.percentile(score[:, 0], self.args['percentile']))[0]
        else:
            indexes = np.where(score > np.percentile(score, self.args['percentile']))[0]
        return indexes

