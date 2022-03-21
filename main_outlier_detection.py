from models.outlier_detector import OutlierDetector
from models.imputer import Imputer

import warnings
warnings.filterwarnings('ignore')


class DataOutlier():
    def __init__(self, config, raw_data):
        """
        :param config: config 
        :type config: dictionary
        
        :param raw_data: train data whose shape is (num_index x num_variable)
        :type raw_data: dataframe
        
        example
            >>> config = { 
                    'algorithm': 'IF', # outlier detection에 활용할 알고리즘 정의, {'SR', 'LOF', 'MoG', 'KDE', 'IF'} 중 택 1
                    'imputation': 'KNN', # outlier를 대체(impute/replace)하기 위한 방법론 정의, {'KNN', 'Stats'} 중 택 1
                    
                    'alg_parameter': {
                        'percentile': 95, # 예측시 활용되는 outlier 임계값, int or float
                        'IF_estimators': 100, # ensemble에 활용하는 모델 개수, int(default: 100, 데이터 크기에 적합하게 설정)
                        'IF_max_samples': 'auto', # 각 모델에 사용하는 샘플 개수(샘플링 적용), int or float(default: 'auto')
                        'IF_contamination': 'auto', # 모델 학습시 활용되는 데이터의 outlier 비율, ‘auto’ or float(default: ’auto’, float인 경우 0 초과, 0.5 이하로 설정)
                        'IF_max_features': 1.0, # 각 모델에 사용하는 변수 개수(샘플링 적용), int or float(default: 1.0)
                        'IF_bootstrap': False}, # bootstrap적용 여부, bool(default: False)

                    'imp_parameter': {
                        'KNN_missing_values': np.nan, # 대체하기 위한 부분 구분자, int, float, str, np.nan or None(default: np.nan)
                        'KNN_neighbors': 5, # 대체에 참고하기 위한 이웃 개수, int(default: 5)
                        'KNN_weights': 'uniform', # 예측하는 과정에서 이웃에 부여할 가중치 여부, {‘uniform’, ‘distance’} or callable(default: ’uniform’)
                        'KNN_metric': 'nan_euclidean'} # 이웃을 정의하기 위한 거리 척도, {‘nan_euclidean’} or callable(default: ’nan_euclidean’)
                
                }
            
            >>> data_outlier = mod.DataOutlier(config, raw_data)
            >>> replaced_data, index_list = data_outlier.getResult()
        """

        self.algorithm = config['algorithm']
        self.imputation = config['imputation']
        self.alg_parameter = config['alg_parameter']
        self.imp_parameter = config['imp_parameter']
        self.raw_data = raw_data
        
    def getResult(self):
        """
        getResult by outlier detection algorithm and imputation method
        :return: replaced data, index_list
        :rtype: dataFrame, list
        """
        
        # Outlier Detection
        outlier_detector = OutlierDetector(self.raw_data, self.algorithm, self.imputation, self.alg_parameter)
        nan_data, index_list = outlier_detector.getResult()
            
        # Imputation
        imputer = Imputer(nan_data, self.imputation, self.imp_parameter)
        replaced_data = imputer.getResult()
        return replaced_data, index_list
