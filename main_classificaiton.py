# from models.RNN import RNN_model
# from models.CNN import CNN_model

from train_model import Train_Test
import warnings
warnings.filterwarnings('ignore')


class Classification():
    def __init__(self, config):
        """
        :param config: config 
        :type config: dictionary
        
        example
                    # Case 1. w/o data representation & LSTM model 
                    config1 = {
                            'with_representation': False, # classification에 사용되는 representation이 있을 경우 True, 아닐 경우 False
                            'algorithm': 'LSTM', # classification에에 활용할 알고리즘 정의, {'RNN', 'LSTM', 'GRU', 'CONV_1D', 'FC_layer'} 중 택 1

                            'alg_parameter': {
                                'window_size' : 50, # input time series data를 windowing 하여 자르는 길이(size)
                                'num_layers' : 2, # recurrnet layers의 수, Default : 1
                                'hidden_size' : 64, # hidden state의 벡터차원 수
                                'attention' : False, # True일 경우 attention layer를 추가
                                'dropout' : 0.2, # If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
                                'bidirectional' : True, # 모델의 양방향성 여부
                                'batch_size' : 64 #batch size
                                'data_dir : './data'
                                'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available
                                'num_epochs' : 200 # 학습 시 사용할 epoch 수
                                }
                    }
            
        """
        self.config = config
        

    def getResult(self):
        """
        getResult by classification algorithm and data representation
        return: test set accuracy
        rtype: float
        """
        
        # Classification
        Classifier = Train_Test(self.config)
        test_accuracy = Classifier.get_accuracy()
        test_accuracy = test_accuracy.detach().cpu().numpy()


        return test_accuracy
