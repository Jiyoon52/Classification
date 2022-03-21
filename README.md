# Outlier Detection

DataFrame 형태의 시계열 데이터를 입력으로 활용하는 Outlier detection에 대한 설명 <br><br>
* **실행 방법 : Test.ipynb 예시 참고** <br><br>
* **입력 데이터 형태 : T x P (P>=2) 차원의 다변량 시계열 데이터 (multivariate time-series data)** <br><br>
* **Outlier dection 사용 시, 설정해야하는 3가지 값**
- **1. 이상탐지 알고리즘 :**
  * SR (Spectral Residual)
  * MoG (Mixture of Gaussian) 
  * LOF (Local Outlier Factor) 
  * KDE (Kernel Density Estimation) 
  * IF (Isolation Forest)

- **2. 이상탐지 알고리즘 hyperparameter :** 아래에 자세히 설명.
  * SR (Spectral Residual) hyperparameter 
  * MoG (Mixture of Gaussian) hyperparameter 
  * LOF (Local Outlier Factor) hyperparameter 
  * KDE (Kernel Density Estimation) hyperparameter 
  * IF (Isolation Forest) hyperparameter

- **3. 이상치 imputation 여부 및 방법론(k-NN, statistical) :** 아래에 자세히 설명.
  * 이상치 **imputation 미선택 시** 최종 output : outlier index (row, column)
  * 이상치 **imputation 선택 시** 최종 output : imputation이 완료된 dataframe 
  * 현재는 (1)outlier index, (2)imputation 완성된 dataframe이 저장
<br>

---------------------------
## <br> 이상탐지 알고리즘 hyperparameter
#### 1. SR (Spectral Residual)
- **amp_window_size** : 원하는 window length
- **series_window_size** : 원하는 window length
- **score_window_size** : period보다 충분히 큰 size로 설정
<br>

#### 2. MoG (Mixture of Gaussian)
- **threshold** : 예측확률이 일정 값(threshold) 이하일 경우 이상치로 탐지하도록 설정
<br>

#### 3. LOF (Local Outlier Factor)
- **n_neighbors :** 밀도를 계산하고자 하는 주변 관측치 개수
- **algorithm :** ['auto', 'ball_tree', 'kd_tree', 'brute'], default='auto'
- **leaf_size :** Tree algorithm에서 leaf node의 개수, default=30
- **metric :** ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'minkowski'], default='minkowski'
<br>

#### 4. KDE (Kernel Density Estimation)
- **bandwidth :** 대역폭
- **algorithm :** ['kd_tree', 'ball_tree', 'auto'], default='auto'
- **kernel :** ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'], default='gaussian'
- **metric :** ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis'], default='euclidean'
- **breadth_first :** *boolean*, default=True
- **leaf_size :** Tree algorithm에서 leaf node의 개수, default=40
<br>

#### 5. IF (Isolation Forest)
- **n_estimators :** 원하는 기본 estimators 수, default=100
- **max_samples :** 하나의 estimator에 들어가는 sample 수(*int* or *float*), default='auto'
- **contamination :** 데이터 세트 내 이상치 개수 비율('auto' or *float*), default='auto'
- **max_features :** estimator의 최대 columns 수(*int* or *float*), default=1.0
- **bootstrap :** 데이터 중복(bootstrap)할 것인지 여부(*boolean*), default=False
<br>

---------------------------
## <br> 이상치 imputation 방법론

#### 1. k-NN (k-Nearest Neighbor)
- **missing_values** : imputaion 해야하는 값(*int*, *float*, *str*, *np.nan* or *None*), default=np.nan
- **n_neighbors** : k-NN 알고리즘 시 사용하는 neighbor 수(*int*), default=5
- **weights** : ['uniform', 'distance'], default='uniform'
- **metric** : ['nan_euclidean']
- **copy** : imputation 후 복사본을 생성할지 여부(*boolean*), default=True              
<br>

#### 2. statistical imputation 
- **missing_values** : imputaion 해야하는 값(*int*, *float*, *str*, *np.nan* or *None*), default=np.nan
- **strategy** : ['mean', 'median', 'most_frequent', 'constant'], default='mean'
- **copy** : imputation 후 복사본을 생성할지 여부(*boolean*), default=True              
<br>



# Time Series Classification
시계열 데이터 분류
<br><br><br>
## 1. Without data representation

- DataFrame 형태의 시계열 데이터를 입력으로 활용하는 time series classification에 대한 설명.
- 입력 데이터 형태 : TXP (P>=2) 차원의 다변량 시계열 데이터(multivariate time-series data)
<br>
<br>

**time series classification 사용 시, 설정해야하는 값**

* **시계열 분류 모델 :**
  * LSTM
  * GRU
  * 1D CNN 


* **시계열 분류 모델 hyperparameter :** 아래에 자세히 설명.
  * LSTM hyperparameter 
  * GRU hyperparameter 
  * 1D CNN  hyperparameter 
<br>

```c
python time series classification.py --model='lstm' \
                                     --attention=False \
                                     --hidden_size=20 \
                                     --num_layers=2 \
                                     --dropout=0.1 \
                                     --bidirectional=False \
```
<br><br>

#### 시계열 분류 모델 hyperparameter <br>

#### 1. LSTM & GRU
- **attention** : If True, adds an attention layer to RNN. Default: False
- **hidden_size** : The number of features in the hidden state h
- **num_layers** : The number of recurrent layers. E.g., setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
- **dropout** : If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0
- **bidirectional** : If True, becomes a bidirectional RNN. Default: False
- **bias** : If False, then the layer does not use bias weights b_ih and b_hh. Default: True
 
 
 #### 2. 1D CNN
- **num_layers** : Number of convolutional layers.
- **activation** : Type of activation functions to be used. Default : relu
- **dropout** : If non-zero, introduces a Dropout layer on the outputs of each CNN layer except the last layer, with dropout probability equal to dropout. Default: 0
- **batch_norm** : If True, applies Batch Normalization after CNN layers. Default: False
- **kernel_size** : Size of the convolving kernel
- **stride** : Stride of the convolution. Default: 1
- **padding** : Padding added to both sides of the input. Default: 0
- **dilation** : Spacing between kernel elements. Default: 1
- **bias** : If True, adds a learnable bias to the output. Default: True
 

<br><br>
## 2. With data representation
- 일정한 형식의 representation을 입력으로 활용하는 classification에 대한 설명.
- 입력 데이터 형태 : P (P>=2) 차원 벡터<br>


```c
python time series classification with data representation.py --model='fc' \
                                                              --num_layers=2 \
                                                              --activation=relu \
                                                              --dropout=0.2 \
                                                              --batch_norm=True
```
<br><br>


**time series classification 사용 시, 설정해야하는 값**

* **분류 모델 :**
  * FC layers (Fully Connected layers)



* **분류 모델 hyperparameter :** 아래에 자세히 설명.
  * FC layers (Fully Connected layers)


#### 분류 모델 hyperparameter <br>

#### 1. FC layers
- **num_layers** : The number of linear layers.
- **activation** : Type of activation functions to be used. Default : relu
- **dropout** : If non-zero, introduces a Dropout layer on the outputs of each CNN layer except the last layer, with dropout probability equal to dropout. Default: 0
- **batch_norm** : If True, applies Batch Normalization after CNN layers. Default: False
- **bias** : If True, adds a learnable bias to the output. Default: True
 




