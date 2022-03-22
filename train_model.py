import time
import copy
from models.RNN import RNN_model
from dataloader import create_classification_dataset
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class Train_Test():
    def __init__(self, config):
            self.algorithm = config['algorithm']
            self.alg_parameter = config['alg_parameter']


    def train(self, model, dataloaders, criterion, num_epochs, optimizer):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # 각 epoch마다 순서대로 training과 validation을 진행
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 training mode로 설정
                else:
                    model.eval()   # 모델을 validation mode로 설정

                running_loss = 0.0
                running_corrects = 0
                running_total = 0

                # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.alg_parameter['device'])
                    labels = labels.to(self.alg_parameter['device'], dtype=torch.long)

                    # parameter gradients를 0으로 설정
                    optimizer.zero_grad()

                    # forward
                    # training 단계에서만 gradient 업데이트 수행
                    with torch.set_grad_enabled(phase == 'train'):
                        # input을 model에 넣어 output을 도출한 후, loss를 계산함
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                        _, preds = torch.max(outputs, 1)

                        # backward (optimize): training 단계에서만 수행
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # batch별 loss를 축적함
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_total += labels.size(0)

                # epoch의 loss 및 accuracy 도출
                epoch_loss = running_loss / running_total
                epoch_acc = running_corrects.double() / running_total

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        # 전체 학습 시간 계산
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
        model.load_state_dict(best_model_wts)
        
        # best model 가중치 저장
        # torch.save(best_model_wts, '../output/best_model.pt')
        return model, val_acc_history


    def get_accuracy(self):
        if self.algorithm == 'LSTM':
            model = RNN_model(input_size = 561, hidden_size = self.alg_parameter['hidden_size'],
                              num_layers = self.alg_parameter['num_layers'], num_classes = 6, bidirectional = self.alg_parameter['bidirectional'], 
                              rnn_type='lstm')

        # if self.algorithm == 'GRU':
        #     model = RNN_model(input_size = 561, hidden_size = self.alg_parameter['hidden_size'],
        #                       num_layers = self.alg_parameter['num_layers'], num_classes = 6, bidirectional = self.alg_parameter['bidirectional'], 
        #                       rnn_type='gru')

        model = model.to(self.alg_parameter['device'])
        train_loader, valid_loader, test_loader = create_classification_dataset(self.alg_parameter['window_size'], self.alg_parameter['data_dir'], 
                                                                                self.alg_parameter['batch_size'])
        dataloaders_dict = {'train': train_loader, 'val': valid_loader}
        criterion = nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(), lr=0.0001)
        best_model, val_acc_history = self.train(model, dataloaders_dict, criterion, self.alg_parameter['num_epochs'], optimizer)
        test_accuracy = self.test(best_model, test_loader)

        return test_accuracy

    def test(self, model, test_loader):
        model.eval()   # 모델을 validation mode로 설정
        
        # test_loader에 대하여 검증 진행 (gradient update 방지)
        with torch.no_grad():
            corrects = 0
            total = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(self.alg_parameter['device'])
                labels = labels.to(self.alg_parameter['device'], dtype=torch.long)

                # forward
                # input을 model에 넣어 output을 도출
                outputs = model(inputs)

                # output 중 최댓값의 위치에 해당하는 class로 예측을 수행
                _, preds = torch.max(outputs, 1)

                # batch별 정답 개수를 축적함
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        # accuracy를 도출함
        test_acc = corrects.double() / total
        # print('Testing Acc: {:.4f}'.format(test_acc))       
        return test_acc


