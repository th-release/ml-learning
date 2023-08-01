# 간단한 식을 학습해서 식에 맞춰보는 ai

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.tensor([[1], [2], [3]])
y_train = torch.tensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs + 1):
    y_pred = x_train * W + b
    loss = torch.mean((y_pred - y_train) ** 2)

    optimizer.zero_grad() # gradient 초기화
    loss.backward() # 손실함수를 미분하여 gradient 계산
    optimizer.step() # 업데이트

    if epoch % 100 == 0:
        print('Epoch{:4d}/{} W: {:.3f}, b:{:.3f} loss:{:.6f}'.format(
            epoch, num_epochs, W.item(), b.item(), loss.item()
        ))

result_x = torch.tensor([[300.0]])
print('\nresult', result_x * W + b)
