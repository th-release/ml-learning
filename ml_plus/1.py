import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([[1, 0], [1, 1], [2, 1], [1, 2], [3, 3]])
y_train = torch.FloatTensor([[1], [2], [3], [3], [6]])

# 가중치와 편향 선언
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros((1,), requires_grad=True)

# 경사 하강법
optimizer = optim.SGD([W, b], lr=1e-3)

num_epochs = 20000
for epoch in range(num_epochs + 1):
    y_pred = x_train.matmul(W) + b
    loss = torch.mean((y_pred - y_train) ** 2)

    optimizer.zero_grad()  # gradient 초기화
    loss.backward()  # 손실함수를 미분하여 gradient 계산
    optimizer.step()  # 업데이트

    if epoch % 1000 == 0:
        print('Epoch{:4d}/{} W: {}, b:{:.3f} loss:{:.6f}'.format(
            epoch, num_epochs, W.squeeze().tolist(), b.item(), loss.item()
        ))

x_result = torch.FloatTensor([[73, 80]])
y_result = x_result.matmul(W) + b.view(1)
print(y_result)
