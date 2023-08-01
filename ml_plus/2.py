import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([[1, 0], [1, 1], [2, 1], [1, 2], [3, 3]])
y_train = torch.FloatTensor([[1], [2], [3], [3], [6]])

# 선형회귀 모델 선언
model = nn.Linear(2, 1)

# model.parameters() 로 파라미터를 쉽게 넣어줌
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1999  # 원하는만큼 경사 하강법을 반복
for epoch in range(num_epochs + 1):
    # model(x) 를 사용하여 연산
    y_pred = model(x_train)

    # 손실함수
    loss = F.mse_loss(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {}, b: {:.3f} loss: {:.6f}'.format(
            epoch, num_epochs, model.weight.data.squeeze().tolist(), model.bias.item(), loss.item()
        ))

x_result = torch.FloatTensor([[73, 80]])
result_y = model(x_result)
print('\nresult', result_y)
