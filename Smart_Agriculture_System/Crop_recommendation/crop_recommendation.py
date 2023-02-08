import json
import joblib
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class MLP_Custom(torch.nn.Module):
  def __init__(self, inputs, outputs):
    super(MLP_Custom, self).__init__()
    self.linear0 = torch.nn.Linear(inputs, 64)
    self.linear1 = torch.nn.Linear(64, outputs)
  def forward(self, x):
    x = self.linear0(x)
    x = torch.nn.ReLU()(x)

    x = self.linear1(x)
    x = torch.nn.Softmax(dim=1)(x)

    return x
if __name__ == '__main__':
  idx_2_crop = None
  with open('./idx_2_crop.json') as user_file:
    idx_2_crop = json.load(user_file)
  scaler = joblib.load('./scaler.save')
  model = torch.load('./crop_best.pt')

  N = float(input("Input the ratio of Nitrogen content in soil (N): "))
  P = float(input('Input the ratio of Nitrogen content in soil (P): '))
  K = float(input('Input the ratio of Potassium content in soil (K):'))
  temperature = float(input('Input the temperature in degree Celsius: '))
  humidity = float(input("Input the relative humidity: "))
  pH = float(input("Input the value of the soil: "))
  rainfall = float(input('Input the rainfall in mm: '))

  inp = np.array([N, P, K, temperature, humidity, pH, rainfall]).reshape(1, -1)
  inp = scaler.transform(inp)
  inp = torch.tensor(inp, dtype=torch.float32)
  
  predict = model(inp)

  recommend = list(map(lambda x: [idx_2_crop[str(_)] for _ in x], predict.topk(5).indices.tolist()))[0]

  print('The recommendation crops include:')
  for _ in recommend:
    print (f"* {_}")
