import matplotlib.pyplot as plt 
import pandas as pd

data = pd.read_csv('./ans/rnnAns.csv')

print(data)
plt.title('CNN CIFAR-10 Training status')
plt.plot(range(100) ,data['loss'])
plt.show()