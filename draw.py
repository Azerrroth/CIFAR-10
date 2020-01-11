import matplotlib.pyplot as plt
import pandas as pd
# rnnAns fcnnAns logisticAns cnnAns
data = pd.read_csv('./ans/rnnAns.csv')

print(data)

plt.figure(figsize=(20, 5))
plt.suptitle('RNN CIFAR-10 Training Status')
plt.subplot(1, 2, 1)
plt.ylim(0,0.005)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(len(data['loss'])), data['loss'])

plt.subplot(1, 2, 2)
plt.ylim(0, 100)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(range(len(data['loss'])), data['acc'])

plt.show()
