import matplotlib.pyplot as plt

x = np.arange(1, epochs+1, 1)
y1 = lst_training_loss
#y2 = lst_val_loss
y3 = lst_accuracy

z1 = lst_training_loss_lstm
z2 = lst_accuracy_lstm
 
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, label = "$TCN training loss$", color = "r")
ax1.plot(x, z1, label = "$LSTM training loss$", color = "k")

ax1.set_ylabel("Loss")
ax1.set_xlabel("Epoch")

ax = plt.gca()
ax.locator_params("x", nbins = 20)
 
ax2 = plt.twinx()
ax2.set_ylabel("Accuracy")
ax2.plot(x, y3, label = "$TCN validation accuracy$")
ax2.plot(x, z2, label = "$LSTM validation accuracy$")
ax2.legend(bbox_to_anchor=(1.1, 0.8), loc='upper left')
ax1.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left')

plt.show()
