import matplotlib.pyplot as plt

#Draw Graphs of Training
path_txt = 'models/resnet_True/log_95.txt'
file = open(path_txt)
max_accuracy = -1
min_error = 9
num_epoch = 0
epoch = []
train_accuracy = []
test_accuracy = []
mean_loss = []
for line in file:
    if line[0] == '*':
        epoch.append(int(num_epoch))
        num_epoch += 1
    if line[1] == 'r':
        train_accuracy.append(float(line[16:-3]))
    if line[2] == 's':
        test_accuracy.append(float(line[15:-3]))
        if (float(line[15:-3])) > max_accuracy:
            max_accuracy = float(float(line[15:-3]))
            epoch_max_accuracy = num_epoch
    if line[0] == 'M':
        mean_loss.append(float(line[11:-2]))
        if float(line[11:-1]) < min_error:
            min_error = float(line[11:-2])
            epoch_min_error = num_epoch
file.close()
# print(epoch)
# print(train_accuracy)
# print(test_accuracy)
# print(mean_loss)
print('Max Test Accuracy: {0} on Epoch № {1}'.format(max_accuracy, epoch_max_accuracy))
print('Min Error: {0} on Epoch № {1}'.format(min_error, epoch_min_error))
leg1 = ['Train Acc.', 'Test Acc.']
leg2 = ['Mean Loss']

plt.figure(1)
plt.subplot(211)
plt.plot(epoch, train_accuracy)
plt.plot(epoch, test_accuracy)
plt.legend(leg1)
plt.ylabel('mAP')

plt.subplot(212)
plt.plot(epoch, mean_loss)
plt.legend(leg2)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()