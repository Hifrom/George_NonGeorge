import csv

# Train CSV
with open('train.csv', mode='w', newline='') as file:
    for i in range(1, 2146):
        data = [str(i) + '.jpg', '1']
        writer = csv.writer(file)
        writer.writerow(data)
    for i in range(2146, 4839):
        data = [str(i) + '.jpg', '0']
        writer = csv.writer(file)
        writer.writerow(data)

# Test CSV
with open('test.csv', mode='w', newline='') as file:
    for i in range(1, 537):
        data = [str(i) + '.jpg', '1']
        writer = csv.writer(file)
        writer.writerow(data)
    for i in range(537, 1210):
        data = [str(i) + '.jpg', '0']
        writer = csv.writer(file)
        writer.writerow(data)
