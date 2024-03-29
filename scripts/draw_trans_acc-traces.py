import matplotlib.pyplot as plt
import os

acc_files_path = "Metrics/TransTest_Resnet18/"

def draw_acc_traces_from_files():
    acc_files_name = os.listdir(acc_files_path)
    accs = []
    for file in acc_files_name:
        with open(acc_files_path + file, "r") as f:
            lines = f.readlines()
            accs.append([float(x) for x in lines])

    for i, acc in enumerate(accs):
        plt.plot(acc, label="transformation" + str(i))
    plt.xlabel("Traces")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

draw_acc_traces_from_files()
