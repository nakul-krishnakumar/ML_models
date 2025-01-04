import matplotlib.pyplot as plt

def plot_train_val_metrics(xlabel, xticks, accuracy_list_train, accuracy_list_val):
    plt.title('Train x Validation metrics')
    plt.xlabel(xlabel)
    plt.ylabel('accuracy')
    plt.xticks(ticks=range(len(xticks)), labels=xticks)
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(['Train', 'Validation'])
    plt.show()