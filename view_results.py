from matplotlib import pyplot as plt


def plot_cifar_ensemble():
    plt.bar([1, 2, 3, 4], [72.21, 76.07, 76.33, 81.57], width=.5, tick_label=[f"shallow\n{str(72.21)}%", f"medium\n{str(76.07)}%", f"deep\n{str(76.33)}%", f"ensemble\n{str(81.57)}%"])

    plt.ylim([0, 100])
    plt.xlabel("Models")
    plt.ylabel("% Accuracy on CIFAR Test Set")
    plt.title("Individual and Ensembled Model Performance - CIFAR")
    plt.show()


def plot_mnist_ensemble():
    plt.bar([1, 2, 3, 4], [99.23, 99.33, 99.18, 99.57], width=.5, tick_label=[f"shallow\n{str(99.23)}%", f"medium\n{str(99.33)}%", f"deep\n{str(99.18)}%", f"ensemble\n{str(99.57)}%"])

    plt.ylim([90, 100])
    plt.xlabel("Models")
    plt.ylabel("% Accuracy on MNIST Test Set")
    plt.title("Individual and Ensembled Model Performance - MNIST")
    plt.show()


plot_mnist_ensemble()