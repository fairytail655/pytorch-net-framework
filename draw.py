from matplotlib import pyplot as plt

def draw(epochs, train_losses, train_prec, val_prec):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    axs.set_title('Train and Validata')
    axs.set_xlabel('epochs')
    axs.set_ylabel('loss')
    axs.plot(epochs, train_losses, label='train_losses')
    axs.plot(epochs, train_prec, label='train_prec')
    axs.plot(epochs, val_prec, label='val_prec')
    axs.legend()
    plt.show()

def draw2(epochs, train_losses, val_losses, train_prec, val_prec):
    fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    axs[0].set_title('Loss of Train and Validata')
    # axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')
    axs[0].plot(epochs, train_losses, label='train')
    axs[0].plot(epochs, val_losses, label='validate')
    axs[0].legend()

    axs[1].set_title('Precision of Train and Validata')
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('prec')
    axs[1].plot(epochs, train_prec, label='train')
    axs[1].plot(epochs, val_prec, label='validate')
    axs[1].legend()

    plt.show()
