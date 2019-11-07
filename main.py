import pickle
import matplotlib.pyplot as plt


def unpickle(batch_path):
    with open(batch_path, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def unmingle(binary_data):
    return binary_data.reshape(binary_data.shape[0], 3, 32, 32).transpose([0, 2, 3, 1])


def get_img_with_label(label_name):
    label_idx = meta[b'label_names'].index(label_name.encode())
    img_inx = [idx for idx, label_number in enumerate(batch1[b'labels']) if label_number == label_idx]
    data = batch1[b'data'][img_inx[:30]]
    return unmingle(data)


def matplt_histogram(im, ax):
    ax.hist(im.flatten(), bins=51, rwidth=.8)


meta = unpickle('data/cifar-10-batches-py/batches.meta')
batch1 = unpickle('data/cifar-10-batches-py/data_batch_1')

automobiles = get_img_with_label('automobile')
print('automobiles loaded')
deers = get_img_with_label('deer')
print('deers loaded')
ships = get_img_with_label('ship')
print('ships loaded')

# --- Plot if necessary ---
for imgs in [automobiles, deers, ships]:
    fig, axs = plt.subplots(8, 8)
    for i, ax in enumerate(axs.flat):
        if i % 2 == 0 and i / 2 < len(imgs):
            ax.imshow(imgs[int(i / 2)])
        elif i / 2 < len(imgs):
            matplt_histogram(imgs[int((i - 1) / 2)].mean(axis=2), ax)
        else:
            ax.set_visible(False)
    print('set plotted')
plt.show()
