import pickle
import matplotlib.pyplot as plt
import numpy as np


def unpickle(batch_path):
    with open(batch_path, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def unmingle(binary_data):
    return binary_data.reshape(binary_data.shape[0], 3, 32, 32).transpose([0, 2, 3, 1])


def get_img_with_label(label_name, batch, size):
    label_idx = meta[b'label_names'].index(label_name.encode())
    img_inx = [idx for idx, label_number in enumerate(batch[b'labels']) if label_number == label_idx]
    data = batch[b'data'][img_inx[:size]]
    return unmingle(data)


def matplt_histogram(im, ax):
    ax.hist(im.flatten(), bins=51, rwidth=.8)


def np_histogram(im):
    hist, bin_edges = np.histogram(im, bins=51)
    return hist


def euc_dist(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)


# --- Main ---
meta = unpickle('data/cifar-10-batches-py/batches.meta')
batch1 = unpickle('data/cifar-10-batches-py/data_batch_1')
test = unpickle('data/cifar-10-batches-py/test_batch')

automobiles = get_img_with_label('automobile', batch1, 30)
automobiles_test = get_img_with_label('automobile', test, 10)

deers = get_img_with_label('deer', batch1, 30)
deers_test = get_img_with_label('deer', test, 10)

ships = get_img_with_label('ship', batch1, 30)
ships_test = get_img_with_label('ship', test, 10)

histograms = [np_histogram(im) for batch in [automobiles, deers, ships] for im in batch]
histograms_test = [np_histogram(im) for batch in [automobiles_test, deers_test, ships_test] for im in batch]

fitted_labels = ['automobile' if idx < 30 else 'ship' if idx >= 60 else 'deer' for idx in
                 [np.argmin([euc_dist(test, hist) for hist in histograms]) for test in histograms_test]]
err = (np.repeat(['automobile', 'deer', 'ship'], 10) == fitted_labels).sum() / 30
print(f'Error: {err}%')

# --- Plot if necessary ---
# for imgs in [automobiles, deers, ships]:
#     fig, axs = plt.subplots(8, 8)
#     for i, ax in enumerate(axs.flat):
#         if i % 2 == 0 and i / 2 < len(imgs):
#             ax.imshow(imgs[int(i / 2)])
#         elif i / 2 < len(imgs):
#             matplt_histogram(imgs[int((i - 1) / 2)].mean(axis=2), ax)
#         else:
#             ax.set_visible(False)
#     print('set plotted')
# plt.show()
