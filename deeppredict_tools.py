import errno
import os
from enum import Enum
from os import path
from typing import List

SCRIPT_DIR = path.dirname(path.realpath(__file__))


def try_make_directories():
    data_directories = ["npy_dataset"]
    for directory in data_directories:
        try_make_dir(directory)


def try_make_dir(name):
    try:
        os.makedirs(name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def time_signal(file, code):
    """
    load the time series in the file and convert units in SI
    file= 'filename' from the ./dataset/ folder
    code= ' DE' or 'FE', other code will give error
    """
    import re
    try:
        import scipy.io as sio
        data = sio.loadmat(f"{SCRIPT_DIR}/dataset/{file}.mat")
        for key in data:
            if re.findall(rf"({code})", key):
                dict_code = key
        unit_conv = 0.0254
        x = data[dict_code] * unit_conv
        return x
    except:
        print('File does not exist or code wrong. Code must be either "DE or "FE" ')


def spectrogram_to_image(file, code, fs=12000., *args):
    """
    Plot and save the spectrogram of the mat file
    file = 'filename' from the ./dataset/ folder
    code = ' DE' or 'FE', other code will give error
    """
    from scipy import signal
    import matplotlib.pyplot as plt
    import numpy as np

    x = time_signal(file=file, code=code).flatten()
    f, t, sxx = signal.spectrogram(x, fs, window=signal.get_window('hamming', 1024),
                                   nperseg=1024, noverlap=100, scaling='density', mode='magnitude')
    fig = plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(sxx), shading='gouraud')
    plt.ylabel('Power/Frequency $(dB/Hz)$')
    plt.xlabel('time $(s)$')
    plt.show()
    fig.savefig('./spectrogram/stft_' + file + '_' + code + '.png')


def scalogram_to_image(file, code, fs=12000.):
    """
    Plot and save the scalogram of the mat file
    file = 'filename' from the ./dataset/ folder
    code = ' DE' or 'FE', other code will give error
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import pywt

    x = time_signal(file=file, code=code).flatten()[0:512]
    widths = np.arange(1, 128)
    scal, freq = pywt.cwt(x, widths, 'morl', sampling_period=1. / fs)
    fig = plt.figure()
    plt.imshow(scal, cmap='ocean', aspect='auto', interpolation='bicubic', norm=mpl.colors.Normalize())

    plt.ylabel('scale')
    plt.xlabel('time $(s)$')
    plt.show()
    fig.savefig('./wavelet/wv_' + file + '_' + code + '.png')


def qtransform_to_image(file, code, fs=12000., *args):
    """
    Plot and save the qtransform of the mat file
    file = 'filename' from the ./dataset/ folder
    code = ' DE' or 'FE', other code will give error
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa

    sr = fs
    x = time_signal(file=file, code=code).flatten()
    qtransform = np.abs(librosa.cqt(x, sr=sr, fmin=librosa.note_to_hz('C3'), n_bins=60 * 2, bins_per_octave=12 * 2))

    fig = plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(qtransform, ref=np.max),
                             sr=sr, x_axis='time', y_axis='hz', shading='gourand', cmap='ocean')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    fig.savefig('./qtransform/qt_' + file + '_' + code + '.png')


def split_exact(x, n_chunks=2, axis=1):
    import numpy as np
    l = np.shape(x)[axis]
    x_split = x
    if l > n_chunks > 1:
        n = n_chunks
        if axis == 0:
            x_split = np.split(x[:-(l % n)], n, axis=axis)
        elif axis == 1:
            x_split = np.split(x[:, :-(l % n)], n, axis=axis)
    return x_split


from typing import Tuple
from numpy.core.multiarray import ndarray
import numpy as np


def load_raw_matlab_data(chunk_size=128) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Load the mat files and splits it in chunks of size chunk_size
    returns:
        array_normal, array_ir, array_b, array_or
    """
    y_n: ndarray = np.array([])
    # normal signal
    for i in (0, 1, 2, 3):
        x = time_signal(file='Normal_' + str(i), code='DE').flatten()
        x = x / np.std(x)
        y_n = np.append(y_n, x)
    n_splits = len(y_n) // chunk_size
    y_n = split_exact(y_n, n_chunks=n_splits, axis=0)

    y_ir: ndarray = np.array([])
    for i in ('007', '014', '021', '028'):
        for j in (0, 1, 2, 3):
            x = time_signal(file='IR' + i + '_' + str(j), code='DE').flatten()
            x = x / np.std(x)
            y_ir = np.append(y_ir, x)
    n_splits = len(y_ir) // chunk_size
    y_ir = split_exact(y_ir, n_chunks=n_splits, axis=0)

    y_b: ndarray = np.array([])
    for i in ('007', '014', '021', '028'):
        for j in (0, 1, 2, 3):
            x = time_signal(file='B' + i + '_' + str(j), code='DE').flatten()
            x = x / np.std(x)
            y_b = np.append(y_b, x)
    n_splits = len(y_b) // chunk_size
    y_b = split_exact(y_b, n_chunks=n_splits, axis=0)

    return y_n, y_ir, y_b


def min_max_norm(ary):
    import numpy as np
    ary = (ary - ary.min()) / np.abs(ary.max() - ary.min())
    return ary


def save_figure(fig, plt, file_label='n', image='w'):
    import matplotlib as mpl
    import uuid

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(mpl.ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(mpl.ticker.NullLocator())
    fig.savefig(f'./image_dataset/{file_label}_{image}_{str(uuid.uuid1())}.png')


def npy_save(ary, file_label='n', image='w'):
    import numpy as np
    import uuid

    file_name = str('./npy_dataset/' + file_label + '_' + image + '_' + str(uuid.uuid1()))
    np.save(file_name, ary)


def generate_scalogram_image_with_label(data_y_vector: ndarray, image_shape: tuple, label: str):
    return generate_scalogram_image(data_y_vector, image_shape), label


def generate_scalogram_image(data_y_vector: ndarray, image_shape: tuple):
    """
    Calculate the scalogram image of an array
    """
    import numpy as np
    import pywt
    from skimage.transform import resize

    fs: float = 12000.
    wavelet: str = 'morl'
    coefficients: int = len(data_y_vector) + 1

    widths = np.arange(1, coefficients)

    scal, freq = pywt.cwt(data_y_vector, widths, wavelet, sampling_period=1. / fs)
    scal = min_max_norm(scal)
    scal = resize(scal, image_shape, mode='constant', anti_aliasing=True)

    return scal


def generate_spectrogram_image_with_label(data_y_vector: ndarray, image_shape: tuple, label: str):
    return generate_scalogram_image(data_y_vector, image_shape), label


def generate_spectrogram_image(data_y_vector: ndarray, image_shape: tuple):
    """
    Calculate the spectrogram of an array ary
    """
    from scipy import signal
    from skimage.transform import resize

    fs = 12000.
    data_y_vector_len = np.shape(data_y_vector)[0]
    hamming_window = int(data_y_vector_len / 5)

    f, t, sxx = signal.spectrogram(
        data_y_vector,
        fs)

    sxx = min_max_norm(sxx)
    sxx = resize(sxx, image_shape, mode='constant', anti_aliasing=True)

    # render_spectrogram_image(sxx, 512, label, figure_save=True)

    return sxx


def generate_qtransform_image_with_label(data_y_vector, image_shape: tuple, label: str):
    return generate_qtransform_image(data_y_vector, image_shape), label


def generate_qtransform_image(data_y_vector, image_shape: tuple):
    """
    Calculate the qtransform of an array ary
    """
    import librosa
    import numpy as np

    from skimage.transform import resize

    fs = 12000.
    coefficients: int = len(data_y_vector) + 1

    qtransform = np.abs(librosa.cqt(data_y_vector, sr=fs))
    qtransform = min_max_norm(qtransform)
    qtransform = resize(qtransform, image_shape, mode='constant', anti_aliasing=True)

    return qtransform


def render_qtransform_image(data_to_render, coefficients, y_vector_len, file_label, figure_save=False, ary_save=False):
    import matplotlib.pyplot as plt

    ary_len = float(y_vector_len)
    dpi = min(data_to_render.shape[0], data_to_render.shape[1])

    if figure_save:
        fig = plt.figure(figsize=(1, 1), dpi=dpi)
        plt.imshow(generate_qtransform_image_with_label, interpolation='bicubic', cmap='hot',
                   extent=[0, coefficients, 0, ary_len],
                   aspect=coefficients / ary_len)
        save_figure(fig, plt, file_label=file_label, image='w')

    if ary_save:
        npy_save(generate_qtransform_image_with_label, file_label=file_label, image='q')


def render_spectrogram_image(data_to_render, coefficients, file_label, figure_save=False, ary_save=False):
    import numpy as np
    import matplotlib.pyplot as plt

    dpi = min(data_to_render.shape[0], data_to_render.shape[1])

    dS = 10 ** -8  # needed for plotting in logscale
    ary_len = float(len(data_to_render))

    if figure_save:
        fig = plt.figure(figsize=(1, 1), dpi=dpi)
        plt.imshow(10 * np.log10(data_to_render + dS), interpolation='bicubic', extent=[0, coefficients, 0, ary_len],
                   aspect=coefficients / ary_len)
        save_figure(fig, plt, file_label=file_label, image='w')

    if ary_save:
        npy_save(data_to_render, file_label=file_label, image='s')


def load_image_dataset():
    import glob
    import numpy as np
    import re
    import matplotlib.pyplot as plt

    labels = []

    prototyte_file = plt.imread(glob.glob(f'{SCRIPT_DIR}/image_dataset/*.png')[0])

    ary = np.expand_dims(np.empty_like(prototyte_file), axis=0)

    for code in ('n_', 'ir_', 'b_'):
        for file in glob.glob(f'{SCRIPT_DIR}/image_dataset/{code}*.png'):
            labels.append(re.findall(f'{code}', file))
            tmp_ary = np.expand_dims(plt.imread(file), axis=0)
            ary = np.append(ary, tmp_ary, axis=0)

    labels = [item for item in labels]
    ary = np.delete(ary, 0, axis=0)

    return ary, labels


class ImageType(Enum):
    SCALOGRAM = 1
    SPECTROGRAM = 2
    QTRANSFORM = 3


def load_images_by_labels_by_params(
        number_of_chunks: int,
        chunk_size: int,
        image_types: List[ImageType],
        image_shape: tuple,
        noise_scale: float = 0.0,
        chunks_are_shuffled_across_time=True):
    filename = npy_dataset_filename(
        number_of_chunks,
        chunk_size,
        image_types,
        image_shape,
        noise_scale,
        chunks_are_shuffled_across_time)

    loaded = np.load(filename)
    print(f"Successfully loaded file {filename}")
    return loaded.item()  # item() gets the actual dictionary from the loaded object


def npy_dataset_filename(
        number_of_chunks: int,
        chunk_size: int,
        image_types: List[ImageType],
        image_shape: tuple,
        noise_scale: float,
        chunks_are_shuffled_across_time):
    filename_suffix = \
        f'__chunks{number_of_chunks}' \
        f'_size{chunk_size}' \
        f'_types{"-".join([str(t)[10:12] for t in image_types])}' + \
        (f'_noise{noise_scale}' if noise_scale > 0 else '') + \
        (f'_timeshuffled' if chunks_are_shuffled_across_time else '') + \
        f'_shape{image_shape[0]}-{image_shape[1]}'
    filename = f'{SCRIPT_DIR}/npy_dataset/images_by_labels_for_cnn{filename_suffix}.npy'
    return filename


def load_data_and_labels_for_cnn_by_params(
        number_of_chunks: int,
        chunk_size: int,
        image_types: List[ImageType],
        image_shape: tuple):
    filename_suffix = \
        f'__chunks{number_of_chunks}' \
        f'_size{chunk_size}' \
        f'_types{"-".join([str(t)[10:12] for t in image_types])}' \
        f'_shape{image_shape[0]}-{image_shape[1]}'

    return load_data_and_labels_for_cnn(
        data_file=f'npy_dataset/images_for_cnn{filename_suffix}.npy',
        labels_file=f'npy_dataset/labels_for_cnn{filename_suffix}.npy')


def load_data_and_labels_for_cnn(data_file: str, labels_file: str):
    import numpy as np

    labels = np.load(f'{SCRIPT_DIR}/{labels_file}')
    data = np.load(f'{SCRIPT_DIR}/{data_file}')

    return data, labels


def ROC(model, X_test, y_test, n_class):
    from sklearn.metrics import roc_curve, roc_auc_score, auc
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt

    # Predict the labels of the test set: y_pred
    y_pred = model.predict_classes(X_test)

    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = model.predict_proba(X_test)
    y_test_binary = label_binarize(y_test, classes=range(0, n_class))

    y_test_normal = y_test_binary[:, 0]
    y_prob_normal = y_pred_prob[:, 0]

    y_test_inner_race = y_test_binary[:, 1]
    y_prob_inner_race = y_pred_prob[:, 1]

    y_test_ball = y_test_binary[:, 2]
    y_prob_ball = y_pred_prob[:, 2]

    # Compute and print the confusion matrix and classification report
    confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test_normal, y_prob_normal)

    # Plot ROC curve
    plt.figure(figsize=(4, 4), dpi=150)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve, Normal state')
    plt.show()

    print('The area under the ROC curve is {}'.format(roc_auc_score(y_test_normal, y_pred_prob[:, 0])))

    from itertools import cycle
    # Compute ROC curve and ROC area for each class
    lw = 2

    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()

    fpr[0], tpr[0], thresholds[0] = roc_curve(y_test_normal, y_prob_normal)
    fpr[1], tpr[1], thresholds[1] = roc_curve(y_test_inner_race, y_prob_inner_race)
    fpr[2], tpr[2], thresholds[2] = roc_curve(y_test_ball, y_prob_ball)

    for i in range(3):
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(4, 4), dpi=150)

    colors = cycle(['darkgray', 'darkorange', 'darkblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def append_images(images_with_label_to_append: List[ndarray], images: ndarray, labels: List):
    import numpy as np

    for image_with_label_to_append in images_with_label_to_append:
        image_to_append = image_with_label_to_append[0]
        images = np.append(images, np.expand_dims(image_to_append, axis=0), axis=0)
        label_to_append = image_with_label_to_append[1]
        labels.append(label_to_append)

    return images, labels


def generate_images_and_append(
        vector_y_data: ndarray,
        image_shape: tuple,
        image_types: List[ImageType],
        label: str,
        append_to_images,
        append_to_labels):
    images_with_label_to_append = []
    if ImageType.SCALOGRAM in image_types:
        images_with_label_to_append.append(generate_scalogram_image_with_label(vector_y_data, image_shape, label=label))
    if ImageType.SPECTROGRAM in image_types:
        images_with_label_to_append.append(
            generate_spectrogram_image_with_label(vector_y_data, image_shape, label=label))
    if ImageType.QTRANSFORM in image_types:
        images_with_label_to_append.append(
            generate_qtransform_image_with_label(vector_y_data, image_shape, label=label))

    append_to_images, append_to_labels = append_images(
        images_with_label_to_append=images_with_label_to_append,
        images=append_to_images,
        labels=append_to_labels)

    return append_to_images, append_to_labels


def ary_to_rgba(ary, nchannels=4):
    from matplotlib import cm

    ary_dim = (np.shape(ary))

    im = np.empty(shape=(ary_dim[0], ary_dim[1], ary_dim[2], nchannels)).astype('uint8')
    for i in np.arange(0, len(ary)):
        im[i, :] = cm.ScalarMappable(cmap='viridis').to_rgba(10 * np.log(ary[i] + 0.001), bytes=True, norm=True)
        # im[i, :] = cm.ScalarMappable(cmap='jet').to_rgba(10 * np.log(ary[i] + 0.001), bytes=True, norm=True)
    return im


def generate_rgba_images_from_data(
        vector_y_data: ndarray,
        image_shape: tuple,
        image_types: List[ImageType]):
    images = []
    if ImageType.SCALOGRAM in image_types:
        images.append(generate_scalogram_image(vector_y_data, image_shape))
    if ImageType.SPECTROGRAM in image_types:
        images.append(generate_spectrogram_image(vector_y_data, image_shape))
    if ImageType.QTRANSFORM in image_types:
        images.append(generate_qtransform_image(vector_y_data, image_shape))

    images_rgba = ary_to_rgba(images)

    return images_rgba


def create_image_by_label_database(
        number_of_chunks: int,
        chunk_size: int,
        image_types: List[ImageType],
        image_shape: tuple,
        noise_scale: float = 0.0,
        shuffle_chunks_across_time=True,
        overwrite_existing_file=False):
    import numpy as np
    from tqdm import tqdm

    try_make_directories()

    filename_for_images_by_labels = npy_dataset_filename(
        number_of_chunks,
        chunk_size,
        image_types,
        image_shape,
        noise_scale,
        shuffle_chunks_across_time
    )

    if overwrite_existing_file is False:
        import os
        if os.path.isfile(filename_for_images_by_labels):
            print(
                "File already exists. Use 'overwrite_existing_file' if you wish to regenerate and overwrite it. Bye.")
            return

    normal_data_clean, ir_data_clean, b_data_clean = load_raw_matlab_data(chunk_size=chunk_size)

    len_n = np.shape(normal_data_clean)[0]
    len_ir = np.shape(ir_data_clean)[0]
    len_b = np.shape(b_data_clean)[0]
    len_data: int = min(len_n, len_ir, len_b, number_of_chunks)

    # add some noise maybe
    normal_data = [x + y for x, y in
                   zip(normal_data_clean, [np.random.normal(0, noise_scale, chunk_size) for i in range(len_n)])]
    ir_data = [x + y for x, y in
               zip(ir_data_clean, [np.random.normal(0, noise_scale, chunk_size) for i in range(len_ir)])]
    b_data = [x + y for x, y in zip(b_data_clean, [np.random.normal(0, noise_scale, chunk_size) for i in range(len_b)])]

    if shuffle_chunks_across_time:
        # so the first N chunks will be taken from a random time from the whole time interval
        np.random.shuffle(normal_data)
        np.random.shuffle(ir_data)
        np.random.shuffle(b_data)

    images_by_labels = {'n': [], 'ir': [], 'b': []}

    for i in tqdm(np.arange(len_data)):
        n_images = generate_rgba_images_from_data(normal_data[i], image_shape, image_types)
        ir_images = generate_rgba_images_from_data(ir_data[i], image_shape, image_types)
        b_images = generate_rgba_images_from_data(b_data[i], image_shape, image_types)

        images_by_labels['n'].extend(n_images)
        images_by_labels['ir'].extend(ir_images)
        images_by_labels['b'].extend(b_images)

    print(f'Saving {filename_for_images_by_labels}')
    np.save(f'{filename_for_images_by_labels}', images_by_labels)


def lists_interleave(lists):
    return [val for tup in zip(*lists) for val in tup]


def separate_images_from_labels(images_by_labels, interleave: bool):
    all_labels = list(images_by_labels.keys())
    NB_CLASSES = len(all_labels)
    all_labels_onehot = np.identity(NB_CLASSES).astype('uint8')
    label_to_onehot_mapper = {all_labels[i]: all_labels_onehot[i] for i in np.arange(NB_CLASSES)}

    all_images_for_cnn = []
    all_labels_for_cnn_onehot = []

    for key in all_labels:
        images = images_by_labels[key]
        labels_onehot = np.full(shape=(len(images), NB_CLASSES), fill_value=label_to_onehot_mapper[key])

        all_images_for_cnn.append(images)
        all_labels_for_cnn_onehot.append(labels_onehot)

    if interleave:
        all_images_for_cnn = lists_interleave(all_images_for_cnn)
        all_labels_for_cnn_onehot = lists_interleave(all_labels_for_cnn_onehot)
    else:  # flatten the lists
        all_images_for_cnn = [image for sublist in all_images_for_cnn for image in sublist]
        all_labels_for_cnn_onehot = [label for sublist in all_labels_for_cnn_onehot for label in sublist]

    return np.array(all_images_for_cnn), np.array(all_labels_for_cnn_onehot)
