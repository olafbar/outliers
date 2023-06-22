import math
import random as rn
import os
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from PIL import ImageDraw
from skimage.transform import probabilistic_hough_line
from scipy.ndimage import rotate
import imagehash
from IPython.display import display
import cv2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from settings import *


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED, fast_n_close=False):
    """
        Enable 100% reproducibility on operations related to tensor and randomness.
        Parameters:
        seed (int): seed value for global randomness
        fast_n_close (bool): whether to achieve efficient at the cost of determinism/reproducibility
    """
    set_seeds(seed=seed)
    if fast_n_close:
        return

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # # Not working with tf v2.7 but in v2.7 it not necessary
    from tfdeterminism import patch
    #patch()
    from tensorflow.python.ops import nn
    from tensorflow.python.ops import nn_ops
    from tfdeterminism.patch import _new_bias_add_1_14
    tf.nn.bias_add = _new_bias_add_1_14  # access via public API
    nn.bias_add = _new_bias_add_1_14  # called from tf.keras.layers.convolutional.Conv
    nn_ops.bias_add = _new_bias_add_1_14  # called from tests


def draw_text(text, color=255, width=120):
    img = Image.new("L", (width, 12), 0)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, color)
    return np.array(img)


def do_augmentation(trainX, mul=100):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    arr = []
    for image in trainX:
        image = tf.expand_dims(image, 0)
        for i in range(0, mul):
            augmented_image = data_augmentation(image)
            arr.append(augmented_image[0])
    return np.vstack([arr])


def build_unsupervised_dataset(data, labels, kind='mnist'):
    # grab all indexes of the supplied class label that are *truly*
    # that particular label, then grab the indexes of the image
    # labels that will serve as our "anomalies"
    if kind == 'mnist':
        validIdxs = np.where(labels == 1)[0]
        anomalyIdxs = np.where(labels == 3)[0]
    elif kind == 'hits_vs_artefacts':
        validIdxs = np.where(labels != 4)[0]
        anomalyIdxs = np.where(labels == 4)[0]
    elif kind == 'tracks_vs_worms':
        validIdxs = np.where(labels == 2)[0]
        anomalyIdxs = np.where(labels == 3)[0]
    elif kind == 'dots_vs_worms':
        validIdxs = np.where(labels == 1)[0]
        anomalyIdxs = np.where(labels == 3)[0]
    elif kind == 'dots_vs_tracks':
        validIdxs = np.where(labels == 1)[0]
        anomalyIdxs = np.where(labels == 2)[0]
    else:
        raise Exception('Bad kind')

    # randomly shuffle both sets of indexes
    rn.shuffle(validIdxs)
    rn.shuffle(anomalyIdxs)

    # compute the total number of anomaly data points to select
    # i = int(len(validIdxs) * contam)
    # anomalyIdxs = anomalyIdxs[:i]

    # use NumPy array indexing to extract both the valid images and
    # "anomlay" images
    validImages = data[validIdxs]
    anomalyImages = data[anomalyIdxs]

    # stack the valid images and anomaly images together to form a
    # single data matrix and then shuffle the rows
    images = np.vstack([validImages, anomalyImages])
    # images = np.vstack([validImages])
    np.random.shuffle(images)

    # return the set of images
    return np.vstack([validImages]), np.vstack([anomalyImages])


def normalize_indicator(image, normalize=True):
    blacks = 1
    if normalize:
        blacks = count_non_black_pixels(image)
    if blacks == 0:
        blacks = 1
    return blacks


def dm_func_mean(image, recon):
    err = np.mean((image - recon) ** 2)
    return math.log2(err * 5000)


def dm_func_avg_hash(image, recon, normalize=True):
    blacks = normalize_indicator(image, normalize)

    image_hash = imagehash.average_hash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.average_hash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return (image_hash - recon_hash) / (blacks ** 2)


def dm_func_p_hash(image, recon, normalize=True):
    blacks = normalize_indicator(image, normalize)

    image_hash = imagehash.phash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.phash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return (image_hash - recon_hash) / (blacks ** 2)


def dm_func_d_hash(image, recon, normalize=True):
    blacks = normalize_indicator(image, normalize)

    image_hash = imagehash.dhash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.dhash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return (image_hash - recon_hash) / (blacks ** 2)


def dm_func_haar_hash(image, recon, normalize=True):
    blacks = normalize_indicator(image, normalize)

    image_hash = imagehash.whash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.whash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return (image_hash - recon_hash) / (blacks ** 2)


def dm_func_db4_hash(image, recon, normalize=True):
    blacks = normalize_indicator(image, normalize)

    image_hash = imagehash.whash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)), mode='db4')
    recon_hash = imagehash.whash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)), mode='db4')
    return (image_hash - recon_hash) / (blacks ** 2)


def dm_func_cr_hash(image, recon, normalize=True):
    blacks = normalize_indicator(image, normalize)

    image_hash = imagehash.crop_resistant_hash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.crop_resistant_hash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return (image_hash - recon_hash) / (blacks ** 2)


def dm_func_color_hash(image, recon, normalize=True):
    blacks = normalize_indicator(image, normalize)

    image_hash = imagehash.colorhash(Image.fromarray(np.uint8(np.squeeze(image, axis=-1) * 255)))
    recon_hash = imagehash.colorhash(Image.fromarray(np.uint8(np.squeeze(recon, axis=-1) * 255)))
    return (image_hash - recon_hash) / (blacks ** 2)


def visualize_predictions(decoded, images, dm_func=dm_func_mean, marked_first_half=False, max_samples=None):
    # initialize our list of output images
    gt = images
    outputs2 = None
    samples = math.ceil(math.sqrt(gt.shape[0]))
    if max_samples is not None:
        samples = min(samples, max_samples)

    errors = []
    for (image, recon) in zip(images, decoded):
        # compute the mean squared error between the ground-truth image
        # and the reconstructed image, then add it to our list of errors
        mse = dm_func(image, recon)
        errors.append(mse)
    errors_sorted = np.argsort(errors)[::-1]

    # loop over our number of output samples
    for y in range(0, samples):
        outputs = None
        for x in range(0, samples):
            i = y * samples + x
            if i >= gt.shape[0]:
                original = np.full(gt[0].shape, 0)
                recon = original
                i_sorted = 0
            else:
                # grab the original image and reconstructed image
                i_sorted = errors_sorted[i]
                original = (gt[i_sorted] * 255).astype("uint8")
                recon = (decoded[i_sorted] * 255).astype("uint8")

            # stack the original and reconstructed image side-by-side
            output = np.hstack([original, recon])
            v = "" if i >= gt.shape[0] else ' no %3d: %0.6f' % (errors_sorted[i], errors[errors_sorted[i]])
            color = 255
            if marked_first_half and i_sorted < gt.shape[0]/2:
                color = 128
            text = np.expand_dims(draw_text(v, color, width=decoded.shape[1]*2), axis=-1)
            output = np.vstack([output, text])

            # if the outputs array is empty, initialize it as the current
            # side-by-side image display
            if outputs is None:
                outputs = output

            # otherwise, vertically stack the outputs
            else:
                outputs = np.vstack([outputs, output])

        if outputs2 is None:
            outputs2 = outputs

        # otherwise, horizontally stack the outputs
        else:
            outputs2 = np.hstack([outputs2, outputs])

    # return the output images
    return outputs2, errors


def tmp_visualize(images1, images2):
    from dataset_loader import prepare_dataset
    img1 = prepare_dataset(images1)
    img2 = prepare_dataset(images2)
    vis, err = visualize_predictions(img1, img2, lambda x,y:0, False, 16)
    img_path = 'cache/tmp.png'
    cv2.imwrite(img_path, vis)
    display(Image.open(img_path))


def prepare_dataset_old(args, augmentation=False):
    if args["kind"] == "mnist":
        from tensorflow.keras.datasets import mnist
        print("[INFO] loading MNIST dataset...")
        ((train_set, trainY), (unused_set, unused_set2)) = mnist.load_data()
    else:
        from dataset_loader import load_dataset
        print("[INFO] loading CREDO dataset...")
        train_set, trainY = load_dataset()

    # build our unsupervised dataset of images with a small amount of
    # contamination (i.e., anomalies) added into it
    print("[INFO] creating unsupervised dataset...")
    images, anomalies = build_unsupervised_dataset(train_set, trainY, kind=args["kind"])

    # add a channel dimension to every image in the dataset, then scale
    # the pixel intensities to the range [0, 1]
    images = np.expand_dims(images, axis=-1)
    images = images.astype("float32") / 255.0

    anomalies = np.expand_dims(anomalies, axis=-1)
    anomalies = anomalies.astype("float32") / 255.0

    # construct the training and testing split
    (train_set, test_set) = train_test_split(images, test_size=0.2)

    if augmentation:
        train_set = do_augmentation(train_set)

    (train_set, validation_set) = train_test_split(train_set, test_size=0.2)

    # prepare test set
    max_test = min(anomalies.shape[0], test_set.shape[0])
    test_set = np.vstack([anomalies[0:max_test], test_set[0:max_test]])

    return train_set, validation_set, test_set


def original_autoencoder(size=60, kl=False, latentDim=16):
    from pyimagesearch.convautoencoder import ConvAutoencoder
 
    (encoder, decoder, autoencoder) = ConvAutoencoder.build(size, size, 1, latentDim=latentDim)
    # opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=INIT_LR, decay_steps=(INIT_LR / EPOCHS), decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    autoencoder.compile(loss="mse", optimizer=opt, metrics=['kullback_leibler_divergence' if kl else 'accuracy'])
    return autoencoder


def train_or_cache(train_set, autoencoder, fncache=None, force_train=False, epochs=EPOCHS, batch_size=BS, shuffle=False, validation_set=None, kl=False, latentDim=16):
    from os.path import exists
    from keras.models import load_model
    import matplotlib.pyplot as plt

    fn = fncache  # 'cache/%s.h5' % str(fncache)

    if fncache is not None and exists(fn) and not force_train:
        print('Load from: %s' % fn)
        return load_model(fn)

    #(input_set, validation_set) = train_test_split(train_set, test_size=0.2)
    # train the convolutional autoencoder
    H = autoencoder.fit(
        train_set,
        train_set,
        shuffle=shuffle,
        validation_data=(validation_set, validation_set) if validation_set is not None else None,
        epochs=epochs,
        batch_size=batch_size
    )
    # r = autoencoder.evaluate(validation_set, validation_set)

    if fncache is not None:
        autoencoder.save(fn)
        print('Saved in: %s' % fn)

        encoder = Model(autoencoder.input, autoencoder.layers[-2].output)
        encoder.summary()
        # opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=INIT_LR, decay_steps=(INIT_LR / EPOCHS), decay_rate=0.9)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        encoder.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
        encoder.save(fn.replace('.h5', '-encoder.h5'))

        decoder_input = Input(shape=(latentDim,))
        decoder = Model(decoder_input, autoencoder.layers[-1](decoder_input))
        decoder.summary()
        decoder.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
        decoder.save(fn.replace('.h5', '-decoder.h5'))

    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    if validation_set is not None:
        plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(fn.replace('.h5', '_loss.png'))

    if kl:
        N = np.arange(0, EPOCHS)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["kullback_leibler_divergence"], label="kullback_leibler_divergence")
        if validation_set is not None:
            plt.plot(N, H.history["val_kullback_leibler_divergence"], label="val_kullback_leibler_divergence")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig(fn.replace('.h5', '_kullback_leibler_divergence.png'))
    else:
        N = np.arange(0, EPOCHS)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["accuracy"], label="accuracy")
        if validation_set is not None:
            plt.plot(N, H.history["val_accuracy"], label="val_accuracy")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig(fn.replace('.h5', '_accuracy.png'))

    return autoencoder


def binarize_image(image, cutoff_qt_value=0):
    """
    Binaryzacja obrazka na podstawie podanego progu. Domyślnie, co nie jest całkiem czarne, jest białe.

    :param image: źródlowy obrazek
    :param cutoff_qt_value: próg binaryzacji, 1 - większe od progu
    :return:
    """
    return np.where(image > cutoff_qt_value, 1, 0)


def cutoff_reconstruction_background(image, reconstruction):
    """
    Odcina tło od rekonstrukcji, która w oryginalnym obrazku była tłem.

    W oryginalnym obrazku mamy coś na tle czerni (0).

    :return: rekonstrukcja z obciętym tłem
    """
    return binarize_image(image, 0) * reconstruction

def count_non_black_pixels(image):
    """
    Zwraca liczbę nieczarnych pikseli z obrazka.

    :param image: obrazek.
    :return: liczba nieczarnych pikseli.
    """
    return np.count_nonzero(image)

def compute_errors(image, recon, dm_func, normalize=True):
    """
    Obliczanie błędu.

    :param image: obrazek źródłowy.
    :param recon: rekonstrukcja.
    :param dm_func: funkcja porównująca, jako parametr przyjmuje (image, recon), zwraca skalar będący miarą podobieństwa.
    :param normalize: jeśli true, to dzieli wynik dm_func przez liczbę nieczarnych pikseli z image.
    :return: tablica
    """
    return dm_func(image, recon, normalize)


def prepare_for_histogram(images, reconstructions, dm_func, normalize=True, cutoff_background=False, binarize_for_compare=False):
    errors = []
    for (image, recon) in zip(images, reconstructions):
        try:
            if cutoff_background:
                recon = cutoff_reconstruction_background(image, recon)
            if binarize_for_compare:
                image = binarize_image(image)
                recon = binarize_image(recon)
            mse = compute_errors(image, recon, dm_func, normalize)
            errors.append(mse)
        except:
            errors.append(0)
    return errors


def dm_func_mean2(image, recon, normalize=True):
    blacks = normalize_indicator(image, normalize)

    err = np.mean((image - recon) ** 2) / (blacks ** 2)
    if err == 0:
        return 0
    return math.log2(err * 5000)


def dm_func_mean3(image, recon, normalize=True):
    blacks = normalize_indicator(image, normalize)

    err = np.mean((image - recon) ** 2) / (blacks ** 2)
    return err


def calc_similarity(autoencoder, dots_set, tracks_set, worms_set, artifacts_set, dm_func=dm_func_mean2, **argv):
    dots_reconstruction = autoencoder.predict(dots_set)
    worms_reconstruction = autoencoder.predict(worms_set)
    tracks_reconstruction = autoencoder.predict(tracks_set)
    artifacts_reconstruction = autoencoder.predict(artifacts_set)

    return {
        'dots': prepare_for_histogram(dots_set, dots_reconstruction, dm_func, **argv),
        'worms': prepare_for_histogram(worms_set, worms_reconstruction, dm_func, **argv),
        'tracks': prepare_for_histogram(tracks_set, tracks_reconstruction, dm_func, **argv),
        'artifacts': prepare_for_histogram(artifacts_set, artifacts_reconstruction, dm_func, **argv)
    }


def calc_encoded(encoder, dots_set, tracks_set, worms_set, artifacts_set):

    #autoencoder.summary()

    #decoder_input = Input(shape=(16,))
    #decoder = Model(decoder_input, autoencoder.layers[-1](decoder_input))
    #decoder.summary()

    #ai = autoencoder.input
    #ao = autoencoder.layers[-2].output
    #encoder = Model(ai, ao)
    encoder.summary()

    dots_encoded = encoder.predict(dots_set)
    worms_encoded = encoder.predict(worms_set)
    tracks_encoded = encoder.predict(tracks_set)
    artifacts_encoded = encoder.predict(artifacts_set)

    return {
        'dots': dots_encoded,
        'worms': worms_encoded,
        'tracks': tracks_encoded,
        'artifacts': artifacts_encoded
    }


def save_encoded_channel(fn, channel, files):
    from numpy import savetxt
    #f = np.asarray(files)
    #f = np.expand_dims(f, axis=1)
    #data = np.hstack([f, channel])
    #savetxt(fn, [[f, channel]], delimiter='\t', fmt=['%s', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e', '%.18e'])
    with open(fn + '.txt', 'w') as f:
        for i in range(0, len(files)):
            #f.write(files[i].split('\\')[-1])
            f.write(files[i])
            #for a in channel[i]:
                #f.write('\t%.18e' % a)
            f.write('\n')
    np.save(fn + '.npy', channel)


def save_encoded(encoded, prefix, dots_files, tracks_files, worms_files, artifacts_files):
    #save_encoded_channel(prefix + 'dots', encoded['dots'], dots_files)
    #save_encoded_channel(prefix + 'worms', encoded['worms'], worms_files)
    #save_encoded_channel(prefix + 'tracks', encoded['tracks'], tracks_files)
    #save_encoded_channel(prefix + 'artifacts', encoded['artifacts'], artifacts_files)
    #files = [*dots_files, *tracks_files, *worms_files, *artifacts_files]
    #channel = np.vstack([encoded['dots'], encoded['worms'], encoded['tracks'], encoded['artifacts']])
    files = [*tracks_files, *worms_files]
    channel = np.vstack([encoded['worms'], encoded['tracks']])
    save_encoded_channel(prefix, channel, files)


def append_label_column(v, c):
    with_c = np.vstack([v, np.ones(v.shape) * c])
    transposed = with_c.transpose()
    return transposed


def find_threshold(sa, sb):
    a = sorted(sa)
    b = sorted(sb)

    la = len(a)
    lb = len(b)

    if la > lb:
        from scipy.ndimage import zoom
        a2 = zoom(a, lb / la)
        a = a2
    elif la < lb:
        from scipy.ndimage import zoom
        b2 = zoom(b, la / lb)
        b = b2

    la = len(a)
    lb = len(b)

    ta = int(la // 2)
    tb = int(lb // 2)

    ca = a[ta]
    cb = b[tb]

    if ca > cb:
        swap = a
        a = b
        b = swap

    np_a = np.array(a)
    np_b = np.array(b)

    np_a_ex = append_label_column(np_a, 1)
    np_b_ex = append_label_column(np_b, 2)

    left = 4
    right = 3

    np_ab = np.vstack([np_a_ex, np_b_ex])
    np_ab = np_ab[np_ab[:, 0].argsort()]

    labels = np_ab[:,1]

    cumsum_a = np.where(labels == 1, 1 / len(a), 0).cumsum()
    cumsum_ar = np.flip(np.flip(np.where(labels == 1, 1 / len(a), 0)).cumsum())
    cumsum_b = np.where(labels == 2, 1 / len(b), 0).cumsum()
    cumsum_br = np.flip(np.flip(np.where(labels == 2, 1 / len(b), 0)).cumsum())

    np_ab_ex = np.hstack([np_ab, np.vstack([cumsum_a, cumsum_ar, cumsum_b, cumsum_br]).transpose()])

    half = 0
    for row in np_ab_ex:
        rl = row[left]
        rr = row[right]
        if rl > rr:
            break
        half += 1

    threshold = (np_ab_ex[half - 1, 0] + np_ab_ex[half, 0])/2

    fp_a = np_ab[np.where((np_ab[:,1] == 1) & (np_ab[:,0] >= threshold))]
    fp_b = np_ab[np.where((np_ab[:,1] == 2) & (np_ab[:,0] <= threshold))]

    percent = (fp_a.shape[0] + fp_b.shape[0]) / np_ab.shape[0] * 100

    return threshold, percent


def plot_threshold(a, b, threshold, name_a, name_b):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.hist(a, bins=100, alpha=0.5, label=name_a)
    plt.hist(b, bins=100, alpha=0.5, label=name_b)
    plt.vlines(x = threshold, ymin = 0, ymax = 50, colors = 'purple', label = 'Threshold')
    plt.xlabel("Data", size=14)
    plt.ylabel("Count", size=14)
    plt.legend(loc='upper right')
    plt.show()


def confusion_matrix(on):
    import matplotlib.pyplot as plt

    x_labels = ["dots", "tracks", "worms", "artifacts"]
    y_labels = x_labels

    matrix = np.zeros((len(x_labels), len(x_labels)), dtype=np.int32)

    for i in range(0, len(x_labels)):
        for j in range(0, len(x_labels)):
            if i == j:
                continue
            la = x_labels[i]
            lb = x_labels[j]
            a = on[la][la]
            b = on[la][lb]
            threshold, percent = find_threshold(a, b)

            print('Channel %s, compare %s vs %s, threshold: %s, fp/fn percent: %s %%' % (
            la, la, lb, str(threshold), str(percent)))
            plot_threshold(a, b, threshold, la, lb)

            matrix[i, j] = int(100 - percent)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("x - data set, y - channel")
    fig.tight_layout()
    plt.show()


def angle(x, y):
    rad = np.arctan2(y, x)
    degrees = rad*180/np.pi
    return degrees


def round_normalize(image):
    lines = probabilistic_hough_line(image, threshold=10, line_length=5, line_gap=3)
    angles = []
    for l in lines:
        vector_1 = [l[1][1] - l[0][1], l[1][0] - l[0][0]]
        #print(vector_1)
        angles.append(angle(l[1][0] - l[0][0], l[1][1] - l[0][1]))
    deg = np.average(angles) if len(angles) else 0
    #print(deg)
    #print("")

    mask = np.where(image == 0, 0.0, 1.0)
    rotated_mask = rotate(mask, deg, reshape=False)
    rotated_image = rotate(image, deg, reshape=False)
    rotated_image = np.where(rotated_mask < 0.25, 0, rotated_image)
    rotated_image = np.where(rotated_image < 0, 0, rotated_image)
    rotated_image = np.where(rotated_image > 1, 1, rotated_image)
    return rotated_image


def round_normalize_spread(image, r1=0, r2=180):
    mask = np.where(image == 0, 0.0, 1.0)
    min_var = 1000
    deg = 0
    for d in range(r1, r2):
        rotated_mask = rotate(mask, d, reshape=False)
        rotated_image = rotate(image, d, reshape=False)
        rotated_image = np.where(rotated_mask < 0.25, 0, rotated_image)
        #rotated_image.sort(axis=0)
        #rotated_image.sort(axis=1)

        s = rotated_image.sum(axis=1)
        v = np.var(s)
        if v < min_var:
            min_var = v
            deg = d
        #print("%d: %f" % (d, v))
        #rotated_image.sort(axis=1)

    mask = np.where(image == 0, 0.0, 1.0)
    deg += 90
    rotated_mask = rotate(mask, deg, reshape=False)
    rotated_image = rotate(image, deg, reshape=False)
    rotated_image = np.where(rotated_mask < 0.25, 0, rotated_image)
    rotated_image = np.where(rotated_image < 0, 0, rotated_image)
    rotated_image = np.where(rotated_image > 1, 1, rotated_image)
    return rotated_image, deg


def normalize_rotation(images, method):
    img = []
    print('Normalize rotation 0 of %d images...' % len(images))
    c = 0
    for i in images:
        c += 1
        img.append(method(i))
        if c % 100 == 0:
            print('Normalize rotation %d of %d images...' % (c, len(images)))
    ret = np.array(img)
    return ret
