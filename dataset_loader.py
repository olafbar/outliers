import glob
import math
import random

from PIL import Image
from imageio import imread
import numpy as np
import tensorflow as tf
from keras.layers import RandomFlip, RandomRotation


DATA_SETS = {
    'dots': "hit-images-final/hits_votes_4_Dots",
    'tracks': "hit-images-final/hits_votes_4_Lines",
    'worms': "hit-images-final/hits_votes_4_Worms",
    'artifacts': "hit-images-final/artefacts"
}


def do_binarize(image):
    img = image.astype('int32')
    blackwhite = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0

    threshold = blackwhite.mean() + blackwhite.std() * 5
    threshold = threshold if threshold < 100 else 100
    mask = np.where(blackwhite > threshold, 1, 0)
    blackwhite = blackwhite * mask
    return blackwhite


def load_images(src, resize=1):
    images = []
    files = list(glob.glob("%s/*.png" % src))
    files = sorted(files)
    for image_path in files:
        #print(image_path)
        #image = np.asarray(Image.open(image_path).convert('L'))
        img = Image.open(image_path)
        if resize != 1:
            img = img.resize((img.width * resize, img.height * resize), Image.BICUBIC)
        image = np.asarray(img)
        image = do_binarize(image)
        images.append(image)
    return np.asarray(images).astype("float32") / 255.0, files


def cut_image_to_size(image, cut_to=60):
    x1 = int((image.shape[0] - cut_to) / 2)
    x2 = x1 + cut_to
    y1 = int((image.shape[1] - cut_to) / 2)
    y2 = y1 + cut_to
    return image[x1:x2, y1:y2]


def prepare_dataset(src):
    """
    Obrazki po wczytaniu z dysku są w shapie (N, 60, 60), a do autoencodera potrzeba (N, 60, 60, 1)
    :param src: tablica ze shape (N, 60, 60) lub (60, 60) gdy to tylko jeden obrazek
    :return: tablica ze shape (N, 60, 60, 1)
    """
    if isinstance(src, list):
        src = np.vstack([src])

    if len(src.shape) == 4 and src.shape[1] in [60, 64] and src.shape[2] in [60, 64] and src.shape[3] == 1:
        # jest ok
        return src

    if len(src.shape) == 3 and src.shape[1] in [60, 64] and src.shape[2] in [60, 64]:
        # jest to świeży image set, więc dodanie ", 1" do shape
        return np.expand_dims(src, axis=-1)

    if len(src.shape) == 2 and src.shape[0] in [60, 64] and src.shape[1] in [60, 64]:
        # to pojedynczy obrazek, więc zrobienie tablicy i dodanie ", 1" do shape
        arr = np.expand_dims(src, axis=0)
        return np.expand_dims(arr, axis=-1)

    raise Exception("Wymagany shape (N, 60, 60), lub (60, 60), ewentualnie (N, 60, 60, 1) ale wtedy nic nie jest zmieniane")


def do_augmentation2(images, mul=1, data_augmentation=None):
    if mul == 1:
        return images

    if data_augmentation is None:
        data_augmentation = tf.keras.Sequential([
            #RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.4),
        ])

    arr = []
    for image in images:
        image = tf.expand_dims(image, 0)
        for i in range(0, mul):
            augmented_image = data_augmentation(image)
            arr.append(augmented_image[0])
    return np.vstack([arr])


def prepare_data(src, augmentation=1):
    images = load_images(src)[0]
    expanded = do_augmentation2(images, augmentation)
    return expanded


def load_dataset_with_cache(dataset, augmentation=1, force_load=False):
    from os.path import exists
    import pickle

    fn = 'cache/dataset_%s_%d.pickle' % (dataset, augmentation)
    if not force_load and exists(fn):
        return pickle.loads(open(fn, "rb").read())

    images = np.expand_dims(prepare_data(DATA_SETS[dataset], 1), axis=-1)
    expanded = np.expand_dims(prepare_data(DATA_SETS[dataset], augmentation), axis=-1)
    data_set = (images, expanded)

    f = open(fn, "wb")
    f.write(pickle.dumps(data_set))
    f.close()

    return data_set


def load_dataset_with_augmentation(source_dir, augmentation=None, min_count=None):
    """
    Wczytanie sampli z opcjonalną augmentacją. Użyć parametru augmentation albo min_count, nie obu na raz.

    :param source_dir: katalog źródłowy z plikami PNG.
    :param augmentation: krotność augmentacji.
    :param min_count: minimalna liczba próbek po augmentacji.
    :return: (dane, augmentowane_dane)
    """
    images = load_images(source_dir)[0]
    if augmentation is not None:
        expanded = do_augmentation2(images, augmentation)
    elif min_count is not None:
        expanded = do_augmentation2(images, math.ceil(min_count / len(images)))
    else:
        expanded = images
    return images, expanded


def save_to_file(fn, data):
    """
    Zrzut danych do pliku. Można je z powrotem załadować za pomocą load_from_file.

    :param fn: nazwa pliku.
    :param data: dane do zapisu.
    """
    import pickle
    f = open(fn, "wb")
    f.write(pickle.dumps(data))
    f.close()


def load_from_file(fn):
    """
    Wczytanie danych zapisanych wcześniej przez save_to_file.

    :param fn: nazwa pliku.
    :return: wczytane dane.
    """

    import pickle
    return pickle.loads(open(fn, "rb").read())


def load_dataset():
    x_spots = load_images("hit-images-final/hits_votes_4_Dots")[0]
    x_tracks = load_images("hit-images-final/hits_votes_4_Lines")[0]
    x_worms = load_images("hit-images-final/hits_votes_4_Worms")[0]
    x_artifacts = load_images("hit-images-final/artefacts")[0]

    x_all = np.vstack([x_spots, x_tracks, x_worms, x_artifacts])

    y_spots = np.full((x_spots.shape[0]), 1)
    y_tracks = np.full((x_tracks.shape[0]), 2)
    y_worms = np.full((x_worms.shape[0]), 3)
    y_artifacts = np.full((x_artifacts.shape[0]), 4)

    y_all = np.hstack([y_spots, y_tracks, y_worms, y_artifacts])
    return x_all, y_all


def load_data(seed=1, div=4):
    """
    Dataset, return based on MNIST dataset:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    """
    x_all, y_all = load_dataset()
    indices = list(range(0, x_all.shape[0]))
    random.seed(seed)
    random.shuffle(indices)
    splitter = int(len(indices) - len(indices) / div)
    train = indices[0:splitter]
    test = indices[splitter:]

    x_train = x_all[train]
    y_train = y_all[train]
    x_test = x_all[test]
    y_test = y_all[test]

    return (x_train, y_train), (x_test, y_test)
