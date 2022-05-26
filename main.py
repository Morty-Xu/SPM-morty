import time
import json
import numpy as np
from tqdm import tqdm
from data_loader import Scene15
from SPM import SPM
from utils import currentTime, json_dump
from sklearn.metrics import classification_report


def train(model, images, labels, precompute_feature = None):
    if precompute_feature is None:
        print("Extracting Dense sift feature")
        low_features = model.feature_dense_sift(images)
        if model.save_sift_feature:
            model.save_feature(low_features)
    else:
        print("Using pre-computed feature")
        low_features = precompute_feature
        model.set_feature_hw(images)

    print("Traing vocabulary")
    if model.do_train_clusters:
        model.train_clusters(low_features)

    print("Extracting BOF feature")
    bof = model.toBOF(low_features)

    print("Training classifier")
    if model.do_train_classifier:
        model.train_classifier(bof, labels)

    model.save_model()


def test(model, images, labels, batch_size = 100, precompute_feature = None):

    predict = inference(model=model, images=images,
                batch_size=batch_size, precompute_feature= precompute_feature)
    report = classification_report(labels, predict, output_dict = True)
    print(report)
    return report


def inference(model, images, batch_size = 100, precompute_feature = None):
    """ inference procedure, able to use batch to accelerate the progress of inference.
    """

    if precompute_feature is None:
        print("Extracting Dense sift feature")
        low_features = model.feature_dense_sift(images)
        if model.save_sift_feature:
            model.save_feature(low_features)
    else:
        print("Using pre-computed feature")
        low_features = precompute_feature
        model.set_feature_hw(images)

    print("Extracting BOF feature")
    bof = model.toBOF(low_features)

    num_samples = bof.shape[0]
    predict = np.zeros((num_samples, ), dtype = int)
    iternum = num_samples // batch_size  + 1

    print("Inference on classifier")
    for i in tqdm(range(iternum)):
        start_idx = batch_size * i
        end_idx = min(batch_size * (i + 1), num_samples)
        if end_idx - start_idx == 0:
            break
        data = bof[start_idx: end_idx, ]
        out = model.predict_clss(data)
        predict[start_idx: end_idx, ] = out

    return predict


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset = Scene15(config["data_loader"]["args"]["data_dir"])
    X1, y1, X2, y2 = dataset.xy(using_clss=config["trainer"]["using_clss"],
                                num_clss_train_image = config["trainer"]["num_clss_train_image"])

    startime = time.perf_counter()
    try:
        s = SPM(
            M=config["trainer"]["M"],
            L=config["trainer"]["L"],
            save_root=config['trainer']['save_dir']
        )

        train(s, X1, y1)
        report = test(s, X2, y2)
        #
        # # # example of training with precomputed features.
        # # train_feature = s.load_feature(s.saveroot + "feature_train.npy")
        # # test_feature = s.load_feature(s.saveroot + "feature_test.npy")
        # # s.train(X1, y1, precompute_feature=train_feature)
        # report = s.test(X2, y2, precompute_feature=test_feature)
        #
        json_dump(report, s.save_root + "save" + currentTime() + ".json")
        json_dump(config, s.save_root + "config" + currentTime() + ".json")

    except Exception as e:
        endtime = time.perf_counter()
        print("Using {} s".format(endtime - startime, '.2f'))
        raise e

    endtime = time.perf_counter()
    print("Using {} s".format(endtime - startime, '.2f'))