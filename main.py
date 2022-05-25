import time
import json
from data_loader import Scene15
from SPM import SPM


def train(model, images, labels, precompute_feature = None):
    if precompute_feature is None:
        print("Extracting Dense sift feature")
        low_features = model.feature_dense_sift(images)
        if model.save_sift_feature:
            model.save_feature(low_features)
    else:
        print("Using pre-computed feature")
        low_features = precompute_feature
        model._set_feature_hw(images)

    print("Traing vocabulary")
    if model.do_train_clusters:
        model.train_clusters(low_features)

    print("Extracting BOF feature")
    bof = model.toBOF(low_features)

    print("Training classifier")
    if model.do_train_classifier:
        model.train_classifier(bof, labels)

    model.save_model()


if __name__ == "__main":
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset = Scene15(config['data_loader']['data_dir'])
    X1, y1, X2, y2 = dataset.xy(using_clss=config["using_clss"],
                                num_clss_train_image = config["num_clss_train_image"])

    startime = time.clock()
    try:
        s = SPM(
            M=config["M"], L=config["L"]
        )

        train(s, X1, y1)
        report = s.test(X2, y2)

        # # example of training with precomputed features.
        # train_feature = s.load_feature(s.saveroot + "feature_train.npy")
        # test_feature = s.load_feature(s.saveroot + "feature_test.npy")
        # s.train(X1, y1, precompute_feature=train_feature)
        # report = s.test(X2, y2, precompute_feature=test_feature)

        json_dump(report, s.save_root + "save" + currentTime() + ".json")
        json_dump(config, s.save_root + "config" + currentTime() + ".json")

    except Exception as e:
        endtime = time.clock()
        print("Using {} s".format(endtime - startime))
        raise e

    endtime = time.clock()
    print("Using {} s".format(endtime - startime))