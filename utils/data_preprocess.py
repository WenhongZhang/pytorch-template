import numpy as np
from PIL import Image
from pathlib import Path

if __name__ == '__main__':
    """
    1/unpack datasets and transform the images into png format, rewrite it to custom your dataset4
        (datasets and python files are in the same dictionary)
    2/split validation set from train set
    3/generate a txt file for dataset object
    """
    import pickle
    import random
    import shutil

    def unpickle(file):
        """"
        unpack the archive and return a dict
        """
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # just for debug. You can set all 3 variables to be True once you just download the
    # dataset archive but not unpack & split & generate txt
    unpack = False
    split_valid = False
    generate_txt = False

    if unpack is True:
        data_dir = Path('.')
        train_dir = Path('./Data/raw_train')
        test_dir = Path('./Data/raw_test')
        # load train data
        for batch in range(1, 6):
            data_path = data_dir/'cifar-10-batches-py'/('data_batch_'+str(batch))
            train_data = unpickle(data_path)

            for index in range(0, 10000):
                img = np.reshape(train_data[b'data'][index], (3, 32, 32))
                img = img.transpose((1, 2, 0))  # for fromarray method
                label_num = str(train_data[b'labels'][index])
                p = train_dir/label_num
                if not p.exists():
                    p.mkdir(parents=True, exist_ok=True)
                img_name = label_num + '_' + str(index + (batch-1)*10000) + '.png'
                img_path = p/img_name
                img = Image.fromarray(img)  # fromarray method takes (height, width, channels) format
                img.save(img_path)
            print('batch {} is loaded'.format(batch))

        # load test data
        data_path = data_dir/'cifar-10-batches-py'/'test_batch'
        test_data = unpickle(data_path)
        for index in range(0, 10000):
            img = np.reshape(test_data[b'data'][index], (3, 32, 32))
            img = img.transpose((1, 2, 0))
            label_num = str(test_data[b'labels'][index])
            p = test_dir/label_num
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            img_name = label_num + '_' + str(index) + '.png'
            img_path = p / img_name
            img = Image.fromarray(img)
            img.save(img_path)
        print('batch {} is loaded'.format('test'))

    if split_valid is True:
        """
        to split validation data from train set
        """
        train_percent = 0.8
        valid_percent = 0.2
        raw_train_dir = Path('./Data/raw_train')
        split_train_dir = Path('./Data/split_train')
        split_valid_dir = Path('./Data/split_valid')

        for s_dirs in raw_train_dir.iterdir():  # 获取 train文件下各文件夹名称
            if s_dirs.is_dir():

                # i_dir = sub_dir.absolute()  # 获取各类的文件夹 绝对路径
                img_list = s_dirs.glob('*.png')  # 获取类别文件夹下所有png图片的路径
                random_img_list = []
                for i, file_path in enumerate(img_list):
                    random_img_list.append(file_path.absolute())

                random.seed(1)
                random.shuffle(random_img_list)
                img_len = len(random_img_list)
                train_point = int(img_len*train_percent)    # the split point for train and valid
                for i in range(img_len):
                    if i < train_point:
                        out_dir = split_train_dir/s_dirs.name
                    else:
                        out_dir = split_valid_dir/s_dirs.name

                    if not out_dir.exists():
                        out_dir.mkdir(parents=True, exist_ok=True)
                    out_file_path = out_dir/random_img_list[i].name
                    shutil.copy(str(random_img_list[i].absolute()), str(out_file_path))
            else:
                continue

    if generate_txt is True:
        """
        generate txt file for dataset
        """

        train_txt_path = Path('./Data/train.txt')
        train_dir = Path('./Data/split_train')

        valid_txt_path = Path('./Data/valid.txt')
        valid_dir = Path('./Data/split_valid')

        test_txt_path = Path('./Data/test.txt')
        test_dir = Path('./Data/raw_test')

        def gen_txt(txt_path, img_dir):
            f = open(txt_path, 'w')
            p = Path(img_dir)
            for s_dirs in p.iterdir():  # 获取 train文件下各文件夹名称
                if s_dirs.is_dir():
                    # i_dir = sub_dir.absolute()  # 获取各类的文件夹 绝对路径
                    img_list = s_dirs.glob('*.png')  # 获取类别文件夹下所有png图片的路径
                    for i, file_path in enumerate(img_list):
                        label = file_path.name.split('_')[0]
                        img_path = file_path.absolute()
                        line = str(file_path) + ' ' + label + '\n'
                        f.write(line)
            f.close()

        gen_txt(train_txt_path, train_dir)
        gen_txt(valid_txt_path, valid_dir)
        gen_txt(test_txt_path, test_dir)