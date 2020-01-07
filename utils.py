import os
import lmdb
import torchvision
import numpy as np
import cv2
from tqdm import tqdm

# def create_lmdb():
#     root_path = '/home/jakc4103/windows/Toshiba/workspace/dataset/ILSVRC/Data/CLS-LOC/'
#     lmdb_path = '/home/jakc4103/windows/Dec19/workspace/dataset/ILSVRC/lmdb/trainval'

#     train_path = os.path.join(root_path, 'train')
#     val_path = os.path.join(root_path, 'val')

#     train_dataset = torchvision.datasets.ImageFolder(train_path)
#     val_dataset = torchvision.datasets.ImageFolder(val_path)
    
#     env = lmdb.open(lmdb_path, max_dbs=6, map_size=1e12)

#     data = env.open_db(("data").encode())
#     shape = env.open_db(("shape").encode())
#     label = env.open_db(("label").encode())

#     vdata = env.open_db(("vdata").encode())
#     vshape = env.open_db(("vshape").encode())
#     vlabel = env.open_db(("vlabel").encode())

#     print("Start write lmdb")

#     idx = 0
#     while idx < len(train_dataset):
#         last = idx + 5000
#         with env.begin(write=True) as txn:
#             while idx < last and idx < len(train_dataset):
#                 image, target = train_dataset[idx]
#                 image = np.array(image)
#                 target = np.array(target)

#                 txn.put((str(idx)).encode(), image, db=data)
#                 txn.put((str(idx)).encode(), target, db=label)
#                 txn.put((str(idx)).encode(), np.array(image.shape), db=shape)

#                 idx += 1
#         print('train {}, total {}, percentage {}%'.format(idx, len(train_dataset), round(100*(idx+1)/len(train_dataset))))

#     idx = 0
#     while idx < len(val_dataset):
#         last = idx + 5000
#         with env.begin(write=True) as txn:
#             while idx < last and idx < len(val_dataset):
#                 image, target = val_dataset[0]
#                 image = np.array(image)
#                 target = np.array(target)

#                 txn.put((str(idx)).encode(), image, db=vdata)
#                 txn.put((str(idx)).encode(), target, db=vlabel)
#                 txn.put((str(idx)).encode(), np.array(image.shape), db=vshape)

#                 idx += 1
#         print('val {}, total {}, percentage {}%'.format(idx, len(val_dataset), round(100*(idx+1)/len(val_dataset))))


def create_jpg_lmdb():
    """
    For more details of jpg lmdb:
    https://stackoverflow.com/questions/44280549/write-jpeg-file-directly-to-lmdb
    """
    root_path = '/home/jakc4103/windows/Toshiba/workspace/dataset/ILSVRC/Data/CLS-LOC/'
    lmdb_path = '/home/jakc4103/windows/Toshiba/workspace/dataset/ILSVRC/lmdb/trainval'

    train_path = os.path.join(root_path, 'train')
    val_path = os.path.join(root_path, 'val')

    train_dataset = torchvision.datasets.ImageFolder(train_path)
    val_dataset = torchvision.datasets.ImageFolder(val_path)
    
    env = lmdb.open(lmdb_path, max_dbs=4, map_size=1e12)

    data = env.open_db(("data").encode())
    label = env.open_db(("label").encode())

    vdata = env.open_db(("vdata").encode())
    vlabel = env.open_db(("vlabel").encode())

    print("Start write jpg lmdb")

    # use 5000 samples as one step to  prevent memory exploded
    idx = 0
    while idx < len(train_dataset):
        last = idx + 5000
        with env.begin(write=True) as txn:
            while idx < last and idx < len(train_dataset):
                image, target = train_dataset[idx]
                image = np.array(image)[:, :, [2,1,0]]
                target = np.array(target)

                txn.put((str(idx)).encode(), cv2.imencode('.jpg', image)[1], db=data)
                txn.put((str(idx)).encode(), target, db=label)

                idx += 1
        print('train {}, total {}, percentage {}%'.format(idx, len(train_dataset), round(100*(idx+1)/len(train_dataset))))

    idx = 0
    while idx < len(val_dataset):
        last = idx + 5000
        with env.begin(write=True) as txn:
            while idx < last and idx < len(val_dataset):
                image, target = val_dataset[0]
                image = np.array(image)[:, :, [2,1,0]]
                target = np.array(target)

                txn.put((str(idx)).encode(), cv2.imencode('.jpg', image)[1], db=vdata)
                txn.put((str(idx)).encode(), target, db=vlabel)

                idx += 1
        print('val {}, total {}, percentage {}%'.format(idx, len(val_dataset), round(100*(idx+1)/len(val_dataset))))

if __name__ == '__main__':
    create_jpg_lmdb()