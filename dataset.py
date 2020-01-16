import torch
import numpy as np
import lmdb
from PIL import Image
import cv2

class ImagenetLMDBDataset(torch.utils.data.Dataset):
    """Car dataset."""
    def __init__(self, root_dir, transform, db_name=['data', 'label']):
        super(ImagenetLMDBDataset, self).__init__()

        self.transform = transform

        self.env = lmdb.open(root_dir, max_dbs=4, map_size=1e12)

        self.data_db = self.env.open_db(db_name[0].encode())
        self.label_db = self.env.open_db(db_name[1].encode())

        self.txn = self.env.begin(write=False)
        self.num_data = self.txn.stat(db=self.data_db)['entries']

        print("ImageNet LMDB created with {} entries".format(self.num_data))

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # Read image, jpeg lmdb version
        img = np.fromstring(self.txn.get(str(idx).encode(), db=self.data_db), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)[:,:,[2,1,0]] # BGR to RGB

        label = self.txn.get(str(idx).encode(), db=self.label_db)
        
        img = Image.fromarray(img)
        label = int(np.frombuffer(label, 'int32')[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, label