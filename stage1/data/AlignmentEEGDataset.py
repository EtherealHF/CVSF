from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os

class AlignmentEEGDataset():
    """
    仅适用一个sub
    """
    def __init__(self, eeg_path, image_path, subjects=None, train=None, n_cls=None, 
                 repeat_times=None, samples_per_class=None, n_channels=None, 
                 frequence=250, image_tensor_path=None, sub_3000=False):
        self.eeg_path = eeg_path
        self.image_path = image_path
        self.n_cls = n_cls
        self.train = train
        self.subjects = subjects
        self.repeat_times = repeat_times
        self.samples_per_class = samples_per_class
        self.image_tensor_path = image_tensor_path
        self.sub_3000 = sub_3000
        
        # 加载多模态数据
        # print(f"loading eeg data...")
        self.eeg_data = self._load_eeg_data()  # Dict of beta/gamma tensors
        # print(f"load image tensor...")
        self.image_tensors = self._load_image_tensors()  # Dict of high/low/have_color/no_color tensors

        self.raw_image, self.raw_eeg = self._load_raw_data() 

        print(f"n_cls: {self.n_cls}, sub_3000: {self.sub_3000}, samples_per_class: {self.samples_per_class}")
        print(f"eeg_data shape: {self.eeg_data['beta'].shape}")
        print(f"image_tensor shape: {self.image_tensors['high'].shape}")
        print(f"raw_image shape: {self.raw_image.shape}")
        print(f"raw_eeg shape: {self.raw_eeg.shape}")
        
    def _load_eeg_data(self):
        """加载beta和gamma频段的EEG数据"""
        eeg_types = ['beta', 'gamma']
        eeg_data = {}
        
        for eeg_type in eeg_types:
            data_list = []
            if self.train:
                if self.sub_3000:
                    file_name = f"preprocessed_eeg_training_{eeg_type}_3000.npy"
                else:
                    file_name = f"preprocessed_eeg_training_{eeg_type}.npy"
            else:
                file_name = f"preprocessed_eeg_test_{eeg_type}.npy"
             
            subject = self.subjects[0]
            eeg_path = os.path.join(self.eeg_path, subject, file_name)
            eeg_data_ = np.load(eeg_path, allow_pickle=True).item()
            preprocessed_eeg_data = torch.from_numpy(eeg_data_['preprocessed_eeg_data']).float().detach()

            for i in range(self.n_cls):
                 start_index = i * self.samples_per_class
                 preprocessed_eeg_data_per_class = preprocessed_eeg_data[start_index:start_index+self.samples_per_class]
                 if self.train:
                       data_list.append(preprocessed_eeg_data_per_class)
                 else:
                       data_list.append(torch.mean(preprocessed_eeg_data_per_class.squeeze(0), 0))
            
            if self.train:
                # data_list: 长度为n_sub * 1654的list, 每个元素是shape为[10, 4, 63, 250]的Tensor张量
                data_tensor = torch.cat(data_list, dim=0).view(-1, *(data_list[0].shape[2:])) # [n_sub * 66160, 63, 250]
            else:
                # data_list: 长度为n_sub * 200的list, 每个元素是shape为[63, 250]的Tensor
                data_tensor = torch.cat(data_list, dim=0).view(-1, *(data_list[0].shape)) # [n_sub * 200, 63, 250] 
            
            eeg_data[eeg_type] = data_tensor
        
        # [n_cls, samples_per_class, 4, C, T]
        return eeg_data
    
    def _load_raw_data(self):
        data_list = []

        if self.train:
            if self.sub_3000:
               eeg_raw_file_name =  "preprocessed_eeg_training_raw_3000.npy"
               image_raw_file_name = "preprocessed_images_train_3000.pt"  # 改这里
            else:
               eeg_raw_file_name =  "preprocessed_eeg_training_raw.npy"
               image_raw_file_name = "preprocessed_images_train.pt"  # 改这里
            
        #    eeg_raw_file_name =  "preprocessed_eeg_training_raw.npy"
        #    image_raw_file_name = "preprocessed_raw_train_images_low_light_3000.pt"  #与低光raw图片对齐
        #    image_raw_file_name = "preprocessed_train_images_normal_3000.pt" # 与正常normal图片对齐
        else:
           eeg_raw_file_name = "preprocessed_eeg_test_raw.npy" # 改这里
        #    image_raw_file_name = "preprocessed_raw_test_images_low_light.pt" # 与低光raw图片对齐
           image_raw_file_name = "preprocessed_images_test.pt" 
        
        eeg_path = os.path.join(self.eeg_path, self.subjects[0], eeg_raw_file_name)
        image_path = os.path.join("/path/to/THINGS-EEG2/features/", image_raw_file_name)

        print(f"AlignmentEEGDataset eeg_file: {eeg_path}")
        print(f"AlignmentEEGDataset image_file: {image_path}")

        eeg_data = np.load(eeg_path, allow_pickle=True).item()
        preprocessed_eeg_data = torch.from_numpy(eeg_data['preprocessed_eeg_data']).float().detach()

        for i in range(self.n_cls):
            start_index = i * self.samples_per_class
            preprocessed_eeg_data_per_class = preprocessed_eeg_data[start_index:start_index+self.samples_per_class]

            if self.train:
                 data_list.append(preprocessed_eeg_data_per_class)
            else:
                 data_list.append(torch.mean(preprocessed_eeg_data_per_class.squeeze(0), 0))
        
        if self.train:
            # data_list: 长度为n_sub * 1654的list, 每个元素是shape为[10, 4, 63, 250]的Tensor张量
            data_tensor = torch.cat(data_list, dim=0).view(-1, *(data_list[0].shape[2:])) # [n_sub * 66160, 63, 250]
        else:
            # data_list: 长度为n_sub * 200的list, 每个元素是shape为[63, 250]的Tensor
            data_tensor = torch.cat(data_list, dim=0).view(-1, *(data_list[0].shape)) # [n_sub * 200, 63, 250] 

        image = torch.load(image_path, weights_only=True)

        return image, data_tensor 
        
    
    def _load_image_tensors(self):
        """加载四种图像特征"""
        # image_types = ['high', 'low', 'have_color', 'no_color']
        image_types = ['high', 'low']
        image_tensors = {}
        
        if self.train == True:
            for img_type in image_types:
                if self.sub_3000:
                    file_path = os.path.join(self.image_tensor_path, f"preprocessed_{img_type}_train_images_3000.pt") # 改这里
                else:
                    file_path = os.path.join(self.image_tensor_path, f"preprocessed_{img_type}_train_images.pt") # 改这里
                image_tensors[img_type] = torch.load(file_path, weights_only=True)  # [16540, 3, 224, 224]
        else:
            for img_type in image_types:
                file_path = os.path.join(self.image_tensor_path, f"preprocessed_{img_type}_test_images.pt") # 改这里
                image_tensors[img_type] = torch.load(file_path, weights_only=True)  # [16540, 3, 224, 224]
        
        return image_tensors
    
    
    def __getitem__(self, index):
     
        eeg_beta = self.eeg_data['beta'][index]
        eeg_gamma = self.eeg_data['gamma'][index]
        eeg = torch.stack([eeg_beta, eeg_gamma])
        eeg_raw = self.raw_eeg[index]

        # 获取图像特征 [4, 3, 224, 224]
        # img_idx = class_idx * self.samples_per_class + sample_idx

        index_n_sub = self.n_cls * self.samples_per_class * self.repeat_times
        if self.train:
            image_index = (index % index_n_sub) // (self.repeat_times)
        else:
            image_index = (index % index_n_sub)
        image_raw = self.raw_image[image_index]

        image_features = torch.stack([
            self.image_tensors['high'][image_index],
            self.image_tensors['low'][image_index],
            # self.image_tensors['have_color'][image_index],
            # self.image_tensors['no_color'][image_index]
        ])  # [4, 3, 224, 224]

        # print(f"AlignmentEEGDataset index: {index} | AlignmentEEGDataset image_index: {image_index}")

        return image_features, eeg, image_raw, eeg_raw

    
    def __len__(self):
        if self.train:
            return self.n_cls * self.samples_per_class * self.repeat_times
        else:
            return self.n_cls * self.samples_per_class * 1 # 测试的时候取mean


def get_alignment_loader(eeg_path, image_path, subjects, train, n_cls, 
                         repeat_times, samples_per_class, n_channels, 
                         frequence, image_tensor_path,  batch_size, workers, sub_3000):
    """
    创建数据加载器 (DataLoader)，用于训练和测试
    """
    # 初始化 AlignmentEEGDataset 类
    dataset = AlignmentEEGDataset(
        eeg_path=eeg_path,
        image_path=image_path,
        subjects=subjects,
        train=train,
        n_cls=n_cls,
        repeat_times=repeat_times,
        samples_per_class=samples_per_class,
        n_channels=n_channels,
        frequence=frequence,
        image_tensor_path=image_tensor_path,
        sub_3000=sub_3000
    )

    # 创建 DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if train else False,  # 训练时打乱数据，测试时不打乱
        num_workers=workers,
        pin_memory=True  # 将数据加载到固定内存中提高效率
    )

    return data_loader


def get_loaders(batch_size, workers, opt, config):
    """
    获取训练和测试数据加载器
    """

    if opt.sub_3000:
        n_cls = 300
    else:
        n_cls = 1654
    
    print(f"n_cls: {n_cls}, n_channels: {opt.num_channels}")
    
    # 获取训练加载器
    train_loader = get_alignment_loader(
        eeg_path=config['train']['eeg_path'],
        image_path=config['train']['image_path'],
        subjects=["sub-08"],
        train=True,
        n_cls=n_cls,
        repeat_times=config['train']['repeat_times'],
        samples_per_class=config['train']['samples_per_class'],
        n_channels=opt.num_channels,
        frequence=config['train']['frequence'],
        image_tensor_path=config['train']['image_tensor_path'],
        batch_size=batch_size,
        workers=workers,
        sub_3000=config['sub_3000']
    )

    # 获取测试加载器
    test_loader = get_alignment_loader(
        eeg_path=config['test']['eeg_path'],
        image_path=config['test']['image_path'],
        subjects=["sub-08"],
        train=False,
        n_cls=config['test']['n_cls'],
        repeat_times=config['test']['repeat_times'],
        samples_per_class=config['test']['samples_per_class'],
        n_channels=opt.num_channels,
        frequence=config['test']['frequence'],
        image_tensor_path=config['test']['image_tensor_path'],
        batch_size=1,
        workers=workers,
        sub_3000=config['sub_3000']
    )

    return train_loader, test_loader

# 使用示例
if __name__ == "__main__":
    dataset = AlignmentEEGDataset(
        eeg_path="/path/to/THINGS-EEG2/preprocessed_eeg_data",
        image_path="/path/to/THINGS-EEG2/image_filtering_dataset/training_images",
        image_tensor_path="/path/to/THINGS-EEG2/features",
        subjects=["sub-08"],
        train=True,
        n_cls=300,
        samples_per_class=10,
        repeat_times=4
    )
    
    # 测试数据加载
    eeg, img_feat = dataset[0]
    print(f"EEG shape: {eeg.shape}")        # 应输出 torch.Size([2, C, T])
    print(f"Image shape: {img_feat.shape}") # 应输出 torch.Size([4, 3, 224, 224])
    print(f"len: {len(dataset)}")

    dataset = AlignmentEEGDataset(
        eeg_path="/path/to/THINGS-EEG2/preprocessed_eeg_data",
        image_path="/path/to/THINGS-EEG2/image_filtering_dataset/test_images",
        image_tensor_path="/path/to/THINGS-EEG2/features",
        subjects=["sub-08"],
        train=False,
        n_cls=200,
        samples_per_class=1,
        repeat_times=80
    )
    
    # 测试数据加载
    eeg, img_feat = dataset[0]
    print(f"EEG shape: {eeg.shape}")        # 应输出 torch.Size([2, C, T])
    print(f"Image shape: {img_feat.shape}") # 应输出 torch.Size([4, 3, 224, 224])
    print(f"len: {len(dataset)}") 

        
    
    
        