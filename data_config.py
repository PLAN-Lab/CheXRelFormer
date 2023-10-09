
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'CXRData':
            self.root_dir = '/home/amarachi/CheXRelFormer/datasets/CXRData'
    
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='CXRData')
    print(data.data_name)
    print(data.root_dir)


