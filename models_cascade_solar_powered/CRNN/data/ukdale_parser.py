import os
import numpy            as     np
import pandas           as     pd
from   .nilm_dataset     import NILMDataset


class UK_Dale_Parser:
    def __init__(self, args):

        self.data_location = os.path.join(args.data_path)
        self.house_indicies_train = args.house_indicies_train
        self.house_indicies_test = args.house_indicies_test
        self.appliance_names = args.appliance_names

        self.val_size = args.validation_size
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.window_stride = args.window_stride

        self.x, self.y = self.load_data(self.house_indicies_train)
        self.x_test, self.y_test = self.load_data(self.house_indicies_test)
        print('DATA SPLIT DONE!')


    def load_data(self, house_indicies):

       

        for house_id in house_indicies:
            house_data = pd.read_csv(os.path.join(self.data_location, f'nilm_solar_HOUSE_{house_id}_FINAL.csv'))
            house_data = house_data.set_index('Time')

            if house_id == house_indicies[0]:
                entire_data = house_data
                if len(house_indicies) == 1:
                    entire_data = entire_data.reset_index(drop=True)
            else:
                entire_data = entire_data.append(house_data, ignore_index=True)

        entire_data = entire_data.dropna().copy()

        return entire_data.values[:, -1:], entire_data.values[:, :-1]



    def get_datasets(self):
        val_end = int(self.val_size * len(self.x))

        # Normalization of aggregate in [-1,1]
        self.x_min = min(self.x[val_end:])
        self.x_max = max(self.x[val_end:])
        x_train =  2 * (self.x[val_end:]-self.x_min)/(self.x_max-self.x_min) - 1
        x_val = 2 * (self.x[:val_end]-self.x_min)/(self.x_max-self.x_min) - 1
        x_test = 2 * (self.x_test-self.x_min)/(self.x_max-self.x_min) - 1

        # Normalization of appliances in [0,1]
        self.y_min = np.min(self.y[val_end:], axis=0)
        self.y_max = np.max(self.y[val_end:], axis=0)
        self.y = (self.y-self.y_min)/(self.y_max-self.y_min)
        self.y_test = (self.y_test-self.y_min)/(self.y_max-self.y_min)

        val = NILMDataset(x_val, self.y[:val_end], self.seq_len, self.pred_len, self.seq_len)

        # overlapping patches for training
        train = NILMDataset(x_train, self.y[val_end:], self.seq_len, self.pred_len, self.window_stride)

        # no overlapping patches for testing
        test = NILMDataset(x_test, self.y_test, self.seq_len, self.pred_len, self.seq_len)

        return train, val, test, self.x_min, self.x_max, self.y_min, self.y_max
