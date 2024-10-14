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

        #for appliance in self.appliance_names:
        #    assert appliance in ['active', 'solar', 'kettle', 'microwave', 'fridge', 'washing machine', 'dish washer']

        for house_id in house_indicies:
            house_data = pd.read_csv(os.path.join(self.data_location, f'nilm_solar_HOUSE_{house_id}_FINAL.csv'))
            if house_id != 1 and house_id != 5:
                house_data = house_data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.1.1'])
            else:
                house_data = house_data.drop(columns=['Unnamed: 0.1'])
            house_data['Unnamed: 0'] = pd.to_datetime(house_data['Unnamed: 0'])
            house_data = house_data.rename(columns={'Unnamed: 0': 'Time'})
            house_data = house_data.set_index('Time')

            if house_id == house_indicies[0]:
                entire_data = house_data
                if len(house_indicies) == 1:
                    entire_data = entire_data.reset_index(drop=True)
            else:
                entire_data = entire_data.append(house_data, ignore_index=True)

        entire_data = entire_data.dropna().copy()
        # ritorna l'input che è il net load e l'output come lista di solar + nilm
        return entire_data.values[:, -1:], [entire_data.values[:, 1:2], np.concatenate([entire_data.values[:, :1],entire_data.values[:,2:-1]], axis=1)]
        #return entire_data.values[:, -1:], [entire_data.values[:, 1:2], entire_data.values[:, 2:-1]]



    def get_datasets(self):
        val_end = int(self.val_size * len(self.x))

        # Normalization of aggregate in [-1,1]
        self.x_min = min(self.x[val_end:])
        self.x_max = max(self.x[val_end:])
        x_train =  2 * (self.x[val_end:]-self.x_min)/(self.x_max-self.x_min) - 1
        x_val = 2 * (self.x[:val_end]-self.x_min)/(self.x_max-self.x_min) - 1
        x_test = 2 * (self.x_test-self.x_min)/(self.x_max-self.x_min) - 1

        # Normalization of appliances in [0,1]
        self.y_min_solar = np.min(self.y[0][val_end:], axis=0)
        self.y_min_nilm = np.min(self.y[1][val_end:], axis=0)
        self.y_max_solar = np.max(self.y[0][val_end:], axis=0)
        self.y_max_nilm = np.max(self.y[1][val_end:], axis=0)

        self.y[0] = (self.y[0] -self.y_min_solar)/(self.y_max_solar -self.y_min_solar)
        self.y[1] = (self.y[1] - self.y_min_nilm) / (self.y_max_nilm - self.y_min_nilm)
        self.y_test[0] = (self.y_test[0]-self.y_min_solar)/(self.y_max_solar-self.y_min_solar)
        self.y_test[1] = (self.y_test[1] - self.y_min_nilm) / (self.y_max_nilm - self.y_min_nilm)

        val = NILMDataset(x_val, [self.y[0][:val_end],self.y[1][:val_end]], self.seq_len, self.pred_len, self.seq_len)

        # overlapping patches for training
        train = NILMDataset(x_train, [self.y[0][val_end:],self.y[1][val_end:]], self.seq_len, self.pred_len, self.window_stride)

        # no overlapping patches for testing
        test = NILMDataset(x_test, [self.y_test[0],self.y_test[1]], self.seq_len, self.pred_len, self.seq_len)

        return train, val, test, self.x_min, self.x_max, [self.y_min_solar, self.y_min_nilm], [self.y_max_solar, self.y_max_nilm]
