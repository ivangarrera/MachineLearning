import pandas as pd

class ParsingMoriarty:
    def __init__(self, data_path, moriarty_path):
        self.data_set = pd.read_csv(data_path, sep=',')
        self.data_set_moriarty = pd.read_csv(moriarty_path, sep=',', error_bad_lines=False)
        self.merged = None
        self.merged_numeric = None
        self.attr = ''
    
    def MergeDataByUUID(self, characteristics):
        self.data_set = self.data_set.loc[:, self.data_set.columns.str.contains('|'.join(characteristics))]
        self.data_set["SessionType"] = 0

        for index in range(len(self.data_set_moriarty["UUID"])):
            # Find when the attack occurred and include the attack in the dataset
            self.merged = self.data_set.query("UUID <= " + \
                                     str(self.data_set_moriarty["UUID"][index]) +\
                                     " <= UUID + 5000")
            if not self.merged.empty:
                self.data_set_moriarty["UUID"][index] = self.merged["UUID"].values[0]
                
                # Get the index of the corresponding attack inside the dataset 
                # and set its attack column to a true value
                index = self.data_set["UUID"].values.tolist().index(self.merged["UUID"].values[0])
                self.data_set["SessionType"][index] = 1
                    
    def MergedWithNumericColumns(self, characteristics):
        self.merged_numeric = self.merged[characteristics]
    
    def CreateSupervisedDataset(self, number_of_non_attacks):
        mydataset = pd.DataFrame(data=None, columns=self.data_set.columns)

        # Select non-attacks and include them into the dataset
        for i in range(300):
            index = int(i * len(self.data_set) / number_of_non_attacks)
            frames = [mydataset, self.data_set.iloc[[index]]]
            mydataset = pd.concat(frames)
        
        # Select attacks and include them into the dataset
        for i in range(len(self.data_set)):
            if self.data_set["SessionType"][i] == 1:
                frames = [mydataset, self.data_set.iloc[[i]]]
                mydataset = pd.concat(frames)
        return mydataset
    