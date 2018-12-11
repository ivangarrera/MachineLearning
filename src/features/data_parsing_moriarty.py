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
        self.data_set["ActionType"] = 0

        indexes = []
        for index in range(len(self.data_set_moriarty["UUID"])):
            self.merged = self.data_set.query("UUID <= " + \
                                     str(self.data_set_moriarty["UUID"][index]) +\
                                     " <= UUID + 5000")
            if not self.merged.empty:
                self.data_set_moriarty["UUID"][index] = self.merged["UUID"].values[0]
                index = self.data_set["UUID"].values.tolist().index(self.merged["UUID"].values[0])
                indexes.append(index)
                self.data_set["SessionType"][index] = 1
                    
    def MergedWithNumericColumns(self, characteristics):
        self.merged_numeric = self.merged[characteristics]
    
    def GetTargets(self):
        target = []
        for i in range(len(self.merged["SessionType"])):
            if "malicious" in self.merged["SessionType"][i]:
                target.append(1)
            else:
                target.append(0)
        return target