import pandas as pd
import pickle
import os
from src.utils.context import Context


class Data:
    def __init__(self, ctx):
        self.context = ctx
        self.config = ctx.config
        self.logger = ctx.logger

        self.data = self.config['data']
        self.suffix = self.config['suffix']
        self.sep = self.config['sep']
        self.ext = self.config['ext']
        self.drop = [] if self.config['drop'] is None else self.config['drop']
        self.cat_names = [] if self.config['cat_names'] is None else self.config['cat_names']
        self.to_disk = self.config['to_disk']
        self.target = self.config['target']
        self.data_basepath = f'{self.context.project_dir}/data/{self.data}'

    def preprocess(self):
        source_fname = f'{self.data_basepath}/{self.data}{self.suffix}.{self.ext}'
        target_fname = f'{self.data_basepath}/{self.data}.{self.ext}'

        self.logger.info(f'Loading Source file: {source_fname}')
        df = pd.read_csv(source_fname, sep=self.sep)
        df = df.drop(self.drop, axis=1)
        df_num, subs = self.category_to_number(df)

        if self.target is not None:
            target_fname_y = f'{self.data_basepath}/{self.data}_labels.{self.ext}'
            y = df_num[self.target]
            df_num = df_num.drop([self.target], axis=1)
            subs['table_colums_name']['label'] = subs['table_colums_name']['label'].drop([self.target])
        else:
            y = None

        subs_frame = f'{self.data_basepath}/subs.pkl'
        self.logger.info(f'Save subs file: {subs_frame}')
        pickle.dump(subs, open(f'{subs_frame}', 'wb'))


        if self.to_disk:
            if self.target is not None:
                self.logger.info(f'Save Target file label: {target_fname_y}')
                y.to_csv(target_fname_y, sep=self.sep, index=False)
            self.logger.info(f'Save Target file: {target_fname}')
            df_num.to_csv(target_fname, sep=self.sep, index=False)

        return df_num, y, subs

    def category_to_number(self, df):
        subs = {}
        df_num = df.copy()

        subs['table_colums_name'] = {'y':[], 'label': df_num.columns}

        # TRANSFORM TO SET TO PREVENT DOUBLE FACTORIZATION
        for z in set(df_num.select_dtypes(include=['object']).columns.tolist() + self.cat_names):
            y, label = pd.factorize(df[z])
            subs[z] = {'y': y, 'label': label}
            df_num[z] = y
        return df_num, subs

    def postprocess(self):
        pass


if __name__ == "__main__":

    ctx = Context("data")
    engine = Data(ctx)
    engine.preprocess()
