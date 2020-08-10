import pandas as pd
import pickle
import os


def transform(ds: str, suffix: str = '', sep: str = ';', ext: str = 'csv', drop: list = None, cat_names: list = None,
              to_disk: bool = True, target: str = '', d_basepath='data'):
    if target == '':  target = False
    if drop is None: drop = []
    if cat_names is None: cat_names = []

    base_fname = f'{d_basepath}/{ds}/{ds}'
    source_fname = f'{base_fname}{suffix}.{ext}'
    print(f'Basepath: {base_fname}')
    print(f'Source file: {source_fname}')
    fname = os.path.basename(os.path.splitext(source_fname)[0])

    df = pd.read_csv(source_fname, sep=sep)
    df = df.drop(drop, axis=1)
    df_num, subs = cat_to_num(df, cat_names=cat_names)
    pickle.dump(subs, open(f'{d_basepath}/{ds}/subs.pkl', 'wb'))

    if target:
        y = df_num[target]
        df_num = df_num.drop([target], axis=1)

    if to_disk:
        if target:
            target_fname_y = f'{base_fname}_labels.csv'
            print(f'Target file label: {target_fname_y}')
            y.to_csv(target_fname_y, sep=';', index=False)
        target_fname = f'{base_fname}.csv'
        print(f'Target file: {target_fname}')
        df_num.to_csv(target_fname, sep=';', index=False)
    if target:
        return df_num, y, subs
    return df_num, subs


def cat_to_num(df, sep=',', cat_names=None):
    if cat_names is None: cat_names = []
    subs = {}
    df_num = df.copy()

    # TRANSFORM TO SET TO PREVENT DOUBLE FACTORIZATION
    for z in set(df_num.select_dtypes(include=['object']).columns.tolist() + cat_names):
        y, label = pd.factorize(df[z])
        subs[z] = {'y': y, 'label': label}
        df_num[z] = y
    return df_num, subs


if __name__ == "__main__":
    a, b, c = transform('Ticket', suffix='_textual', to_disk=True, target='OpCarrierGroup')
    a.head()