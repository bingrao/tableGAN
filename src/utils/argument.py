from argparse import ArgumentParser


def config_opts(parser):
    parser.add_argument('-project_dir', '--project_dir', required=False, type=str, default='')
    parser.add_argument('-config', '--config', required=False, help='Config file path')
    parser.add_argument('-project_log', '--project_log', type=str, default='')
    parser.add_argument('-debug', '--debug', type=bool, default=False)
    parser.add_argument('-data', '--data', type=str, required=False, default='Civilian')


def data_opts(parser):
    parser.add_argument('-suffix', '--suffix', type=str, default='_textual')
    parser.add_argument('-sep', '--sep', type=str, default=',')
    parser.add_argument('-ext', '--ext', type=str, default='csv')
    parser.add_argument('-drop', '--drop', type=list, default=None)
    parser.add_argument('-cat_names', '--cat_names', type=list, default='')
    parser.add_argument('-to_disk', '--to_disk', type=bool, default=True)
    #
    parser.add_argument('-target', '--target', type=str, default=None,
                        help='Ticket <--> OpCarrierGroup; Civilian <--> suicide')
    parser.add_argument('-d_basepath', '--d_basepath', type=str, default='data')


def train_opts(parser):
    parser.add_argument('-nums_server', '--nums_server', type=int, default=5)


def test_opts(parser):
    parser.add_argument('-nums_server', '--nums_server', type=int, default=5)


def generate_opts(parser):
    parser.add_argument('-nums_server', '--nums_server', type=int, default=5)


def get_default_argument(desc='default'):
    parser = ArgumentParser(description=desc)
    config_opts(parser)
    if desc == 'data':
        data_opts(parser)
    elif desc == 'train':
        train_opts(parser)
    elif desc == 'test':
        test_opts(parser)
    elif desc == 'generate':
        generate_opts(parser)
    else:
        train_opts(parser)
    args = parser.parse_args()
    config = vars(args)
    return config
