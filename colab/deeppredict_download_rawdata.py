import errno
import os
from os import path

SCRIPT_DIR = path.dirname(path.realpath(__file__))

# IF downloading from codecentric S3 bucket (recommended)
url_s3 = "https://s3.eu-central-1.amazonaws.com/predictron-datasets/"
# IF downloading from csegroups (original source)
url_original = "http://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/"


def try_make_directories():
    data_directories = ["dataset"]
    for directory in data_directories:
        try_make_dir(f'{SCRIPT_DIR}/{directory}')


def try_make_dir(name):
    try:
        os.makedirs(name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def download_rawdata_files(source_of_files="S3"):
    import urllib.request

    try_make_directories()

    for map_code, map_mat in mapping_code_to_mat:
        row_code = str(map_code).strip()
        row_mat = str(map_mat).strip()
        local_filename = f'{SCRIPT_DIR}/dataset/{row_code}.mat'

        if source_of_files == "S3":
            if row_code.endswith(row_mat):
                remote_filename = f'{row_code}.mat'
            else:
                remote_filename = f'{row_code}_{row_mat}.mat'
            url = url_s3
        elif source_of_files == 'original':
            remote_filename = f'{row_mat}.mat'
            url = url_original
        else:
            print("No valid source of files specified. Exiting.")
            return

        full_url = f'{url}{remote_filename}'
        print(full_url)

        if os.path.isfile(local_filename):
            print(f'File {local_filename} already exists. Skipping.')
            continue
        print('Downloading...')
        urllib.request.urlretrieve(full_url, local_filename)
        print(f'Downloaded: {local_filename}')
    return


# As the matlab file name is just a number,
# ...the corresponding codes needed to be manually read out from the original website.
# This mapping means that, for example, the data for the state 'B007' at load '0'
# ...is contained in the 118.mat file
mapping_code_to_mat = [
    ('B007_0', '118'),
    ('B007_1', '119'),
    ('B007_2', '120'),
    ('B007_3', '121'),
    ('B014_0', '185'),
    ('B014_1', '186'),
    ('B014_2', '187'),
    ('B014_3', '188'),
    ('B021_0', '222'),
    ('B021_1', '223'),
    ('B021_2', '224'),
    ('B021_3', '225'),
    ('B028_0', '3005'),
    ('B028_1', '3006'),
    ('B028_2', '3007'),
    ('B028_3', '3008'),
    ('IR007_0', '105'),
    ('IR007_1', '106'),
    ('IR007_2', '107'),
    ('IR007_3', '108'),
    ('IR014_0', '169'),
    ('IR014_1', '170'),
    ('IR014_2', '171'),
    ('IR014_3', '172'),
    ('IR021_0', '209'),
    ('IR021_1', '210'),
    ('IR021_2', '211'),
    ('IR021_3', '212'),
    ('IR028_0', '3001'),
    ('IR028_1', '3002'),
    ('IR028_2', '3003'),
    ('IR028_3', '3004'),
    ('Normal_0', '97'),
    ('Normal_1', '98'),
    ('Normal_2', '99'),
    ('Normal_3', '100'),
    ('OR007@12_156', '156'),
    ('OR007@12_158', '158'),
    ('OR007@12_159', '159'),
    ('OR007@12_160', '160'),
    ('OR007@3_144', '144'),
    ('OR007@3_145', '145'),
    ('OR007@3_146', '146'),
    ('OR007@3_147', '147'),
    ('OR007@6_130', '130'),
    ('OR007@6_131', '131'),
    ('OR007@6_132', '132'),
    ('OR007@6_133', '133'),
    ('OR014@6_197', '197'),
    ('OR014@6_198', '198'),
    ('OR014@6_199', '199'),
    ('OR014@6_200', '200'),
    ('OR021@12_258', '258'),
    ('OR021@12_259', '259'),
    ('OR021@12_260', '260'),
    ('OR021@12_261', '261'),
    ('OR021@3_246', '246'),
    ('OR021@3_247', '247'),
    ('OR021@3_248', '248'),
    ('OR021@3_249', '249'),
    ('OR021@6_234', '234'),
    ('OR021@6_235', '235'),
    ('OR021@6_236', '236'),
    ('OR021@6_237', '237')
]
