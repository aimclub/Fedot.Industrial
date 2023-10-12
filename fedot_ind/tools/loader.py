import logging
import os
import shutil
import urllib.request as request
import zipfile
from pathlib import Path

import chardet
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sktime.datasets._data_io import load_from_tsfile_to_dataframe
from tqdm import tqdm

from fedot_ind.api.utils.path_lib import PROJECT_PATH


class DataLoader:
    """Class for reading data from ``tsv`` files and downloading from UCR archive if not found locally.
    At the moment supports only ``.txt`` and ``.arff`` formats, but not relational ``.arff`` or ``.ts`` files.

    Args:
        dataset_name: name of dataset

    Examples:
        >>> data_loader = DataLoader('ItalyPowerDemand')
        >>> train_data, test_data = data_loader.load_data()
    """

    def __init__(self, dataset_name: str, folder: str = None):
        self.logger = logging.getLogger('DataLoader')
        self.dataset_name = dataset_name
        self.folder = folder

    def load_data(self) -> tuple:
        """Load data for classification experiment locally or externally from UCR archive.

        Returns:
            tuple: train and test data
        """
        dataset_name = self.dataset_name

        data_path = os.path.join(PROJECT_PATH, 'fedot_ind', 'data') if self.folder is None else self.folder

        _, train_data, test_data = self.read_train_test_files(dataset_name=dataset_name,
                                                              data_path=data_path)

        if train_data is None:
            self.logger.info(f'Downloading...')

            # Create temporary folder for downloaded data
            cache_path = os.path.join(PROJECT_PATH, 'temp_cache/')
            download_path = cache_path + 'downloads/'
            temp_data_path = cache_path + 'temp_data/'
            filename = 'temp_data_{}'.format(dataset_name)
            for _ in (download_path, temp_data_path):
                os.makedirs(_, exist_ok=True)

            url = f"http://www.timeseriesclassification.com/aeon-toolkit/{dataset_name}.zip"
            request.urlretrieve(url, download_path + filename)
            try:
                zipfile.ZipFile(download_path + filename).extractall(temp_data_path + dataset_name)
            except zipfile.BadZipFile:
                self.logger.error(f'Cannot extract data: {dataset_name} dataset not found in UCR archive')
                return None, None

            self.logger.info(f'{dataset_name} data downloaded. Unpacking...')
            train_data, test_data = self.extract_data(dataset_name, temp_data_path)

            shutil.rmtree(cache_path)
            return train_data, test_data
        self.logger.info(f'Data readed successfully from local folder')
        return train_data, test_data

    def read_train_test_files(self, data_path, dataset_name):

        file_path = data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN'
        # If data unpacked as .tsv file
        if os.path.isfile(file_path + '.tsv'):
            self.logger.info(f'Reading data from {data_path + "/" + dataset_name}')
            x_train, y_train, x_test, y_test = self.read_tsv(dataset_name, data_path)
            is_multi = False

        # If data unpacked as .txt file
        elif os.path.isfile(file_path + '.txt'):
            self.logger.info(f'Reading data from {data_path + "/" + dataset_name}')
            x_train, y_train, x_test, y_test = self.read_txt_files(dataset_name, data_path)
            is_multi = False

        # If data unpacked as .ts file
        elif os.path.isfile(file_path + '.ts'):
            self.logger.info(f'Reading data from {data_path + "/" + dataset_name}')
            x_train, y_train, x_test, y_test = self.read_ts_files(dataset_name, data_path)
            is_multi = True

        # If data unpacked as .arff file
        elif os.path.isfile(file_path + '.arff'):
            self.logger.info(f'Reading data from {data_path + "/" + dataset_name}')
            x_train, y_train, x_test, y_test = self.read_arff_files(dataset_name, data_path)
            is_multi = True

        else:
            self.logger.error(f'Data not found in {data_path + "/" + dataset_name}')
            return None, None, None
        return is_multi, (x_train, y_train), (x_test, y_test)

    def predict_encoding(self, file_path: Path, n_lines: int = 20) -> str:
        with Path(file_path).open('rb') as f:
            rawdata = b''.join([f.readline() for _ in range(n_lines)])
        return chardet.detect(rawdata)['encoding']

    def _load_from_tsfile_to_dataframe(self, full_file_path_and_name, return_separate_X_and_y=True,
                                       replace_missing_vals_with='NaN'):
        """Loads data from a .ts file into a Pandas DataFrame.
        Taken from https://github.com/ChangWeiTan/TS-Extrinsic-Regression/blob/master/utils/data_loader.py

        Args: full_file_path_and_name: The full pathname of the .ts file to read. return_separate_X_and_y: true if X
              and Y values should be returned as separate Data Frames (X) and a numpy array (y), false otherwise.
              replace_missing_vals_with: The value that missing values in the text file should be replaced with prior to
              parsing.

        Returns: DataFrame, ndarray: If return_separate_X_and_y then a tuple containing a DataFrame and a numpy array
                 containing the relevant time-series and corresponding class values.
                 DataFrame: If not return_separate_X_and_y then a single DataFrame containing all time-series and
                 (if relevant) a column "class_vals" the associated class values.

        """

        # Initialize flags and variables used when parsing the file
        metadata_started = False
        data_started = False

        has_problem_name_tag = False
        has_timestamps_tag = False
        has_univariate_tag = False
        has_class_labels_tag = False
        has_target_labels_tag = False
        has_data_tag = False

        previous_timestamp_was_float = None
        previous_timestamp_was_int = None
        previous_timestamp_was_timestamp = None
        num_dimensions = None
        is_first_case = True
        instance_list = []
        class_val_list = []
        line_num = 0
        TsFileParseException = Exception

        encoding = self.predict_encoding(full_file_path_and_name)

        with open(full_file_path_and_name, 'r', encoding=encoding) as file:
            dataset_name = os.path.basename(full_file_path_and_name)
            for line in tqdm(file.readlines(), desc='Loading data', leave=False, postfix=dataset_name, unit='lines'):
                # print(".", end='')
                # Strip white space from start/end of line and change to lowercase for use below
                line = line.strip().lower()
                # Empty lines are valid at any point in a file
                if line:
                    # Check if this line contains metadata
                    # Please note that even though metadata is stored in this function it is not currently published externally
                    if line.startswith("@problemname"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")
                        # Check that the associated value is valid
                        tokens = line.transform_for_fit(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException("problemname tag requires an associated value")

                        problem_name = line[len("@problemname") + 1:]
                        has_problem_name_tag = True
                        metadata_started = True
                    elif line.startswith("@timestamps"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.transform_for_fit(' ')
                        token_len = len(tokens)

                        if token_len != 2:
                            raise TsFileParseException("timestamps tag requires an associated Boolean value")
                        elif tokens[1] == "true":
                            timestamps = True
                        elif tokens[1] == "false":
                            timestamps = False
                        else:
                            raise TsFileParseException("invalid timestamps value")
                        has_timestamps_tag = True
                        metadata_started = True
                    elif line.startswith("@univariate"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.transform_for_fit(' ')
                        token_len = len(tokens)
                        if token_len != 2:
                            raise TsFileParseException("univariate tag requires an associated Boolean value")
                        elif tokens[1] == "true":
                            univariate = True
                        elif tokens[1] == "false":
                            univariate = False
                        else:
                            raise TsFileParseException("invalid univariate value")

                        has_univariate_tag = True
                        metadata_started = True
                    elif line.startswith("@classlabel"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.transform_for_fit(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException("classlabel tag requires an associated Boolean value")

                        if tokens[1] == "true":
                            class_labels = True
                        elif tokens[1] == "false":
                            class_labels = False
                        else:
                            raise TsFileParseException("invalid classLabel value")

                        # Check if we have any associated class values
                        if token_len == 2 and class_labels:
                            raise TsFileParseException(
                                "if the classlabel tag is true then class values must be supplied")

                        has_class_labels_tag = True
                        class_label_list = [token.strip() for token in tokens[2:]]
                        metadata_started = True
                    elif line.startswith("@targetlabel"):
                        # Check that the data has not started
                        if data_started:
                            raise TsFileParseException("metadata must come before data")

                        # Check that the associated value is valid
                        tokens = line.transform_for_fit(' ')
                        token_len = len(tokens)

                        if token_len == 1:
                            raise TsFileParseException("targetlabel tag requires an associated Boolean value")

                        if tokens[1] == "true":
                            target_labels = True
                        elif tokens[1] == "false":
                            target_labels = False
                        else:
                            raise TsFileParseException("invalid targetLabel value")

                        has_target_labels_tag = True
                        class_val_list = []
                        metadata_started = True
                    # Check if this line contains the start of data
                    elif line.startswith("@data"):
                        if line != "@data":
                            raise TsFileParseException("data tag should not have an associated value")

                        if data_started and not metadata_started:
                            raise TsFileParseException("metadata must come before data")
                        else:
                            has_data_tag = True
                            data_started = True
                    # If the 'data tag has been found then metadata has been parsed and data can be loaded
                    elif data_started:
                        # Check that a full set of metadata has been provided
                        incomplete_regression_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_target_labels_tag or not has_data_tag
                        incomplete_classification_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_class_labels_tag or not has_data_tag
                        if incomplete_regression_meta_data and incomplete_classification_meta_data:
                            raise TsFileParseException("a full set of metadata has not been provided before the data")

                        # Replace any missing values with the value specified
                        line = line.replace("?", replace_missing_vals_with)

                        # Check if we dealing with data that has timestamps
                        if timestamps:
                            # We're dealing with timestamps so cannot just split line on ':' as timestamps may contain one
                            has_another_value = False
                            has_another_dimension = False

                            timestamps_for_dimension = []
                            values_for_dimension = []

                            this_line_num_dimensions = 0
                            line_len = len(line)
                            char_num = 0

                            while char_num < line_len:
                                # Move through any spaces
                                while char_num < line_len and str.isspace(line[char_num]):
                                    char_num += 1

                                # See if there is any more data to read in or if we should validate that read thus far

                                if char_num < line_len:

                                    # See if we have an empty dimension (i.e. no values)
                                    if line[char_num] == ":":
                                        if len(instance_list) < (this_line_num_dimensions + 1):
                                            instance_list.append([])

                                        instance_list[this_line_num_dimensions].append(pd.Series())
                                        this_line_num_dimensions += 1

                                        has_another_value = False
                                        has_another_dimension = True

                                        timestamps_for_dimension = []
                                        values_for_dimension = []

                                        char_num += 1
                                    else:
                                        # Check if we have reached a class label
                                        if line[char_num] != "(" and target_labels:
                                            class_val = line[char_num:].strip()

                                            # if class_val not in class_val_list:
                                            #     raise TsFileParseException(
                                            #         "the class value '" + class_val + "' on line " + str(
                                            #             line_num + 1) + " is not valid")

                                            class_val_list.append(float(class_val))
                                            char_num = line_len

                                            has_another_value = False
                                            has_another_dimension = False

                                            timestamps_for_dimension = []
                                            values_for_dimension = []

                                        else:

                                            # Read in the data contained within the next tuple

                                            if line[char_num] != "(" and not target_labels:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " does not start with a '('")

                                            char_num += 1
                                            tuple_data = ""

                                            while char_num < line_len and line[char_num] != ")":
                                                tuple_data += line[char_num]
                                                char_num += 1

                                            if char_num >= line_len or line[char_num] != ")":
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " does not end with a ')'")

                                            # Read in any spaces immediately after the current tuple

                                            char_num += 1

                                            while char_num < line_len and str.isspace(line[char_num]):
                                                char_num += 1

                                            # Check if there is another value or dimension to process after this tuple

                                            if char_num >= line_len:
                                                has_another_value = False
                                                has_another_dimension = False

                                            elif line[char_num] == ",":
                                                has_another_value = True
                                                has_another_dimension = False

                                            elif line[char_num] == ":":
                                                has_another_value = False
                                                has_another_dimension = True

                                            char_num += 1

                                            # Get the numeric value for the tuple by reading from the end of the tuple data backwards to the last comma

                                            last_comma_index = tuple_data.rfind(',')

                                            if last_comma_index == -1:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains a tuple that has no comma inside of it")

                                            try:
                                                value = tuple_data[last_comma_index + 1:]
                                                value = float(value)

                                            except ValueError:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains a tuple that does not have a valid numeric value")

                                            # Check the type of timestamp that we have

                                            timestamp = tuple_data[0: last_comma_index]

                                            try:
                                                timestamp = int(timestamp)
                                                timestamp_is_int = True
                                                timestamp_is_timestamp = False
                                            except ValueError:
                                                timestamp_is_int = False

                                            if not timestamp_is_int:
                                                try:
                                                    timestamp = float(timestamp)
                                                    timestamp_is_float = True
                                                    timestamp_is_timestamp = False
                                                except ValueError:
                                                    timestamp_is_float = False

                                            if not timestamp_is_int and not timestamp_is_float:
                                                try:
                                                    timestamp = timestamp.strip()
                                                    timestamp_is_timestamp = True
                                                except ValueError:
                                                    timestamp_is_timestamp = False

                                            # Make sure that the timestamps in the file (not just this dimension or case) are consistent

                                            if not timestamp_is_timestamp and not timestamp_is_int and not timestamp_is_float:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains a tuple that has an invalid timestamp '" + timestamp + "'")

                                            if previous_timestamp_was_float is not None and previous_timestamp_was_float and not timestamp_is_float:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                            if previous_timestamp_was_int is not None and previous_timestamp_was_int and not timestamp_is_int:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                            if previous_timestamp_was_timestamp is not None and previous_timestamp_was_timestamp and not timestamp_is_timestamp:
                                                raise TsFileParseException(
                                                    "dimension " + str(
                                                        this_line_num_dimensions + 1) + " on line " + str(
                                                        line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                            # Store the values

                                            timestamps_for_dimension += [timestamp]
                                            values_for_dimension += [value]

                                            #  If this was our first tuple then we store the type of timestamp we had

                                            if previous_timestamp_was_timestamp is None and timestamp_is_timestamp:
                                                previous_timestamp_was_timestamp = True
                                                previous_timestamp_was_int = False
                                                previous_timestamp_was_float = False

                                            if previous_timestamp_was_int is None and timestamp_is_int:
                                                previous_timestamp_was_timestamp = False
                                                previous_timestamp_was_int = True
                                                previous_timestamp_was_float = False

                                            if previous_timestamp_was_float is None and timestamp_is_float:
                                                previous_timestamp_was_timestamp = False
                                                previous_timestamp_was_int = False
                                                previous_timestamp_was_float = True

                                            # See if we should add the data for this dimension

                                            if not has_another_value:
                                                if len(instance_list) < (this_line_num_dimensions + 1):
                                                    instance_list.append([])

                                                if timestamp_is_timestamp:
                                                    timestamps_for_dimension = pd.DatetimeIndex(
                                                        timestamps_for_dimension)

                                                instance_list[this_line_num_dimensions].append(
                                                    pd.Series(index=timestamps_for_dimension,
                                                              data=values_for_dimension))
                                                this_line_num_dimensions += 1

                                                timestamps_for_dimension = []
                                                values_for_dimension = []

                                elif has_another_value:
                                    raise TsFileParseException(
                                        "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                            line_num + 1) + " ends with a ',' that is not followed by another tuple")

                                elif has_another_dimension and target_labels:
                                    raise TsFileParseException(
                                        "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                            line_num + 1) + " ends with a ':' while it should list a class value")

                                elif has_another_dimension and not target_labels:
                                    if len(instance_list) < (this_line_num_dimensions + 1):
                                        instance_list.append([])

                                    instance_list[this_line_num_dimensions].append(pd.Series(dtype=np.float32))
                                    this_line_num_dimensions += 1
                                    num_dimensions = this_line_num_dimensions

                                # If this is the 1st line of data we have seen then note the dimensions

                                if not has_another_value and not has_another_dimension:
                                    if num_dimensions is None:
                                        num_dimensions = this_line_num_dimensions

                                    if num_dimensions != this_line_num_dimensions:
                                        raise TsFileParseException("line " + str(
                                            line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                            # Check that we are not expecting some more data, and if not, store that processed above

                            if has_another_value:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ',' that is not followed by another tuple")

                            elif has_another_dimension and target_labels:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ':' while it should list a class value")

                            elif has_another_dimension and not target_labels:
                                if len(instance_list) < (this_line_num_dimensions + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dimensions].append(pd.Series())
                                this_line_num_dimensions += 1
                                num_dimensions = this_line_num_dimensions

                            # If this is the 1st line of data we have seen then note the dimensions

                            if not has_another_value and num_dimensions != this_line_num_dimensions:
                                raise TsFileParseException("line " + str(
                                    line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                            # Check if we should have class values, and if so that they are contained in those listed in the metadata

                            if target_labels and len(class_val_list) == 0:
                                raise TsFileParseException("the cases have no associated class values")
                        else:
                            dimensions = line.transform_for_fit(":")
                            # If first row then note the number of dimensions (that must be the same for all cases)
                            if is_first_case:
                                num_dimensions = len(dimensions)

                                if target_labels:
                                    num_dimensions -= 1

                                for dim in range(0, num_dimensions):
                                    instance_list.append([])
                                is_first_case = False

                            # See how many dimensions that the case whose data in represented in this line has
                            this_line_num_dimensions = len(dimensions)

                            if target_labels:
                                this_line_num_dimensions -= 1

                            # All dimensions should be included for all series, even if they are empty
                            if this_line_num_dimensions != num_dimensions:
                                raise TsFileParseException("inconsistent number of dimensions. Expecting " + str(
                                    num_dimensions) + " but have read " + str(this_line_num_dimensions))

                            # Process the data for each dimension
                            for dim in range(0, num_dimensions):
                                dimension = dimensions[dim].strip()

                                if dimension:
                                    data_series = dimension.transform_for_fit(",")
                                    data_series = [float(i) for i in data_series]
                                    instance_list[dim].append(pd.Series(data_series))
                                else:
                                    instance_list[dim].append(pd.Series())

                            if target_labels:
                                class_val_list.append(float(dimensions[num_dimensions].strip()))

                line_num += 1

        # Check that the file was not empty
        if line_num:
            # Check that the file contained both metadata and data
            complete_regression_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_target_labels_tag and has_data_tag
            complete_classification_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_class_labels_tag and has_data_tag

            if metadata_started and not complete_regression_meta_data and not complete_classification_meta_data:
                raise TsFileParseException("metadata incomplete")
            elif metadata_started and not data_started:
                raise TsFileParseException("file contained metadata but no data")
            elif metadata_started and data_started and len(instance_list) == 0:
                raise TsFileParseException("file contained metadata but no data")

            # Create a DataFrame from the data parsed above
            data = pd.DataFrame(dtype=np.float32)

            for dim in range(0, num_dimensions):
                data['dim_' + str(dim)] = instance_list[dim]

            # Check if we should return any associated class labels separately

            if target_labels:
                if return_separate_X_and_y:
                    return data, np.asarray(class_val_list)
                else:
                    data['class_vals'] = pd.Series(class_val_list)
                    return data
            else:
                return data
        else:
            raise TsFileParseException("empty file")

    def read_tsv(self, dataset_name: str, data_path: str) -> tuple:
        """Read ``tsv`` file that contains data for classification experiment. Data must be placed
        in ``data`` folder with ``.tsv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to temporary folder with downloaded data
        Returns:
            tuple: (x_train, x_test) and (y_train, y_test)

        """
        df_train = pd.read_csv(os.path.join(data_path, dataset_name, f'{dataset_name}_TRAIN.tsv'),
                               sep='\t',
                               header=None)

        x_train = df_train.iloc[:, 1:]
        y_train = df_train[0].values

        df_test = pd.read_csv(os.path.join(data_path, dataset_name, f'{dataset_name}_TEST.tsv'),
                              sep='\t',
                              header=None)

        x_test = df_test.iloc[:, 1:]
        y_test = df_test[0].values
        try:
            y_train, y_test = y_train.astype(int), y_test.astype(int)
        except ValueError:
            y_train, y_test = y_train.astype(str), y_test.astype(str)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def read_txt_files(dataset_name: str, temp_data_path: str):
        """
        Reads data from ``.txt`` file.

        Args:
            dataset_name: name of dataset
            temp_data_path: path to temporary folder with downloaded data

        Returns:
            train and test data tuple
        """
        data_train = np.genfromtxt(temp_data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN.txt')
        data_test = np.genfromtxt(temp_data_path + '/' + dataset_name + f'/{dataset_name}_TEST.txt')
        x_train, y_train = data_train[:, 1:], data_train[:, 0]
        x_test, y_test = data_test[:, 1:], data_test[:, 0]
        return x_train, y_train, x_test, y_test

    def read_ts_files(self, dataset_name, data_path):
        try:
            x_test, y_test = load_from_tsfile_to_dataframe(data_path + '/' + dataset_name + f'/{dataset_name}_TEST.ts',
                                                           return_separate_X_and_y=True)
            x_train, y_train = load_from_tsfile_to_dataframe(
                data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN.ts',
                return_separate_X_and_y=True)
            return x_train, y_train, x_test, y_test
        except Exception:
            x_test, y_test = self._load_from_tsfile_to_dataframe(
                data_path + '/' + dataset_name + f'/{dataset_name}_TEST.ts',
                return_separate_X_and_y=True)
            x_train, y_train = self._load_from_tsfile_to_dataframe(
                data_path + '/' + dataset_name + f'/{dataset_name}_TRAIN.ts',
                return_separate_X_and_y=True)
            return x_train, y_train, x_test, y_test

    def read_arff_files(self, dataset_name, temp_data_path):
        """Reads data from ``.arff`` file.

        """
        train = loadarff(temp_data_path + dataset_name + f'/{dataset_name}_TRAIN.arff')
        test = loadarff(temp_data_path + dataset_name + f'/{dataset_name}_TEST.arff')

        data_train = np.asarray([train[0][name] for name in train[1].names()])
        x_train = data_train[:-1].T.astype('float64')
        y_train = data_train[-1]

        data_test = np.asarray([test[0][name] for name in test[1].names()])
        x_test = data_test[:-1].T.astype('float64')
        y_test = data_test[-1]
        return x_train, y_train, x_test, y_test

    def extract_data(self, dataset_name: str, data_path: str):
        """Unpacks data from downloaded file and saves it into Data folder with ``.tsv`` extension.

        Args:
            dataset_name: name of dataset
            data_path: path to folder downloaded data

        Returns:
            tuple: train and test data

        """
        try:
            is_multi, (x_train, y_train), (x_test, y_test) = self.read_train_test_files(data_path, dataset_name)

        except Exception as e:
            self.logger.error(f'Error while unpacking data: {e}')
            return None, None

        # Conversion of target values to int or str
        try:
            y_train = y_train.astype('int64')
            y_test = y_test.astype('int64')
        except ValueError:
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)

        # Save data to tsv files
        new_path = os.path.join(PROJECT_PATH, 'fedot_ind','data', dataset_name)
        os.makedirs(new_path, exist_ok=True)

        self.logger.info(f'Saving {dataset_name} data files')
        for subset in ('TRAIN', 'TEST'):
            if not is_multi:
                df = pd.DataFrame(x_train if subset == 'TRAIN' else x_test)
                df.insert(0, 'class', y_train if subset == 'TRAIN' else y_test)
                df.to_csv(os.path.join(new_path, f'{dataset_name}_{subset}.tsv'), sep='\t', index=False, header=False)
                del df

            else:
                old_path = os.path.join(data_path, dataset_name, f'{dataset_name}_{subset}.ts')
                shutil.move(old_path, new_path)

        if is_multi:
            return (x_train, y_train), (x_test, y_test)
        else:
            return (pd.DataFrame(x_train), y_train), (pd.DataFrame(x_test), y_test)


if __name__ == '__main__':
    data_loader = DataLoader('Car')
    _train_data, _test_data = data_loader.load_data()
