import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


def training(dataset_train):
    # select the data
    data_train = dataset_train.iloc[:, :dataset_train.columns.size - 4]
    # Select the targets
    target_train1 = dataset_train.iloc[:, dataset_train.columns.size - 1] * 1
    target_train2 = dataset_train.iloc[:, dataset_train.columns.size - 2] * 1
    target_train3 = dataset_train.iloc[:, dataset_train.columns.size - 3] * 1
    target_train4 = dataset_train.iloc[:, dataset_train.columns.size - 4] * 1

    # Models Initialisation (by default number of layer is 5,which can be changed to any number of your choice)
    ann1 = MLPClassifier(hidden_layer_sizes=5)
    ann2 = MLPClassifier(hidden_layer_sizes=5)
    ann3 = MLPClassifier(hidden_layer_sizes=5)
    ann4 = MLPClassifier(hidden_layer_sizes=5)

    # fit the models to our data
    ann1.fit(data_train, target_train1)
    ann2.fit(data_train, target_train2)
    ann3.fit(data_train, target_train3)
    ann4.fit(data_train, target_train4)

    return [ann1, ann2, ann3, ann4]


def testing(dataset_test, trained_models):
    # select the data
    data_test = dataset_test.iloc[:, :dataset_test.columns.size - 4]

    # Select the targets
    target_test1 = dataset_test.iloc[:, dataset_test.columns.size - 1] * 1
    target_test2 = dataset_test.iloc[:, dataset_test.columns.size - 2] * 1
    target_test3 = dataset_test.iloc[:, dataset_test.columns.size - 3] * 1
    target_test4 = dataset_test.iloc[:, dataset_test.columns.size - 4] * 1

    # predict the outputs
    model1, model2, model3, model4 = trained_models

    output_test1 = model1.predict(data_test)
    output_test2 = model2.predict(data_test)
    output_test3 = model3.predict(data_test)
    output_test4 = model4.predict(data_test)

    # check the precision of the models (100 can be changed to any column of your choice)
    num_exemples = data_test.shape[0]

    P1 = np.sum(output_test1 == target_test1) * 100 / num_exemples
    P2 = np.sum(output_test2 == target_test2) * 100 / num_exemples
    P3 = np.sum(output_test3 == target_test3) * 100 / num_exemples
    P4 = np.sum(output_test4 == target_test4) * 100 / num_exemples

    print(f'P1={P1}%,', f'P={P2}%,', f'P3={P3}%,', f'P4={P4}%', sep=' ')


def check_validity(dataset, trained_models):
    # predicts the output
    model1, model2, model3, model4 = trained_models

    output1 = model1.predict(dataset)
    output2 = model2.predict(dataset)
    output3 = model3.predict(dataset)
    output4 = model4.predict(dataset)

    # detect the potential errors
    error1 = output1.sum() == 0
    error2 = output2.sum() == 0
    error3 = output3.sum() == 0
    error4 = output4.sum() == 0

    # check if the dataset is valid
    if not (error1 == error2 == error3 == error4 == False):
        print('valid dataset !')
    else:
        print('invalid dataset ! \n\n', 'Potential errors: \n')

        if not error1:
            lines_error1 = np.argwhere(output1 == 1) + 1
            print(f'\t completeness at line(s) -> {lines_error1.ravel()}', end='\n\n')

        if not error2:
            lines_error2 = np.argwhere(output2 == 1) + 1
            print(f'\t accuracy at line(s) -> {lines_error2.ravel()}', end='\n\n')

        if not error3:
            lines_error3 = np.argwhere(output3 == 1) + 1
            print(f'\t inconsistence at line(s) -> {lines_error3.ravel()}', end='\n\n')

        if not error4:
            lines_error4 = np.argwhere(output4 == 1) + 1
            print(f'\t integrity at line(s) -> {lines_error1.ravel()}', end='\n\n')


def transform_data(list_files_path, max_num_feat=100, NaN_rep_val=-100, for_=None, contain_targets=None):
    if for_ == None:
        raise (
            Exception("argument 'for_' not specified; Please specify if the dataset(s) is for training of testing !"))
    if for_ in [1, 'test', 'testing'] and (len(list_files_path) > 1):
        raise (Exception("length of list_files_path > 1; Only take one dataset for test!"))
    if contain_targets == None:
        raise (Exception("contain_targets argument is None; Please precise if the dataset(s) contain(s) targets !"))

    excel_extensions = ['xltx', 'xls', 'xlsm', 'xlw', 'xml', 'xlt', 'xlam', 'xlsx', 'xla', 'xlsb', 'xltm', 'xlr']
    csv_extensions = ['csv', 'csv2']
    list_dataset = []
    final_dataset = None

    # load dataset with respect to their file format
    for i, file_path in enumerate(list_files_path):
        print(f'{(i + 1) * 100 / len(list_files_path) :.2f}%', sep=' ')
        if file_path.split('.')[-1] in excel_extensions:
            dataset = pd.read_excel(file_path)
            list_dataset.append(dataset)

        if file_path.split('.')[-1] in csv_extensions:
            dataset = pd.read_csv(file_path, engine='python')
            list_dataset.append(dataset)

    # combine the dataset
    for i, dataset in enumerate(list_dataset):

        # remove extra columns
        dataset.dropna(how='all', axis=1, inplace=True)
        dataset.fillna(NaN_rep_val, inplace=True)
        dataset.columns = range(dataset.shape[1])

        # factorize columns with string d type
        for column_id in range(dataset.shape[1]):
            columnn = np.array(dataset.iloc[:, column_id])
            if not np.issubdtype(columnn.dtype, np.number):
                labels, lavels = pd.factorize(pd.Series(columnn))
                # print(labels)
                dataset.iloc[:, column_id] = labels

        # complete the number of columns to the maximum
        if contain_targets in ['yes', 'y', 'Yes', 'YES', 'Y', True]:
            completing_dataset = pd.DataFrame(np.ones((dataset.shape[0], max_num_feat - dataset.shape[1])))
            targets_df = dataset.iloc[:, dataset.shape[1] - 4:]
            data_df = dataset.iloc[:, :dataset.shape[1] - 4]
            completed_dataset = pd.concat([data_df, completing_dataset, targets_df], axis=1, ignore_index=True)
            # change the dataset to its completed version
            list_dataset[i] = completed_dataset
        else:
            completing_dataset = pd.DataFrame(np.ones((dataset.shape[0], max_num_feat - dataset.shape[1] - 4)))
            completed_dataset = pd.concat([dataset, completing_dataset], axis=1, ignore_index=True)
            # change the dataset to its completed version
            list_dataset[i] = completed_dataset

    if (for_ in [0, 'train', 'traning']) and (len(list_dataset) > 1):
        final_dataset = pd.concat(list_dataset, ignore_index=True)
    if (for_ in [0, 'train', 'traning']) and (len(list_dataset) == 1):
        final_dataset = list_dataset[0]
    if for_ in [1, 'test', 'testing', 'generalization', 'gen'] and (len(list_dataset) == 1):
        final_dataset = list_dataset[0]

    final_dataset.fillna(1., inplace=True)

    return final_dataset
