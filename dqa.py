import package

# the default parameters here are:
# - maximum num of features -> max_num_feat = 100
# - the value used to replace the Not a Number -> NaN_rep_val = -100

# load datasets and preprocess training
prepro_dataset_train = package.transform_data(list_files_path=[
    "C:\\Users\\admin\\PycharmProjects\\Thesis_code\\Manchester United 2018-19.xlsx"],
                                              for_='train', contain_targets=True)


trained_models = package.training(prepro_dataset_train)

# load dataset and preprocess for test
prepro_dataset_test = package.transform_data(list_files_path=[
    "C:\\Users\\admin\\PycharmProjects\\Thesis_code\\cluster_A\\county_statistics.csv"],
                                             for_='test', contain_targets=True)

package.testing(prepro_dataset_test, trained_models)

# load dataset and preprocess for generalization
prepro_dataset_gen = package.transform_data(list_files_path=[
    "C:\\Users\\admin\\PycharmProjects\\Thesis_code\\cluster_A\\county_statistics.csv"],
                                            for_='gen', contain_targets=False)

package.check_validity(prepro_dataset_gen, trained_models)
