import package

# the default parameters here are:
# - maximum num of features -> max_num_feat = 100
# - the value used to replace the Not a Number -> NaN_rep_val = -100

# load dataset and preprocess training
prepared_dataset_train = package.transform_data(list_files_path=[
    "Pre configured Dataset path for training"],
    for_='train', contain_targets=True)

trained_models = package.training(prepared_dataset_train)

# load dataset and preprocess for test
prepared_dataset_test = package.transform_data(list_files_path=[
    "Preconfigured dataset path for testing"],
    for_='test', contain_targets=True)

package.testing(prepared_dataset_test, trained_models)

# load dataset and preprocess for generalization
prepared_dataset_gen = package.transform_data(list_files_path=[
    "Generalization Dataset Path"],
    for_='gen', contain_targets=False)

prediction_sizes = package.check_validity(prepared_dataset_gen, trained_models)
package.errors_vs_success_plot(prediction_sizes)
