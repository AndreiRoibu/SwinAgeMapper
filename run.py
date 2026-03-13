import os
import shutil
import argparse
import logging

import torch
import torch.utils.data as data
import numpy as np

from solver import Solver
from utils.data_utils import get_datasets_dynamically, get_test_datasets_dynamically
from utils.settings import Settings
import utils.data_evaluation as evaluations

from SwinAgeMapper import SwinAgeMapper


# Set the default floating point tensor type to FloatTensor
torch.set_default_tensor_type(torch.FloatTensor)


def load_data_dynamically(data_parameters, mapping_evaluation_parameters=None, flag='train'):
    
    if flag=='train':
        print("Data is loading...")
        train_data, validation_data, resolution = get_datasets_dynamically(data_parameters)
        print("Data has loaded!")
        print("Training dataset size is {}".format(len(train_data)))
        print("Validation dataset size is {}".format(len(validation_data)))
        return train_data, validation_data, resolution
    elif flag=='test':
        print("Data is loading...")
        test_data, volumes_to_be_used, prediction_output_statistics_name, resolution = get_test_datasets_dynamically(data_parameters, mapping_evaluation_parameters)
        print("Data has loaded!")
        len_test_data = len(test_data)
        print("Testing dataset size is {}".format(len_test_data))
        return test_data, volumes_to_be_used, prediction_output_statistics_name, len_test_data, resolution
    else:
        print('ERROR: Invalid Flag')
        return None


def train(data_parameters, training_parameters, network_parameters, misc_parameters):

    if training_parameters['optimiser'] == 'adamw':
        optimizer = torch.optim.AdamW
    elif training_parameters['optimiser'] == 'adam':
        optimizer = torch.optim.Adam
    else:
        optimizer = torch.optim.AdamW # Default option

    optimizer_arguments={'lr': training_parameters['learning_rate'],
                        'betas': training_parameters['optimizer_beta'],
                        'eps': training_parameters['optimizer_epsilon'],
                        'weight_decay': training_parameters['optimizer_weigth_decay']
                        }

    if training_parameters['loss_function'] == 'mse':
        loss_function = torch.nn.MSELoss()
    elif training_parameters['loss_function'] == 'mae':
        loss_function = torch.nn.L1Loss()
    else:
        print("Loss function not valid. Defaulting to MSE!")
        loss_function = torch.nn.MSELoss()

    train_data, validation_data, resolution = load_data_dynamically(data_parameters=data_parameters, flag='train')

    if data_parameters['fix_seed'] == True:

        import random

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        train_loader = data.DataLoader(
            dataset=train_data,
            batch_size=training_parameters['training_batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=data_parameters['num_workers'],
            worker_init_fn=seed_worker,
            generator=g,
        )
        validation_loader = data.DataLoader(
            dataset=validation_data,
            batch_size=training_parameters['validation_batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=data_parameters['num_workers'],
            worker_init_fn=seed_worker,
            generator=g,
        )

        torch.manual_seed(0)
        AgeMapperModel = SwinAgeMapper(
                                    img_size = network_parameters['img_size'],
                                    in_channels = network_parameters['in_channels'],
                                    depths = network_parameters['depths'],
                                    num_heads = network_parameters['num_heads'],
                                    feature_size = network_parameters['feature_size'],
                                    drop_rate = network_parameters['drop_rate'],
                                    attn_drop_rate = network_parameters['attn_drop_rate'],
                                    dropout_path_rate = network_parameters['dropout_path_rate'],
                                    use_checkpoint = network_parameters['use_checkpoint'],
                                    spatial_dims = network_parameters['spatial_dims'],
                                    downsample = network_parameters['downsample'],
                                    fully_connected_activation = network_parameters['fully_connected_activation'],
                                    resolution=resolution,
                                    patch_size=network_parameters['patch_size'],
                                    )

    else:
    
        train_loader = data.DataLoader(
            dataset=train_data,
            batch_size=training_parameters['training_batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=data_parameters['num_workers']
        )
        validation_loader = data.DataLoader(
            dataset=validation_data,
            batch_size=training_parameters['validation_batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=data_parameters['num_workers']
        )

        AgeMapperModel = SwinAgeMapper(
                                        img_size = network_parameters['img_size'],
                                        in_channels = network_parameters['in_channels'],
                                        depths = network_parameters['depths'],
                                        num_heads = network_parameters['num_heads'],
                                        feature_size = network_parameters['feature_size'],
                                        drop_rate = network_parameters['drop_rate'],
                                        attn_drop_rate = network_parameters['attn_drop_rate'],
                                        dropout_path_rate = network_parameters['dropout_path_rate'],
                                        use_checkpoint = network_parameters['use_checkpoint'],
                                        spatial_dims = network_parameters['spatial_dims'],
                                        downsample = network_parameters['downsample'],
                                        fully_connected_activation = network_parameters['fully_connected_activation'],
                                        resolution=resolution,
                                        patch_size=network_parameters['patch_size'],
                                        )

    if training_parameters['use_pre_trained']:
        pre_trained_path = "saved_models/" + training_parameters['pre_trained_experiment_name'] + ".pth.tar"
        AgeMapperModel_pretrained = torch.load(pre_trained_path, map_location=torch.device('cpu'))
        AgeMapperModel.load_state_dict(AgeMapperModel_pretrained)
        del AgeMapperModel_pretrained
        print('--> Using PRE-TRAINED NETWORK: ', pre_trained_path)
        print('\n') 
        print('Total number of model parameters')
        print(sum([p.numel() for p in AgeMapperModel.parameters()]))
        model_parameters = filter(lambda p: p.requires_grad, AgeMapperModel.parameters())
        print('Total number of trainable parameters')
        params = sum([p.numel() for p in model_parameters])
        print(params)
        print('\n')


    solver = Solver(model=AgeMapperModel,
                    number_of_classes=network_parameters['number_of_classes'],
                    experiment_name=training_parameters['experiment_name'],
                    optimizer=optimizer,
                    optimizer_arguments=optimizer_arguments,
                    loss_function=loss_function,
                    model_name=training_parameters['experiment_name'],
                    number_epochs=training_parameters['number_of_epochs'],
                    loss_log_period=training_parameters['loss_log_period'],
                    learning_rate_scheduler_gamma=training_parameters['learning_rate_scheduler_gamma'],
                    use_last_checkpoint=training_parameters['use_last_checkpoint'],
                    experiment_directory=misc_parameters['experiments_directory'],
                    logs_directory=misc_parameters['logs_directory'],
                    checkpoint_directory=misc_parameters['checkpoint_directory'],
                    best_checkpoint_directory=misc_parameters['best_checkpoint_directory'],
                    save_model_directory=misc_parameters['save_model_directory'],
                    learning_rate_scheduler_flag = training_parameters['learning_rate_scheduler_flag'],
                    learning_rate_scheduler_patience=training_parameters['learning_rate_scheduler_patience'],
                    learning_rate_scheduler_threshold=training_parameters['learning_rate_scheduler_threshold'],
                    learning_rate_scheduler_min_value=training_parameters['learning_rate_scheduler_min_value'],
                    lr_cosine_scheduler_warmup_epochs = training_parameters['lr_cosine_scheduler_warmup_epochs'],
                    lr_cosine_scheduler_max_epochs = training_parameters['lr_cosine_scheduler_max_epochs'],
                    early_stopping_patience=training_parameters['early_stopping_patience'],
                    early_stopping_min_patience=training_parameters['early_stopping_min_patience'],
                    early_stopping_min_delta=training_parameters['early_stopping_min_delta'],
                    use_pre_trained = training_parameters['use_pre_trained'],
                    )

    solver.train(train_loader, validation_loader)

    del train_data, validation_data, train_loader, validation_loader, AgeMapperModel, solver, optimizer
    torch.cuda.empty_cache()


def evaluate_data(mapping_evaluation_parameters, data_parameters, network_parameters):

    test_data, volumes_to_be_used, prediction_output_statistics_name, len_test_data, resolution = load_data_dynamically(
                                                                                                            data_parameters=data_parameters, 
                                                                                                            mapping_evaluation_parameters=mapping_evaluation_parameters, 
                                                                                                            flag='test'
                                                                                                            )

    if data_parameters['fix_seed'] == True:

        import random

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        test_loader = data.DataLoader(
            dataset = test_data,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=data_parameters['num_workers'],
            worker_init_fn=seed_worker,
            generator=g,
        )

    else:

        test_loader = data.DataLoader(
            dataset = test_data,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=data_parameters['num_workers']
        )

    AgeMapperModel = SwinAgeMapper(
                                    img_size = network_parameters['img_size'],
                                    in_channels = network_parameters['in_channels'],
                                    depths = network_parameters['depths'],
                                    num_heads = network_parameters['num_heads'],
                                    feature_size = network_parameters['feature_size'],
                                    drop_rate = network_parameters['drop_rate'],
                                    attn_drop_rate = network_parameters['attn_drop_rate'],
                                    dropout_path_rate = network_parameters['dropout_path_rate'],
                                    use_checkpoint = network_parameters['use_checkpoint'],
                                    spatial_dims = network_parameters['spatial_dims'],
                                    downsample = network_parameters['downsample'],
                                    # activation= network_parameters['activation'],
                                    fully_connected_activation = network_parameters['fully_connected_activation'],
                                    resolution=resolution,
                                    patch_size=network_parameters['patch_size'],
                                    )

    device = mapping_evaluation_parameters['device']

    experiment_name = mapping_evaluation_parameters['experiment_name']
    trained_model_path = "saved_models/" + experiment_name + ".pth.tar"
    prediction_output_path = experiment_name + "_predictions"
    control = mapping_evaluation_parameters['control']
    dataset_sex = data_parameters['dataset_sex']
    
    evaluations.evaluate_data(
                        model = AgeMapperModel,
                        test_loader = test_loader,
                        volumes_to_be_used = volumes_to_be_used, 
                        prediction_output_statistics_name = prediction_output_statistics_name, 
                        trained_model_path = trained_model_path,
                        device = device,
                        prediction_output_path = prediction_output_path,
                        control = control,
                        dataset_sex = dataset_sex,
                        len_test_data = len_test_data,
                    )


def delete_files(folder):
    for object_name in os.listdir(folder):
        file_path = os.path.join(folder, object_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exception:
            print(exception)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True,
                        help='run mode, valid values are train, evaluate-data, clear-checkpoints, clear-checkpoints-completely, clear-logs, clear-experiment, clear-experiment-completely, train-and-evaluate-mapping, lr-range-test, solver-logger-test')
    parser.add_argument('--model_name', '-n', required=True,
                        help='model name, required for identifying the settings file modelName.ini & modelName_eval.ini')
    parser.add_argument('--use_last_checkpoint', '-c', required=False,
                        help='flag indicating if the last checkpoint should be used if 1; useful when wanting to time-limit jobs.')
    parser.add_argument('--number_of_epochs', '-e', required=False,
                        help='flag indicating how many epochs the network will train for; should be limited to ~3 hours or 2/3 epochs')

    arguments = parser.parse_args()

    settings_file_name = arguments.model_name + '.ini'
    evaluation_settings_file_name = arguments.model_name + '_eval.ini'

    settings = Settings(settings_file_name)
    data_parameters = settings['DATA']
    training_parameters = settings['TRAINING']
    network_parameters = settings['NETWORK']
    misc_parameters = settings['MISC']

    if arguments.use_last_checkpoint == '1':
        training_parameters['use_last_checkpoint'] = True
    elif arguments.use_last_checkpoint == '0':
        training_parameters['use_last_checkpoint'] = False

    if arguments.number_of_epochs is not None:
        training_parameters['number_of_epochs'] = int(arguments.number_of_epochs)

    if arguments.mode == 'train':
        train(data_parameters, training_parameters, network_parameters, misc_parameters)

    elif arguments.mode == 'evaluate-data':
        logging.basicConfig(filename='evaluate-data-error.log')
        settings_evaluation = Settings(evaluation_settings_file_name)
        mapping_evaluation_parameters = settings_evaluation['MAPPING']
        evaluate_data(mapping_evaluation_parameters, data_parameters, network_parameters)

    elif arguments.mode == 'clear-checkpoints':

        warning_message = input("Warning! This command will delete all checkpoints. Continue [y]/n: ")
        if warning_message == 'y':
            if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                print('Cleared the current experiment checkpoints successfully!')
            else:
                print('ERROR: Could not find the experiment checkpoints.')
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-checkpoints-completely':
        warning_message = input("WARNING! This command will delete all checkpoints (INCL BEST). DANGER! Continue [y]/n: ")
        if warning_message == 'y':
            warning_message2 = input("ARE YOU SURE? [y]/n: ")
            if warning_message2 == 'y':
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                    print('Cleared the current experiment checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory']))
                    print('Cleared the current experiment best checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment best checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name']))
                    print('Cleared the current experiment folder successfully!')
                else:
                    print("ERROR: Could not find the experiment folder.")
            else:
                print("Action Cancelled!")
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-logs':

        warning_message = input("Warning! This command will delete all checkpoints and logs. Continue [y]/n: ")
        if warning_message == 'y':
            if os.path.exists(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name'])):
                shutil.rmtree(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name']))
                print('Cleared the current experiment logs directory successfully!')
            else:
                print("ERROR: Could not find the experiment logs directory!")
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-experiment':

        warning_message = input("Warning! This command will delete all checkpoints and logs. Continue [y]/n: ")
        if warning_message == 'y':
            if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                print('Cleared the current experiment checkpoints successfully!')
            else:
                print('ERROR: Could not find the experiment checkpoints.')
            if os.path.exists(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name'])):
                shutil.rmtree(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name']))
                print('Cleared the current experiment logs directory successfully!')
            else:
                print("ERROR: Could not find the experiment logs directory!")
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-experiment-completely':
        warning_message = input("WARNING! This command will delete all checkpoints (INCL BEST) and logs. DANGER! Continue [y]/n: ")
        if warning_message == 'y':
            warning_message2 = input("ARE YOU SURE? [y]/n: ")
            if warning_message2 == 'y':
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                    print('Cleared the current experiment checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory']))
                    print('Cleared the current experiment best checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment best checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name']))
                    print('Cleared the current experiment folder successfully!')
                else:
                    print("ERROR: Could not find the experiment folder.")
                if os.path.exists(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name'])):
                    shutil.rmtree(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name']))
                    print('Cleared the current experiment logs directory successfully!')
                else:
                    print("ERROR: Could not find the experiment logs directory!")
            else:
                print("Action Cancelled!")
        else:
            print("Action Cancelled!")

    # elif arguments.mode == 'clear-everything':
    #     delete_files(misc_parameters['experiments_directory'])
    #     delete_files(misc_parameters['logs_directory'])
    #     print('Cleared the all the checkpoints and logs directory successfully!')

    elif arguments.mode == 'train-and-evaluate-data':
        settings_evaluation = Settings(evaluation_settings_file_name)
        mapping_evaluation_parameters = settings_evaluation['MAPPING']
        train(data_parameters, training_parameters,
              network_parameters, misc_parameters)
        logging.basicConfig(filename='evaluate-mapping-error.log')
        evaluate_data(mapping_evaluation_parameters)
      
    else:
        raise ValueError('Invalid mode value! Only supports: train, evaluate-data, evaluate-mapping, train-and-evaluate-mapping, clear-checkpoints, clear-logs,  clear-experiment and clear-everything (req uncomment for safety!)')
