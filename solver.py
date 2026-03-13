import os
import numpy as np
import torch
import torch.nn as nn
import glob
from collections import OrderedDict

from datetime import datetime
from utils.misc import create_folder, mae
from utils.logging_functions import LogWriter
from utils.early_stopping import EarlyStopping
from utils.modified_lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import lr_scheduler

from torch.nn import L1Loss

checkpoint_extension = 'path.tar'


class Solver():
    def __init__(self,
                 model,
                 number_of_classes,
                 experiment_name,
                 optimizer,
                 optimizer_arguments={},
                 loss_function=torch.nn.MSELoss(),
                 model_name='BrainMapper',
                 number_epochs=10,
                 loss_log_period=5,
                 learning_rate_scheduler_gamma=0.5,
                 use_last_checkpoint=True,
                 experiment_directory='experiments',
                 logs_directory='logs',
                 checkpoint_directory='checkpoints',
                 best_checkpoint_directory = 'best_checkpoint_directory',
                 save_model_directory='saved_models',
                 learning_rate_scheduler_flag = 'LinearWarmupCosineAnnealingLR',
                 learning_rate_scheduler_patience=5,
                 learning_rate_scheduler_threshold=1e-6,
                 learning_rate_scheduler_min_value=5e-6,
                 lr_cosine_scheduler_warmup_epochs=4,
                 lr_cosine_scheduler_max_epochs=300,
                 early_stopping_min_patience=50,
                 early_stopping_patience=10,
                 early_stopping_min_delta=0,
                 use_pre_trained = False
                 ):

        self.model = model
        self.parallelism = False

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.backends.cudnn.benchmark = True #enables benchmark mode, making cudnn look for optimal algorithms for config, leading to faster runtime https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936

        if self.device == "cpu":
            print("WARNING: Default device is CPU, not GPU!")
        elif torch.cuda.device_count()>1:
            self.parallelism = True
            print("ATTENTION! Multiple GPUs detected. {} GPUs will be used for training".format(torch.cuda.device_count()))
        else:
            print("A single GPU detected")

        if optimizer_arguments['weight_decay']!=0:
            prelus = {name for name, module in model.named_modules() if isinstance(module, torch.nn.PReLU)}
            if len(prelus)!=0:
                prelu_parameter_names = {name for name, _ in model.named_parameters() if name.rsplit('.', 1)[0] in prelus}
                parameters = [
                    {'params': [parameter for parameter_name, parameter in model.named_parameters() if parameter_name not in prelu_parameter_names]},
                    {'params': [parameter for parameter_name, parameter in model.named_parameters() if parameter_name in prelu_parameter_names], 'weight_decay': 0.0}
                ]
            else:
                parameters = model.parameters()
        else:
            parameters = model.parameters()
        self.optimizer = optimizer(parameters, **optimizer_arguments)

        if torch.cuda.is_available():
            if hasattr(loss_function, 'to'):
                self.loss_function = loss_function.to(self.device)
                self.MAE = L1Loss().to(self.device)
            else:
                self.loss_function = loss_function
                self.MAE = L1Loss()

        else:
            self.loss_function = loss_function

        self.model_name = model_name
        self.number_epochs = number_epochs
        self.loss_log_period = loss_log_period


        self.learning_rate_scheduler_flag = learning_rate_scheduler_flag

        if self.learning_rate_scheduler_flag == 'ReduceLROnPlateau':
            self.learning_rate_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer,
                                                                        factor = learning_rate_scheduler_gamma,
                                                                        patience = learning_rate_scheduler_patience,
                                                                        threshold = learning_rate_scheduler_threshold,
                                                                        threshold_mode='abs',
                                                                        min_lr= learning_rate_scheduler_min_value,
                                                                        verbose=True
                                                                        ) 
        elif self.learning_rate_scheduler_flag == 'LinearWarmupCosineAnnealingLR':
            self.learning_rate_scheduler = LinearWarmupCosineAnnealingLR(optimizer = self.optimizer, 
                                                                        warmup_epochs=lr_cosine_scheduler_warmup_epochs, 
                                                                        max_epochs=lr_cosine_scheduler_max_epochs,
                                                                        )
        else:
            print('ERROR - NO ADEQUATE SCHEDUELER WAS SELECTED. One should be used!') 
            self.learning_rate_scheduler = None

        self.use_last_checkpoint = use_last_checkpoint

        experiment_directory_path = os.path.join(experiment_directory, experiment_name)
        self.experiment_directory_path = experiment_directory_path

        self.checkpoint_directory = checkpoint_directory
        self.best_checkpoint_directory = best_checkpoint_directory

        create_folder(experiment_directory)
        create_folder(experiment_directory_path)
        create_folder(os.path.join(experiment_directory_path, self.checkpoint_directory))
        create_folder(os.path.join(experiment_directory_path, self.best_checkpoint_directory))

        self.start_epoch = 1
        self.start_iteration = 1

        self.LogWriter = LogWriter(number_of_classes=number_of_classes,
                                   logs_directory=logs_directory,
                                   experiment_name=experiment_name,
                                   use_last_checkpoint=use_last_checkpoint,
                                   )

        self.early_stop = False
        self.early_stopping_min_patience = early_stopping_min_patience

        self.save_model_directory = save_model_directory
        self.final_model_output_file = experiment_name + ".pth.tar"

        self.best_score_early_stop = None
        self.counter_early_stop = 0
        self.previous_loss = None
        self.valid_epoch = None
        self.previous_age_deltas = None

        if use_last_checkpoint:
            self.load_checkpoint()
            self.EarlyStopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta, best_score=self.best_score_early_stop, counter=self.counter_early_stop)
        else:
            self.EarlyStopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)

        self.use_pre_trained = use_pre_trained

    def train(self, train_loader, validation_loader):

        model, optimizer, learning_rate_scheduler = self.model, self.optimizer, self.learning_rate_scheduler
        dataloaders = {'train': train_loader, 'validation': validation_loader}

        print('-----------------------------------------------')
        print('DEBUGGING PRINT: LR SCHEDULER STATE DICT')
        print(learning_rate_scheduler.state_dict())
        print('-----------------------------------------------')

        numberOfParameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.parallelism == True:
            model = nn.DataParallel(model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # clear memory
            model.to(self.device)  # Moving the model to GPU

        print('****************************************************************')
        print('TRAINING IS STARTING!')
        print('=====================')
        print('Model Name: {}'.format(self.model_name))
        print('Model Parameters: {}'.format(numberOfParameters))
        if torch.cuda.is_available():
            print('Device Type: {}'.format(
                torch.cuda.get_device_name(self.device)))
        else:
            print('Device Type: {}'.format(self.device))
        start_time = datetime.now()
        print('Started At: {}'.format(start_time))
        print('----------------------------------------')

        iteration = self.start_iteration

        for epoch in range(self.start_epoch, self.number_epochs+1):
               
            if self.early_stop == True:
                print("ATTENTION!: Training stopped due to previous early stop flag!")
                break

            print("Epoch {}/{}".format(epoch, self.number_epochs))
            epoch_start_time = datetime.now()

            for phase in ['train', 'validation']:
                print('--> Phase: {}'.format(phase))

                losses = []
                age_deltas = []

                if phase == 'train':

                    model.train()
                    for batch_index, sampled_batch in enumerate(dataloaders[phase]):
                        X = sampled_batch[0].type(torch.FloatTensor)
                        y_age = sampled_batch[1].type(torch.FloatTensor)
                        y_age = y_age.reshape(-1,1)

                        # We add an extra dimension (~ number of channels) for the 3D convolutions.
                        if len(X.size())<5:
                            X = torch.unsqueeze(X, dim=1)

                        if torch.cuda.is_available():
                            X = X.cuda(self.device, non_blocking=True)
                            y_age = y_age.cuda(self.device, non_blocking=True)

                        # Operation carried out by original SwinUNETR code.
                        # Equivalent to optimizer.zero_grad() ? 
                        # for param in model.parameters():
                        #     param.grad = None

                        y_hat = model(X)   # Forward pass
                        loss = self.loss_function(y_hat, y_age)

                        age_delta = self.MAE(y_hat, y_age)

                        optimizer.zero_grad()  # Zero the parameter gradients
                        loss.backward()  # Backward propagation
                        optimizer.step()

                        if batch_index % self.loss_log_period == 0:

                            self.LogWriter.loss_per_iteration(loss.item(), batch_index, iteration)
                            self.LogWriter.learning_rate_per_iteration(optimizer.param_groups[0]['lr'], batch_index, iteration)
                        
                        iteration += 1

                        losses.append(loss.item())
                        age_deltas.append(age_delta.item())

                        # Clear the memory
                        
                        del X, y_hat, loss, y_age, age_delta
                        torch.cuda.empty_cache()

                    # Operation carried out by original SwinUNETR code.
                    # Equivalent to optimizer.zero_grad() ? 
                    # for param in model.parameters():
                    #     param.grad = None

                else:

                    model.eval()
                    with torch.no_grad():
                        for batch_index, sampled_batch in enumerate(dataloaders[phase]):
                            X = sampled_batch[0].type(torch.FloatTensor)
                            y_age = sampled_batch[1].type(torch.FloatTensor)
                            y_age = y_age.reshape(-1,1)

                            # We add an extra dimension (~ number of channels) for the 3D convolutions.
                            if len(X.size())<5:
                                X = torch.unsqueeze(X, dim=1)

                            if torch.cuda.is_available():
                                X = X.cuda(self.device, non_blocking=True)
                                y_age = y_age.cuda(self.device, non_blocking=True)

                            y_hat = model(X)   # Forward pass
                            loss = self.loss_function(y_hat, y_age)

                            age_delta = self.MAE(y_hat, y_age)

                            losses.append(loss.item())
                            age_deltas.append(age_delta.item())

                            # Clear the memory
                            
                            del X, y_hat, loss, y_age, age_delta
                            torch.cuda.empty_cache()

                            if batch_index != len(dataloaders[phase]) - 1:
                                print("#", end='', flush=True)
                            else:
                                print("100%", flush=True)

                with torch.no_grad():

                    if phase == 'train':
                        self.LogWriter.loss_per_epoch(losses, phase, epoch)
                        self.LogWriter.learning_rate_per_epoch(optimizer.param_groups[0]['lr'], phase, epoch)
                        self.LogWriter.age_delta_per_epoch(age_deltas, phase, epoch)

                    elif phase == 'validation':
                        self.LogWriter.loss_per_epoch(losses, phase, epoch, previous_loss=self.previous_loss)
                        self.previous_loss = np.mean(losses)
                        self.LogWriter.learning_rate_per_epoch(optimizer.param_groups[0]['lr'], phase, epoch)
                        self.validation_losses = losses
                        self.LogWriter.age_delta_per_epoch(age_deltas, phase, epoch, previous_loss=self.previous_age_deltas)
                        self.previous_age_deltas = np.mean(age_deltas)

            if self.learning_rate_scheduler is not None:
                if self.learning_rate_scheduler_flag == 'ReduceLROnPlateau':
                    learning_rate_scheduler.step(np.mean(self.validation_losses))    
                elif self.learning_rate_scheduler_flag == 'LinearWarmupCosineAnnealingLR':
                    learning_rate_scheduler.step()

            with torch.no_grad():

                if epoch <= self.early_stopping_min_patience:
                    counter_overwrite = True
                else:
                    counter_overwrite = False

                early_stop, best_score_early_stop, counter_early_stop = self.EarlyStopping(np.mean(self.validation_losses), counter_overwrite=counter_overwrite)

                if epoch <= self.early_stopping_min_patience:
                    self.early_stop = False
                    self.counter_early_stop = 0
                    self.best_score_early_stop = None
                else:
                    self.early_stop = early_stop
                    self.counter_early_stop = counter_early_stop
                    self.best_score_early_stop = best_score_early_stop
                
                checkpoint_name = os.path.join(self.experiment_directory_path, self.checkpoint_directory, 'checkpoint_epoch_' + str(epoch) + '.' + checkpoint_extension)
                best_checkpoint_name = os.path.join(self.experiment_directory_path, self.best_checkpoint_directory, 'best_checkpoint' + '.' + checkpoint_extension)
                final_checkpoint_name = os.path.join(self.experiment_directory_path, self.best_checkpoint_directory, 'final_checkpoint' + '.' + checkpoint_extension)

                if self.counter_early_stop == 0:
                    self.valid_epoch = epoch

                    self.save_checkpoint(state={'epoch': epoch + 1,
                            'start_iteration': iteration + 1,
                            'arch': self.model_name,
                            # 'state_dict': model.state_dict(),
                            'state_dict': model.module.state_dict() if self.parallelism==True else model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': learning_rate_scheduler.state_dict(),
                            'best_score_early_stop': self.best_score_early_stop,
                            'counter_early_stop': self.counter_early_stop,
                            'previous_loss': self.previous_loss,
                            'previous_age_deltas': self.previous_age_deltas,
                            'early_stop': self.early_stop,
                            'valid_epoch': self.valid_epoch
                            },
                        filename=best_checkpoint_name
                        )

                self.save_checkpoint(state={'epoch': epoch + 1,
                                            'start_iteration': iteration + 1,
                                            'arch': self.model_name,
                                            # 'state_dict': model.state_dict(),
                                            'state_dict': model.module.state_dict() if self.parallelism==True else model.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'scheduler': learning_rate_scheduler.state_dict(),
                                            'best_score_early_stop': self.best_score_early_stop,
                                            'counter_early_stop': self.counter_early_stop,
                                            'previous_loss': self.previous_loss,
                                            'previous_age_deltas': self.previous_age_deltas,
                                            'early_stop': self.early_stop,
                                            'valid_epoch': self.valid_epoch
                                            },
                                        filename=checkpoint_name
                                        )

                if epoch == self.number_epochs:
                    self.save_checkpoint(state={'epoch': epoch + 1,
                                        'start_iteration': iteration + 1,
                                        'arch': self.model_name,
                                        # 'state_dict': model.state_dict(),
                                        'state_dict': model.module.state_dict() if self.parallelism==True else model.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'scheduler': learning_rate_scheduler.state_dict(),
                                        'best_score_early_stop': self.best_score_early_stop,
                                        'counter_early_stop': self.counter_early_stop,
                                        'previous_loss': self.previous_loss,
                                        'previous_age_deltas': self.previous_age_deltas,
                                        'early_stop': self.early_stop,
                                        'valid_epoch': self.valid_epoch
                                        },
                                    filename=final_checkpoint_name
                                    )

            print("Epoch {}/{} DONE! Epoch Time = {}".format(epoch, self.number_epochs, datetime.now() - epoch_start_time))

            # Early Stop Condition
            
            if self.early_stop == True:
                print("ATTENTION!: Training stopped early to prevent overfitting!")
                self.load_checkpoint(epoch=self.valid_epoch)
                break
            else:
                continue

        if self.early_stop == True:
            
            self.LogWriter.close()

            print('----------------------------------------')
            print('NO TRAINING DONE TO PREVENT OVERFITTING!')
            print('=====================')
            end_time = datetime.now()
            print('Completed At: {}'.format(end_time))
            print('Training Duration: {}'.format(end_time - start_time))
            print('****************************************************************')
        else:
            model_output_path = os.path.join(self.save_model_directory, self.final_model_output_file)

            create_folder(self.save_model_directory)

            self.load_checkpoint(epoch=self.valid_epoch) # We always save the best epoch even if not overfitting

            if self.parallelism == True:
                torch.save(model.module.state_dict(), model_output_path)
            else:
                torch.save(model.state_dict(), model_output_path)

            self.LogWriter.close()

            print('----------------------------------------')
            print('TRAINING IS COMPLETE!')
            print('=====================')
            end_time = datetime.now()
            print('Completed At: {}'.format(end_time))
            print('Training Duration: {}'.format(end_time - start_time))
            print('Final Model Saved in: {}'.format(model_output_path))
            print('****************************************************************')

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def load_checkpoint(self, epoch=None):

        if epoch is not None:
            checkpoint_file_path = os.path.join(self.experiment_directory_path, self.checkpoint_directory, 'checkpoint_epoch_' + str(epoch) + '.' + checkpoint_extension)
            print("Loading checkpoint at path: ", checkpoint_file_path)
            self._checkpoint_reader(checkpoint_file_path)
        else:
            universal_path = os.path.join(self.experiment_directory_path, self.checkpoint_directory, '*.' + checkpoint_extension) 
            checkpoint_file_path = os.path.join(self.experiment_directory_path, self.checkpoint_directory, 'checkpoint_epoch_' + str(len(glob.glob(universal_path))) + '.' + checkpoint_extension)
            print("Loading checkpoint at path: ", checkpoint_file_path)
            self._checkpoint_reader(checkpoint_file_path)

    def _checkpoint_reader(self, checkpoint_file_path):

        self.LogWriter.log("Loading Checkpoint {}".format(checkpoint_file_path))

        checkpoint = torch.load(checkpoint_file_path)
        self.start_epoch = checkpoint['epoch']
        self.start_iteration = checkpoint['start_iteration']
        # We are not loading the model_name as we might want to pre-train a model and then use it.
        
        if self.parallelism == True:
            # The model is defined without parallel training in mind, which means that if we are training using multiple GPUs, the "module." is added to all keys in the state dict.
            # To allow the state-dict loader to be compatible, the "module." strings needs to be removed.
            correct_state_dict = {}
            for key in checkpoint['state_dict'].keys():
                if key.startswith('module.'):
                    new_key = key.replace('module.', "")
                    correct_state_dict[new_key] = checkpoint['state_dict'][key]
                else:
                    correct_state_dict[key] = checkpoint['state_dict'][key]
            correct_state_dict = OrderedDict(correct_state_dict)
            del checkpoint['state_dict']
            checkpoint['state_dict'] = correct_state_dict

        self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_score_early_stop = checkpoint['best_score_early_stop']
        self.counter_early_stop = checkpoint['counter_early_stop']
        self.previous_loss = checkpoint['previous_loss']
        self.early_stop = checkpoint['early_stop']
        self.valid_epoch = checkpoint['valid_epoch']
        self.previous_age_deltas = checkpoint['previous_age_deltas']

        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler'])
        self.LogWriter.log(
            "Checkpoint Loaded {} - epoch {}".format(checkpoint_file_path, checkpoint['epoch']))
