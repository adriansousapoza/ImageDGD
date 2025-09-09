"""
Hyperparameter optimization using Optuna with MLflow integration.
"""

import optuna
import mlflow
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf
import logging

from ..training import DGDTrainer
from ..data import create_dataloaders, get_sample_batches


class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimization for DGD model.
    """
    
    def __init__(self, config: DictConfig, device):
        """
        Initialize the optimizer.
        
        Parameters:
        ----------
        config: Configuration containing optimization settings
        device: Device to run optimization on
        """
        self.config = config
        self.device = device
        self.optimization_config = config.optimization
        
        # Set up logging
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
        optuna.logging.set_verbosity(optuna.logging.INFO)
        
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with configured settings."""
        opt_config = self.optimization_config
        
        # Create pruner
        if opt_config.pruner.type == "MedianPruner":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=opt_config.pruner.n_startup_trials,
                n_warmup_steps=opt_config.pruner.n_warmup_steps,
                interval_steps=opt_config.pruner.interval_steps
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create sampler
        if opt_config.sampler.type == "TPESampler":
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=opt_config.sampler.n_startup_trials
            )
        else:
            sampler = optuna.samplers.RandomSampler()
        
        # Create study
        study = optuna.create_study(
            study_name=opt_config.study_name,
            storage=opt_config.storage,
            direction=opt_config.direction,
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True
        )
        
        return study
    
    def _suggest_parameters(self, trial: optuna.Trial) -> DictConfig:
        """Suggest parameters for the trial."""
        # Start with base config
        trial_config = OmegaConf.create(self.config)
        
        # Suggest parameters based on configuration
        for param_name, param_config in self.optimization_config.parameters.items():
            if param_config.type == "int":
                value = trial.suggest_int(
                    param_name, 
                    param_config.low, 
                    param_config.high
                )
            elif param_config.type == "float":
                value = trial.suggest_float(
                    param_name, 
                    param_config.low, 
                    param_config.high,
                    log=param_config.get("log", False)
                )
            elif param_config.type == "categorical":
                value = trial.suggest_categorical(
                    param_name, 
                    param_config.choices
                )
            else:
                raise ValueError(f"Unknown parameter type: {param_config.type}")
            
            # Set the parameter in the config
            self._set_nested_parameter(trial_config, param_name, value)
        
        return trial_config
    
    def _set_nested_parameter(self, config: DictConfig, param_name: str, value: Any):
        """Set a nested parameter in the configuration."""
        keys = param_name.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.
        
        Parameters:
        ----------
        trial: Optuna trial object
        
        Returns:
        -------
        Objective value to minimize/maximize
        """
        # Get trial configuration
        trial_config = self._suggest_parameters(trial)
        
        # Create data loaders
        train_loader, test_loader, class_names = create_dataloaders(trial_config)
        sample_data = get_sample_batches(train_loader, test_loader, self.device)
        
        # Create trainer
        trainer = DGDTrainer(trial_config, self.device)
        
        # Start MLflow child run for this trial
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as run:
            # Log trial parameters
            mlflow.log_params(trial.params)
            mlflow.log_param("trial_number", trial.number)
            
            # Train model
            try:
                results = trainer.train(train_loader, test_loader, sample_data, class_names)
                
                # Get objective value
                objective_metric = self.optimization_config.objective.metric
                if objective_metric == "val_total_loss":
                    objective_value = results["final_test_loss"]
                elif objective_metric == "train_total_loss":
                    objective_value = results["final_train_loss"]
                else:
                    raise ValueError(f"Unknown objective metric: {objective_metric}")
                
                # Log objective value and other metrics
                mlflow.log_metric("objective_value", objective_value)
                mlflow.log_metric("final_train_loss", results["final_train_loss"])
                mlflow.log_metric("final_test_loss", results["final_test_loss"])
                mlflow.log_metric("total_training_time", results["total_time"])
                
                # Report intermediate values for pruning
                for epoch, loss in enumerate(results["test_losses"], 1):
                    trial.report(loss, epoch)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        mlflow.set_tag("trial_status", "pruned")
                        raise optuna.exceptions.TrialPruned()
                
                mlflow.set_tag("trial_status", "completed")
                return objective_value
                
            except optuna.exceptions.TrialPruned:
                mlflow.set_tag("trial_status", "pruned")
                raise
            except Exception as e:
                # Log error and return worst possible value
                mlflow.log_param("error", str(e))
                mlflow.set_tag("trial_status", "failed")
                print(f"Trial {trial.number} failed with error: {e}")
                
                if self.optimization_config.direction == "minimize":
                    return float('inf')
                else:
                    return float('-inf')
    
    def optimize(self) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Returns:
        -------
        Completed Optuna study
        """
        # Create study
        study = self._create_study()
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.optimization_config.n_trials,
            timeout=self.optimization_config.timeout
        )
        
        # Print results
        print("Optimization completed!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value}")
        print(f"Best parameters: {study.best_params}")
        
        return study
    
    def get_best_config(self, study: optuna.Study) -> DictConfig:
        """
        Get the best configuration from the study.
        
        Parameters:
        ----------
        study: Completed Optuna study
        
        Returns:
        -------
        Best configuration
        """
        # Start with base config
        best_config = OmegaConf.create(self.config)
        
        # Apply best parameters
        for param_name, value in study.best_params.items():
            self._set_nested_parameter(best_config, param_name, value)
        
        return best_config
