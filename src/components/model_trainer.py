import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "Gradient Boosting":  GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBoost Classifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params={
                "Decision Tree": {
                    'criterion': 'gini',    # Splitting criterion: {'gini', 'entropy'}
                    'max_depth': None,       # Maximum depth of the tree. None means unlimited depth.
                    'min_samples_split': 2,  # Minimum number of samples required to split a node
                    'min_samples_leaf': 1    # Minimum number of samples required at each leaf node

                },
                "Random Forest":{
                    'n_estimators': 100,     # Number of trees in the forest
                    'criterion': 'gini',     # Splitting criterion: {'gini', 'entropy'}
                    'max_depth': None,       # Maximum depth of the trees. None means unlimited depth.
                    'min_samples_split': 2,  # Minimum number of samples required to split a node
                    'min_samples_leaf': 1,   # Minimum number of samples required at each leaf node
                    #'n_jobs': -1 
                },
                "Gradient Boosting":{
                    'loss': 'deviance',       # Loss function to be optimized: {'deviance', 'exponential'}
                    'learning_rate': 0.1,     # Learning rate shrinks the contribution of each tree
                    'n_estimators': 100,      # Number of boosting stages
                    'subsample': 1.0,         # Fraction of samples used for fitting the individual base learners
                    'min_samples_split': 2,   # Minimum number of samples required to split an internal node
                    'min_samples_leaf': 1,    # Minimum number of samples required to be at a leaf node
                    'max_depth': 3,           # Maximum depth of the individual regression estimators
                    'max_features': None,     # Number of features to consider when looking for the best split
                    'random_state': None,     # Seed used by the random number generator 
                    'tol': 1e-4  
                },
                "Logistic Regression":{
                     'penalty': 'l2',             # Penalty term: {'l1', 'l2', 'elasticnet', 'none'}
                     'C': 1.0,                    # Inverse of regularization strength; smaller values specify stronger regularization
                     'solver': 'lbfgs',           # Algorithm to use in the optimization problem: {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}
                     'max_iter': 100,             # Maximum number of iterations for the solver to converge
                     'multi_class': 'auto',       # Strategy to use for multiclass classification: {'ovr', 'multinomial', 'auto'}
                     'random_state': None

                },

            
                "XGBRegressor":{
                    'max_depth': 3,         # Maximum depth of a tree
                    'learning_rate': 0.1,   # Step size shrinkage used in update to prevent overfitting
                    'n_estimators': 100,    # Number of boosting rounds
                    'objective': 'binary:logistic'
                    
                },
    
                "AdaBoost Regressor":{
                    'n_estimators': 50,     # Number of weak learners to train iteratively
                    'learning_rate': 1.0,   # Learning rate shrinks the contribution of each classifier
                    'algorithm': 'SAMME.R'
                }
                
            }


            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy_score = accuracy_score(y_test, predicted)
            return accuracy_score
            



            
        except Exception as e:
            raise CustomException(e,sys)