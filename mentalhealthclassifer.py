import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_score, roc_curve, roc_auc_score, f1_score, recall_score
)
from sklearn.pipeline import Pipeline
import seaborn as sns
import warnings


warnings.filterwarnings('ignore') # will compress all of the warnings that are derived from sklearn library when the models are being built and when it is trying to push out information.

class MentalHealthClassifier:
    """
    The purpose of this class is to design, build and find the best classification-task ML model that can be implemented towards solving the first part of the problem that was put into place
    by the mental health dataset - prediction of the classification of the mental condition that the population group. The target variable for the classification task is the column 'Indicator'
    which was separated into 3 classification:
        - Symptoms of Anxiety Disorder (0)
        - Symptoms of Depressive Disorder (1)
        - Symptoms of Anxiety or Depressive Disorder (2)

    The dataframe had already been preprocessed and encoded to now be prepared to be inserted into the ML model.
    """
    def __init__(
            self,
            fe_instance: object,
            task_type: str = "both",
            test_size: float = 0.3,
            random_state: int = 42
        ):
        """
        This will be used to initialize the classification-task model builder class.

        Parameters:
            - fe_instance (object): The instance object that was used to initalize the class FeatureEngineerML.
            - task_type (str): 'regression', 'classification' or 'both'
            - test_size (float): Proportion that will be used for the train_test_split
            - random_state (int): The number inserted into the model as well as train_test_split for reproducibility (random seed)
        """

        self.fe = fe_instance
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state

        #Storage that will be used for the results
        self.clf_models = {}
        self.results = {}
        self.best_model = None

    def prepare_clf_data_for_modeling(
            self,
            df: pd.DataFrame,
            target_clf: pd.Series
    ):
        """
        This method is used to prepare the train/test splits that will be implemented during the training of the modeling.

        Parameters:
            - df (pd.DataFrame): Feature-engineered DataFrame
            - target_var (pd.Series): The Series structure associated with the classification target variable 'Indicator'
            - clf_pipeline: Preprocessing pipeline that was previously generated for the classification data.

        Returns:
            - tuple (tuple): A tuple which will contain the following information:
                - X_train_clf (the X train split for the classification data)
                - X_test_clf (the X test split for the classification data)
                - y_train_clf (the y train split for the classification data)
                - y_test clf (the y test split for the classification data)
        """
        X = df.drop(columns=['Value', 'Indicator'], errors='ignore')
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, target_clf, test_size=self.test_size, random_state=self.random_state, stratify=target_clf) #Stratify parameter will ensure that the training and testing splits will have the same proportion of the classes/labels as the original dataset.

        print(f"{len(X_train_clf)} in train data, {len(X_test_clf)} in test data prepared for the classification data for the ML models.")
        return X_train_clf, X_test_clf, y_train_clf, y_test_clf
    
    def build_clf_models(
            self,
            X_train: np.array,
            y_train: np.array,
            prepipeline: Pipeline,
            use_grid_search: bool=True
    ):
        """
        This method will create all of the models that are designated for the classification-task. 
        In this case, the classification task will be testing three models:
            - Gradient Boosting Classifier
            - Stochastic Gradient (SGD) Classifier
            - Logistic Regression
        
        If the boolean parameter "use_grid_search" will be marked as "True," the method will also tune the hyperparameters and insert them back into the models prior
        to displaying the best models.

        Parameters:
            - X_train (np.array): The X-portion array of data that will be used to train the models.
            - y_train (np.array): The y-portion array of data that later will be used for scoring procedures.
            - prepipeline (Pipeline): The preprocessing pipeline that was generated specifically for the classification data.
            - use_grid_search (bool, optional): Determines if fine-tuning of the hyperparameters will occur. Default value is "True."

        Returns:
            - models (dict): A dictionary with all of the models that were built and fine-tuned for the classification data (GradientBoostingClassifier, SGDClassifier, and LogisiticRegression).
        """
        sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=self.random_state, class_weight='balanced', max_iter=1000, tol=1e-3, learning_rate='optimal', early_stopping=True, validation_fraction=0.1) #Early_stopping parameter helps with speed

        gradient_clf= GradientBoostingClassifier(random_state=self.random_state, n_estimators=300, max_depth=3, learning_rate=0.1, max_features='sqrt', loss='log_loss', validation_fraction=0.1)

        log_clf = LogisticRegression(penalty='l2', C=1, class_weight='balanced', random_state=self.random_state, solver='saga', max_iter=1000, multi_class='multinomial', tol=1e-3)

        self.clf_models = {
            'SGD Classifier': sgd_clf,
            'Gradient Boosting Classifier': gradient_clf,
            'Logistic Regression': log_clf
        }
    
        #Creates three new pipelines using the original preprocessor pipeline - one for each model.
        pipelines = {}
        for name, model in self.clf_models.items():
            pipelines[name] = Pipeline([
                ('preprocessor', prepipeline.named_steps['preprocessor']),
                ('feature_selector', prepipeline.named_steps['feature_selector']),
                ('model', model)
            ])

        if use_grid_search == True:
            potential_alpha = c_values = np.logspace(-5, 5, base=2, num=11)
            tol_values = np.logspace(-5, -1, base=10, num=11)
            sgd_params = {
                'model__loss': ['hinge', 'log_loss', 'modified_huber'],
                'model__learning_rate': ['constant', 'optimal', 'adaptive']
            }

            lr_values = np.logspace(-5, -5, base=2, num=11)
            gradient_params = {
               'model__learning_rate': lr_values,
               'model__criterion': ['friedman_mse', 'squared_error'],
                'model__max_leaf_nodes': [2, 3, 4]
            }
            log_params = {
                'model__tol': tol_values,
                'model__C': c_values,
                'model__solver': ['lbfgs', 'sag', 'saga']
            }

            param_grids = {
                'SGD Classifier': sgd_params,
                'Gradient Boosting Classifier': gradient_params,
                'Logistic Regression': log_params
            }
            for name, pipeline in pipelines.items():
                if param_grids[name]:
                    grid_search = RandomizedSearchCV(
                        pipeline,
                        param_grids[name],
                        n_iter=5,
                        cv=3,
                        scoring='f1_macro',
                        verbose=1
                    )
                    grid_search.fit(X_train, y_train)

                self.results[name] = {
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'best_estimator': grid_search.best_estimator_,
                    'cv_results': grid_search.cv_results_
                }
            
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['best_score'])
            self.best_model = self.results[best_model_name]['best_estimator']

            return self.results