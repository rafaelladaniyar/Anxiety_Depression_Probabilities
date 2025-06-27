import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, PredictionErrorDisplay
from sklearn.pipeline import Pipeline
import seaborn as sns
import warnings

warnings.filterwarnings('ignore') # will compress all of the warnings that are derived from sklearn library when the models are being built and when it is trying to push out information.

class PopulationPercentRegressor:
    """
    The purpose of this class is to design, build and find the best regression-task ML model that can be implemented towards solving the second part of the problem that was put into place
    by the mental health dataset - determination of the percentage of the population group specified that will display the symptoms of the mental condition specified by the first ML model.

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
        self.reg_models = {}
        self.results = {}
        self.best_model = None

    def prepare_reg_data_for_modeling(
            self,
            df: pd.DataFrame,
            target_reg: pd.Series
    ):
        """
        This method is used to prepare the train/test splits that will be implemented during the training of the modeling.

        Parameters:
            - df (pd.DataFrame): Feature-engineered DataFrame
            - target_var (pd.Series): The Series structure associated with the classification target variable 'Indicator'
            - clf_pipeline: Preprocessing pipeline that was previously generated for the classification data.

        Returns:
            - tuple (tuple): A tuple which will contain the following information:
                - X_train_reg (the X train split for the classification data)
                - X_test_reg (the X test split for the classification data)
                - y_train_reg (the y train split for the classification data)
                - y_test reg (the y test split for the classification data)
        """
        X = df.drop(columns=['Value', 'Indicator'], errors='ignore')
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, target_reg, test_size=self.test_size, random_state=self.random_state) #Do not need to stratify for regression models.

        print(f"{len(X_train_reg)} in train data, {len(X_test_reg)} in test data prepared for the classification data for the ML models.")
        return X_train_reg, X_test_reg, y_train_reg, y_test_reg
    
    def build_reg_models(
            self,
            X_train: np.array,
            y_train: np.array,
            prepipeline: Pipeline,
            use_grid_search: bool=True
    ):
        """
        This method will create all of the models that are designated for the classification-task. 
        In this case, the classification task will be testing three models:
            - Gradient Boosting Regressor
            - Multi-layer Perceptron regressor
            - Linear Regression
        
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
        gradient_reg = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=300, max_depth=3, max_features='sqrt', criterion='friedman_mse', tol=1e-4)

        neural_reg = MLPRegressor(random_state=self.random_state, activation='relu', solver='lbfgs', alpha=0.001, max_iter=300, tol=1e-3, early_stopping=True)

        linear_reg = LinearRegression(copy_X=True)

        self.reg_models = {
            'Gradient Boosting Regressor': gradient_reg,
            'Multi-layer Perceptron Regressor': neural_reg,
            'Linear Regression': linear_reg
        }
    
        #Creates three new pipelines using the original preprocessor pipeline - one for each model.
        pipelines = {}
        for name, model in self.reg_models.items():
            pipelines[name] = Pipeline([
                ('preprocessor', prepipeline.named_steps['preprocessor']),
                ('feature_selector', prepipeline.named_steps['feature_selector']),
                ('model', model)
            ])

        if use_grid_search == True:
            potential_alpha = c_values = np.logspace(-5, 5, base=2, num=11)
            tol_values = np.logspace(-5, -1, base=10, num=11)
            mlp_params = {
                'model__activation': ['identity', 'logistic', 'relu'],
                'model__solver': ['lbfgs', 'adam'],
                'model__alpha': potential_alpha
            }

            lr_values = np.logspace(-5, -5, base=2, num=11)
            gradient_params = {
                'model__learning_rate': lr_values,
                'model__criterion': ['friedman_mse', 'squared_error'],
                'model__max_leaf_nodes': [3, 4, 5, 7]
            }
            linear_params = {}

            param_grids = {
                'Gradient Boosting Regressor': gradient_params,
                'Multi-layer Perceptron Regressor': mlp_params,
                'Linear Regression': linear_params
            }
            for name, pipeline in pipelines.items():
                if param_grids[name]:
                    grid_search = RandomizedSearchCV(
                        pipeline,
                        param_grids[name],
                        n_iter=5,
                        scoring='neg_mean_squared_error',
                        cv=3,
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
        
