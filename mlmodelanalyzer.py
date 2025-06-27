import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, precision_score, classification_report, confusion_matrix, accuracy_score, root_mean_squared_error, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


class MLModelAnalyzer:
    """
    This class will analyze and extract all of the insights from the GridSearchCV results for both the regression and classification results.
    """
    def __init__(self, regression_results=None, classification_results=None):
        """
        Initializes the class using the results that were obtained after completing the GridSearchCV for all 6 ML Models.

        Parameters:
            - regression_results (dict): results obtained after using the regression model builder
            - classification_results (dict): results obtained after using the classification model builder
        """
        self.reg_results = regression_results
        self.clf_results = classification_results
        self.best_models = {}
        self.performance_summary = {}

    def extract_best_models(self):
        """
        Will find the best models from GridSearchCV results.
        """

        #Extracts all of the regression results and find the best model for each ML model type
        if self.reg_results:
            for model_name, result in self.reg_results.items():
                if hasattr(result, 'best_estimator_'):
                    best_model = result.best_estimator_
                    best_score = result.best_score_
                    best_params = result.best_params_
                elif isinstance(result, dict) and 'best_estimator' in result:
                    best_model = result['best_estimator']
                    best_score = result.get('best_score', 'Unknown')
                    best_params = result.get('best_params', {})
                else:
                    best_model = result
                    best_score = 'Unknown'
                    best_params = {}
                
                self.best_models[f"Regression_{model_name}"] = best_model
                print(f"{model_name}")
                print(f"Best Score: {best_score}")
                print(f"Best Params: {best_params}")
        
        #Will extract all of the classification models and find the best models for each respective ML model type
        if self.clf_results:
            for model_name, result in self.clf_results.items():
                if hasattr(result, 'best_estimator_'):
                    best_model = result.best_estimator_
                    best_score = result.best_score_
                    best_params = result.best_params_
                elif isinstance(result, dict) and 'best_estimator' in result:
                    best_model = result['best_estimator']
                    best_score = result.get('best_score', 'Unknown')
                    best_params = result.get('best_params', {})
                else:
                    best_model = result
                    best_score = 'Unknown'
                    best_params = {}
                
                self.best_models[f"Classification_{model_name}"] = best_model
                print(f"{model_name}")
                print(f"Best Score: {best_score}")
                print(f"Best Params: {best_params}")
        return self.best_models
    
    def generating_performance_scores(self, X_test_reg=None, y_test_reg=None, X_test_clf=None, y_test_clf=None):
        """
        Will generate predictive scores and performance summarys for all of the best models.

        Parameters:
            - X_test_reg: The X test split for the regression data
            - y_test_reg: The y test split for the regression data
            - X_test_clf: The X test split for the classification data
            - y_test_clf: The y test split for the classification data
        
        Returns:
            - evaluation_results (dict): A dictionary containing all of the performance results for all of the best models.
        """
        evaluation_results = {}
        
        #Will Evaluate all of the regression models
        if X_test_reg is not None and y_test_reg is not None:
            for model_name, model in self.best_models.items():
                if 'Regression' in model_name:
                    try:
                        y_hat = model.predict(X_test_reg)
                        r2 = r2_score(y_test_reg, y_hat)
                        rmse = root_mean_squared_error(y_test_reg, y_hat)
                        mse = mean_squared_error(y_test_reg, y_hat)

                        evaluation_results[model_name] = {
                            'type': 'regression',
                            'r2_score': r2,
                            'Root MSE': rmse,
                            'MSE': mse,
                            'Predictions': y_hat,
                            'Actuals': y_test_reg
                        }

                        print(f"{model_name.replace('Regression_', '')}:")
                        print(f"R2 Score: {r2:.4f}")
                        print(f"RMSE: {rmse:.4f}")
                        print(f"MSE: {mse:.4f}")
                        print("-" * 30)
                    except Exception as e:
                        print(f"{model_name} eval failed: {str(e)}")
    
        # FIXED: Changed 'Regression' to 'Classification' in the condition
        if X_test_clf is not None and y_test_clf is not None:
            for model_name, model in self.best_models.items():
                if 'Classification' in model_name:  # FIXED: was 'Regression'
                    try:
                        y_hat = model.predict(X_test_clf)
                        accuracy = accuracy_score(y_test_clf, y_hat)
                        f1 = f1_score(y_test_clf, y_hat, average='macro')
                        precision = precision_score(y_test_clf, y_hat, average='macro')
                        recall = recall_score(y_test_clf, y_hat, average='macro')  # FIXED: was 'macrp'

                        evaluation_results[model_name] = {
                            'type': 'classification',
                            'accuracy_score': accuracy,  # FIXED: key name consistency
                            'f1_score': f1,
                            'Precision': precision,
                            'Recall': recall,
                            'Predictions': y_hat,
                            'Actuals': y_test_clf,
                            'confusion_matrix': confusion_matrix(y_test_clf, y_hat)
                        }

                        print(f"{model_name.replace('Classification_', '')}:")
                        print(f"F1 Score: {f1:.4f}")
                        print(f"Accuracy Score: {accuracy:.4f}")
                        print(f"Precision: {precision:.4f}")
                        print(f"Recall Score: {recall:.4f}")
                        print("-" * 30)
                    except Exception as e:
                        print(f"{model_name} eval failed: {str(e)}")
        
        self.performance_summary = evaluation_results
        return evaluation_results
    
    def create_performance_plots(self):
        """
        This method will generate performance comparison plots for all of the ML models that were generated throughout the project.
        """
        
        # FIXED: Removed duplicate variable assignment
        reg_models = {k: v for k, v in self.performance_summary.items() if v['type'] == 'regression'}
        
        if reg_models:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            modelnames = [k.replace('Regression_', '') for k in reg_models.keys()]
            r2_scores = [v['r2_score'] for v in reg_models.values()]
            rmse_scores = [v['Root MSE'] for v in reg_models.values()]

            # FIXED: bar_label method usage
            bars1 = axes[0].bar(modelnames, r2_scores, color='orange', alpha=0.7, edgecolor='black')
            axes[0].set_title('Regression Models - R2 Score Comparison')
            axes[0].set_ylabel('R2 Score')
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis='x', rotation=45)  # Rotate labels for better readability
            
            # FIXED: Proper bar labeling
            for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.4f}', ha='center', va='bottom')
            
            bars2 = axes[1].bar(modelnames, rmse_scores, color='orange', alpha=0.7, edgecolor='black')
            axes[1].set_title('Regression Models - RMSE Comparison')
            axes[1].set_ylabel('RMSE')
            axes[1].tick_params(axis='x', rotation=45)
            
            # FIXED: Proper bar labeling
            for i, (bar, score) in enumerate(zip(bars2, rmse_scores)):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_scores)*0.01, 
                           f'{score:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()

        # FIXED: Moved classification plotting outside of regression block
        clf_models = {k: v for k, v in self.performance_summary.items() if v['type'] == 'classification'}
        if clf_models:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            modelnames = [k.replace('Classification_', '') for k in clf_models.keys()]
            accuracies = [v['accuracy_score'] for v in clf_models.values()]  # FIXED: key name
            f1_scores = [v['f1_score'] for v in clf_models.values()]

            bars1 = axes[0].bar(modelnames, accuracies, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[0].set_title('Classification Models - Accuracy Comparison')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_ylim(0, 1)
            axes[0].tick_params(axis='x', rotation=45)
            
            # FIXED: Proper bar labeling
            for i, (bar, score) in enumerate(zip(bars1, accuracies)):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.4f}', ha='center', va='bottom')
            
            bars2 = axes[1].bar(modelnames, f1_scores, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1].set_title('Classification Models - F1-Score Comparison')
            axes[1].set_ylabel('F1-Score')
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis='x', rotation=45)
            
            # FIXED: Proper bar labeling and correct axis reference
            for i, (bar, score) in enumerate(zip(bars2, f1_scores)):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()

    def extract_feature_importance(self):  # FIXED: Method name typo
        """
        Can extract feature importance from the best models that were found for each task.
        """
        feature_importance_results = {}
        
        # FIXED: Iterate through actual best_models structure
        for model_name, model in self.best_models.items():
            try:
                print(f"\n Analyzing {model_name}:")
                
                # Extract feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Get top 15 features
                    top_indices = np.argsort(importances)[-15:]
                    top_importances = importances[top_indices]
                    
                    print(f"Top 15 Most Important Features:")
                    for i, (idx, imp) in enumerate(zip(reversed(top_indices), reversed(top_importances))):
                        print(f"    {i+1:2d}. Feature_{idx}: {imp:.4f}")
                    
                    feature_importance_results[model_name] = {
                        'importances': importances,
                        'top_indices': top_indices,
                        'top_importances': top_importances
                    }
                    
                    # Create visualization
                    plt.figure(figsize=(10, 8))
                    plt.barh(range(len(top_importances)), top_importances)
                    plt.yticks(range(len(top_importances)), [f'Feature_{idx}' for idx in reversed(top_indices)])
                    plt.xlabel('Feature Importance')
                    plt.title(f'{model_name} - Top 15 Feature Importances')
                    plt.tight_layout()
                    plt.show()
                    
                elif hasattr(model, 'coef_'):
                    # Linear model coefficients
                    coefs = model.coef_
                    if len(coefs.shape) > 1:
                        coefs = np.mean(np.abs(coefs), axis=0)  # Average across classes
                    else:
                        coefs = np.abs(coefs)
                    
                    top_indices = np.argsort(coefs)[-15:]
                    top_coefs = coefs[top_indices]
                    
                    print(f"Top 15 Most Important Features (by coefficient magnitude):")
                    for i, (idx, coef) in enumerate(zip(reversed(top_indices), reversed(top_coefs))):
                        print(f"    {i+1:2d}. Feature_{idx}: {coef:.4f}")
                    
                    feature_importance_results[model_name] = {
                        'coefficients': coefs,
                        'top_indices': top_indices,
                        'top_coefficients': top_coefs
                    }
                    
                    # Create visualization
                    plt.figure(figsize=(10, 8))
                    plt.barh(range(len(top_coefs)), top_coefs)
                    plt.yticks(range(len(top_coefs)), [f'Feature_{idx}' for idx in reversed(top_indices)])
                    plt.xlabel('Coefficient Magnitude')
                    plt.title(f'{model_name} - Top 15 Feature Coefficients')
                    plt.tight_layout()
                    plt.show()
                    
                else:
                    print(f"No feature importance available for this model type")
                    
            except Exception as e:
                print(f"Error analyzing {model_name}: {str(e)}")
        
        self.feature_importance_results = feature_importance_results
        return feature_importance_results