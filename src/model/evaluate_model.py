import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from src.utils.utils import setup_logging, save_image

logger = setup_logging()

def evaluate_model(model, x_test, y_test):
    try:
        y_preds = model.predict(x_test)
        print(y_preds[:3])
        y_preds = tf.round(model.predict(x_test))
        print(y_preds[:3])
        accuracy = accuracy_score(y_test, y_preds)
        return accuracy
    except Exception as e:
        logger.error(f"Error in Evaluate_model: {e}")
        raise e

def plot_learning_rate_vs_loss(history):
    try:
        lrs = 1e-5 * (10 ** (np.arange(100) / 20))
        plt.figure(figsize=(10, 7))
        plt.semilogx(lrs, history.history["loss"])  # we want the x-axis (learning rate) to be log scale
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning rate vs. loss")
        
        # Save the confusion matrix as an image
        figure = plt.gcf()  # Get the current figure
        save_image(figure, 'learningrate_loss.png')
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in Evaluate_model: {e}")
        raise e

def plot_training_curves(history):
    try:
        pd.DataFrame(history.history).plot()
        plt.title("Model training curves")
        
        # Save the confusion matrix as an image
        figure = plt.gcf()  # Get the current figure
        save_image(figure, 'model_training_curves.png')
        
        plt.show()
    except Exception as e:
        logger.error(f"Error in Evaluate_model: {e}")
        raise e
