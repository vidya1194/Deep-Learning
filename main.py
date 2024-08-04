from src.data.load_process import load_data, process_data
from src.model.train_model import create_train_model
from src.model.evaluate_model import evaluate_model, plot_learning_rate_vs_loss, plot_training_curves
from src.utils.utils import setup_logging
import tensorflow as tf

logger = setup_logging()

def main():
    
    try:
    
        # Load data
        df = load_data('data/employee_attrition.csv')
        logger.info("Data loaded")

        # Load and process data
        x_train, x_test, y_train, y_test = process_data(df)
        logger.info("Data Processed")
        
        # Model with 5 Epochs
        logger.info("Model with 5 Epochs")
        layers = [{'units': 1}]
        model, history = create_train_model(layers, x_train, y_train, epochs=5)
                      
        # Model with 100 Epochs
        logger.info("Model with 100 Epochs")
        layers = [{'units': 1}]
        model, history = create_train_model(layers, x_train, y_train, epochs=100)  
        
        # Model with extra layer
        logger.info("Model with extra layer") 
        layers = [{'units': 1}, {'units': 1}]
        model, history = create_train_model(layers, x_train, y_train, epochs=50) 
        
        # Model with more neurons
        logger.info("Model with more neurons")  
        layers = [{'units': 2}, {'units': 1}]
        model, history = create_train_model(layers, x_train, y_train, epochs=50)
        
        #Adding more neurons to a hidden layer can sometimes decrease the accuracy of a model due to overfitting.
        #Keep the neuron 1 and add new layer with 1 neuron.
        logger.info("Model with more layer")  
        layers = [{'units': 1}, {'units': 1}, {'units': 1}]
        model, history = create_train_model(layers, x_train, y_train, epochs=50) 
        
        #Model with new learning rate
        logger.info("Model with new learning rate")
        layers = [{'units': 1}, {'units': 1}]
        model, history = create_train_model(layers, x_train, y_train, epochs=50, lr_scheduler=0,learning_rate=0.0009) 
          
        #Finding the best learning rate
        logger.info("Finding the best learning rate")
        layers = [{'units': 1}, {'units': 1}]
        model, history = create_train_model(layers, x_train, y_train, epochs=50, lr_scheduler=1,learning_rate=0.0009,loss_val=1)
        
        # Plot the learning rate versus the loss
        logger.info("Plot the learning rate versus the loss")
        #plot_learning_rate_vs_loss(history)
        
        #Activation Function
        logger.info("Model with Activation Functions")
        layers = [{'units': 1}, {'units': 1, 'activation': 'sigmoid'}]
        model, history = create_train_model(layers, x_train, y_train, epochs=50, lr_scheduler=0, learning_rate=0.0009) 
        
        # Evaluate model
        accuracy = evaluate_model(model, x_test, y_test)
        print(f"Test accuracy: {accuracy}")
        
        # Plot results
        plot_training_curves(history)
                
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise e
    
if __name__ == "__main__":
    main()
