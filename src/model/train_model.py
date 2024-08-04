import tensorflow as tf
from src.utils.utils import setup_logging

logger = setup_logging()

def create_train_model(layers, x_train, y_train, epochs=50, lr_scheduler=0, learning_rate=0, loss_val=0):
    try:
        # set a fixed random seed for the model's weight initialization
        tf.keras.utils.set_random_seed(42)

        model = None
       
        # 1. Create the model using the Sequential API
        model = tf.keras.Sequential()
        for layer in layers:
            model.add(tf.keras.layers.Dense(units=layer['units'], activation=layer.get('activation', None)))

        if learning_rate > 0:
            # Use the specified learning rate
            optimizer_value = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer_value = tf.keras.optimizers.SGD()
            
        if loss_val:
           loss_value = "binary_crossentropy" 
        else:
           loss_value = tf.keras.losses.BinaryCrossentropy()
        
        # 2. Compile the model
        model.compile(loss=loss_value, 
                      optimizer=optimizer_value,
                      metrics=['accuracy'])                
        
        if lr_scheduler:
            lrscheduler = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * 0.9**(epoch/3)
            )
            history = model.fit(x_train, y_train, epochs=epochs, verbose=0, callbacks=[lrscheduler])
        else:
            history = model.fit(x_train, y_train, epochs=epochs, verbose=0)
            
        print(model.evaluate(x_train, y_train))
        
        return model, history
            
    except Exception as e:
        logger.error(f"Error in Train_model: {e}")
        raise e
