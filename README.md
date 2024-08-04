# Deep Learning with TensorFlow

This project demonstrates how to build, train, and evaluate deep learning models using TensorFlow. The code includes examples of experimenting with different model architectures, learning rates, and training epochs to optimize performance. The project is primarily focused on solving a binary classification problem.

## Project Structure

The project directory contains the following main components:

- **data**: Directory where the dataset is stored.
- **logs/**: Contains logs generated during the execution.
- **src**: Source code for loading, processing data, model creation, training, and evaluation.
  - **data/load_process.py**: Functions to load and process the dataset.
  - **model/train_model.py**: Functions to create, compile, and train the TensorFlow models.
  - **model/evaluate_model.py**: Functions to evaluate the trained models.
  - **utils/utils.py**: Utility functions including logging setup.
- **storage/**: Directory where generated images and results are saved.
- **main.py**: Main script that orchestrates the entire pipeline from data loading to model evaluation.


## Dependencies
Plese see the requirement.txt file

## How to Run
To execute the project, navigate to your project directory and run the main.py script. Ensure that Python is installed and accessible via your command line:

python main.py

