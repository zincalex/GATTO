from sklearn.metrics import confusion_matrix
import seaborn as sns
import stellargraph as sg
import matplotlib.pyplot as plt

def plot_and_save_confusion_matrix(y_true_labels, y_pred_labels, class_names, save_path="confusion_matrix.png"):
        """
        Plots and saves a confusion matrix.

        Parameters:
        - y_true_labels: True labels.
        - y_pred_labels: Predicted labels.
        - class_names: List of class names.
        - save_path: Path to save the confusion matrix plot.
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_names)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the plot
        plt.savefig(save_path)
        #plt.show()

def plot_and_save_training_history(history, plot_path) : 
    plt.plot(sg.utils.plot_history(history, return_figure=True))
    plt.savefig(plot_path)
    plt.close()