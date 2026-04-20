import matplotlib.pyplot as plt
import numpy as np
import torch
import os


# Dataset Classes Frequency
def compute_class_frequency(dataset):
    levelDisease = ['CN', 'MCI', 'AD']
    counter_CN = 0
    counter_MCI = 0
    counter_AD = 0
    for entry in dataset:
        if entry["ADType"][0] == 1:
            counter_CN += 1
        if entry["ADType"][1] == 1:
            counter_MCI += 1
        if entry["ADType"][2] == 1:
            counter_AD += 1
            
    total_samples = counter_CN+counter_MCI+counter_AD                   #; print(f"Total Sample: {total_samples}")
    frequencies = [counter_CN, counter_MCI, counter_AD]                 #; print(f"Frequencies: {frequencies}")
    scaled_frequencies = [freq/total_samples for freq in frequencies]   #; print(f"Scaled Frequencies: {scaled_frequencies}")
    
    return total_samples, frequencies, scaled_frequencies

# Plotting Class Frequency
def plot_class_frequency(frequencies):    
    levelDisease = ['CN', 'MCI', 'AD']
    
    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Set the bar width
    bar_width = 0.5

    # Set the positions of the bars on the x-axis
    x_pos = range(len(levelDisease))

    # Plot the histogram
    ax.bar(x_pos, frequencies, bar_width, align='center', color='lightskyblue')

    # Set the x-axis labels to the class names
    ax.set_xticks(x_pos)
    ax.set_xticklabels(levelDisease)

    # Set the y-axis label
    ax.set_ylabel('Frequency')
    scale_factor = 5
    plt.yticks(np.arange(1, max(frequencies), int(max(frequencies)/scale_factor)))

    # Set the title of the plot
    ax.set_title('Class Frequency')

    # Display the plot
    plt.show()

# Confusion Matrix of the Experiment
def plot_confusion_matrix(cm, out_class:list, cmap:str="Reds", title:str=None, save_cm:bool=False, save_dir:str=None, file_name:str=None,
                          verbose:bool=False):
    # Use the labels that are in our dataset
    classes = out_class
    
    # if normalize:
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm[np.isnan(cm)] = 0  # Handle NaNs (zero division)

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 3)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Main Loop over data dimensions
    fmt = '.2f' # if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]*100, fmt) + "%", # if normalize else format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black")  # Set text color to black

    fig.tight_layout()
    # Save the plot as a PNG image
    if save_cm == True: plt.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    if verbose: plt.show(ax)

# Plot of Batch features
def plot_features_in_the_batch(batch_tensor:torch.Tensor, samples:int=2, verbose:bool=False, color_map:str="gray"):
    for batch_sample in batch_tensor:
        for i in range(samples):
            if verbose ==True: print(batch_sample[i])
            
            # Create a figure and a 1x3 grid of subplots arranged horizontally
            fig, axes = plt.subplots(1, 3, figsize=(20, 3))
            
            axial = batch_sample[i].detach().cpu().numpy()[:, :, batch_sample[i].shape[2]//2]
            coronal = np.rot90(batch_sample[i].detach().cpu().numpy()[:, batch_sample[i].shape[1]//2, :], k=-1)     # counter-clockwise
            sagittal = np.rot90(batch_sample[i].detach().cpu().numpy()[batch_sample[i].shape[0]//2, :, :], k=-1)    # counter-clockwise
            print("Batch Example Shape: ", batch_sample[i].shape)
            # Plot the data on each subplot
            im_0 = axes[0].imshow(axial, cmap=color_map)
            axes[0].set_title('Assiale (Sopra-Sotto)')
            im_1 = axes[1].imshow(coronal, cmap=color_map)
            axes[1].set_title('Coronale (Fronte-Retro)')
            im_2 = axes[2].imshow(sagittal, cmap=color_map)
            axes[2].set_title('Sagittal (Laterale)')

            # Add colorbars
            cbar0 = fig.colorbar(im_0, ax=axes[0])
            cbar0.set_label('Intensity')

            cbar1 = fig.colorbar(im_1, ax=axes[1])
            cbar1.set_label('Intensity')

            cbar2 = fig.colorbar(im_2, ax=axes[2])
            cbar2.set_label('Intensity')

            # Adjust spacing between subplots
            plt.tight_layout()
            # Show the plot
            plt.show()
        print("#######")