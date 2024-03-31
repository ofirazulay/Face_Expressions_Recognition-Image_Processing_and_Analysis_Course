#%%
#imports

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import skimage as ski
#from image_common_utils import show_image
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.io import imsave
from skimage import io, color
import cv2 
from skimage.color import rgb2gray
from skimage import img_as_ubyte, io
from PIL import Image
from colorspacious import cspace_convert
from pathlib import Path
import seaborn as sns
import cv2
from skimage import exposure, filters
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import keras
from keras import layers
from keras import initializers
from keras import backend as K
from keras import Sequential
from keras.layers import Dense, MaxPool2D,Conv2D,Flatten,Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

#check current environment
def check_current_environment():
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"You are currently in the '{conda_env}' environment.")
    else:
        print("You are not in a Conda environment.")
        
check_current_environment()


#check libraries versions
print(pd.__version__)
print(np.__version__)
print(tf.__version__)
print(ski.__version__)

seed_value = 42; #set_random_seed(seed_value) # Keras uses its source of randomness regardless Theano or TensorFlow.In addition, TensorFlow has its own random number generator that must also be seeded by calling the set_random_seed() function immediately after the NumPy random number generator:
np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)

#%%
### import images

###------------------------------------Import Images------------------------------------###
os.chdir(r"C:\Users\peleg\Desktop\Image_Processing\image_processing_course THE ARCHIVE WITH TUTORIALS AND ENV-20240106\Project")

def convert_img_to_df(dataset, file_format):
    img_dir = Path(dataset)
    filename = list(img_dir.glob(f'**/*.{file_format}'))  # Corrected glob pattern
    label = [os.path.split(os.path.split(x)[0])[1] for x in filename]  # Using list comprehension

    filename = pd.Series(filename, name='Filepath').astype(str)
    label = pd.Series(label, name='Label')
    img_df = pd.concat([filename, label], axis=1)
    return img_df

def data_dir_to_df(data_dir): 
    img_df1= convert_img_to_df(data_dir, 'jpg')
    img_df2 = convert_img_to_df(data_dir, 'jpeg')
    img_df3 = convert_img_to_df(data_dir, 'png')
    img_df4 = convert_img_to_df(data_dir, 'gif')
    img_df = pd.concat([img_df1,img_df2,img_df3,img_df4], axis=0)
    img_df = img_df.sort_values(by='Label')
    return img_df

img_df = data_dir_to_df("Data")


#%%

###------------------------------------Distribution of images labels------------------------------------###
#Label Count
label_count = img_df['Label'].value_counts()

#Setting
plt.figure(figsize=(8, 8))
sns.set_theme(style='darkgrid', palette='pastel')
color = sns.color_palette(palette='pastel')
explode = [0.02]*len(label_count)

#Pie chart
plt.pie(label_count.values, labels=label_count.index, autopct='%1.1f%%', colors=color, explode=explode, textprops={'fontsize': 14})
plt.title('labels Distribution', fontsize=18, fontweight='bold')

#Show
plt.show()


#Label Count
label_count = img_df['Label'].value_counts()

#Setting
plt.figure(figsize=(8, 8), facecolor='none')  # Set facecolor to 'none' for transparent background
sns.set_theme(style='darkgrid', palette='Paired')
color = sns.color_palette(palette='Dark2')
explode = [0.02]*len(label_count)

#Pie chart
plt.pie(label_count.values, labels=label_count.index, autopct='%1.1f%%', colors=color, explode=explode, textprops={'fontsize': 26, 'color': 'white'})  # Change text color to white
plt.title('labels Distribution', fontsize=30, fontweight='bold', color='white')  # Increase title fontsize and set it to bold, and change color to white

#Show
plt.show()


#%%
### histogram functions

##---------------------------Histograms----------------------------##

#---Present the Cumulative histogram vs. histogram of your image for each of the three bands. 
from skimage import exposure

#This funcrion convert image loaded by OpenCV that is in BGR mode to RBG mode
#and use Matplotlib to display the image (Matplotlib displays in RGB mode)
def convert_cv_Img_from_BGR_2_RGB_and_display_by_Matplotlib(original_cv_img, hide_x_y_axis):
    img_RGB = cv2.cvtColor(original_cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    if(hide_x_y_axis):
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    return(img_RGB)
  
#---- Common function

def resize_image(image, width=None, height=None):
    # Get the current size of the image
    h, w = image.shape[:2]

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # Calculate aspect ratio
    if width is None:
        aspect_ratio = height / float(h)
        new_width = int(w * aspect_ratio)
        new_height = height
    elif height is None:
        aspect_ratio = width / float(w)
        new_width = width
        new_height = int(h * aspect_ratio)
    else:
        new_width = width
        new_height = height

    # Resize the image
    resized_img = cv2.resize(image, (new_width, new_height))

    return resized_img


def plot_histogram(image, ax=None, **kwargs):
    ax = ax if ax is not None else plt.gca()
    # `channel` is the red, green, or blue channel of the image.
    for channel, channel_color in zip(iter_channels(image), 'rgb'):
        _plot_histogram(ax, channel, color=channel_color, **kwargs)
    return

# Helper of above
def _plot_histogram(ax, image, alpha=0.3, **kwargs):
    # Use skimage's histogram function which has nice defaults for integer and float images.
    hist, bin_centers = exposure.histogram(image)
    ax.fill_between(bin_centers, hist, alpha=alpha, **kwargs)
    ax.set_xlabel('intensity')
    ax.set_ylabel('# pixels')
    return

# Helper of above
def iter_channels(color_image):
    # Roll array-axis so that we iterate over the color channels of an image.
    for channel in np.rollaxis(color_image, -1):
        yield channel
    return

# Helper of above
def match_axes_height(ax_src, ax_dst):
    plt.draw()
    dst = ax_dst.get_position()
    src = ax_src.get_position()
    ax_dst.set_position([dst.xmin, src.ymin, dst.width, src.height])
    return


def imshow_with_histogram(image, **kwargs):
    width, height = plt.rcParams['figure.figsize']
    fig, (ax_image, ax_hist) = plt.subplots(ncols=2, figsize=(2*width, height))
    kwargs.setdefault('cmap', plt.cm.gray)
    ax_image.imshow(image, **kwargs)
    plot_histogram(image, ax=ax_hist)
    # pretty it up
    ax_image.set_axis_off()
    match_axes_height(ax_image, ax_hist)
    return ax_image, ax_hist


# Plot CDF
def plot_cdf(image, ax=None , color='black'):
    img_cdf, bins = exposure.cumulative_distribution(image)
    ax.plot(bins, img_cdf, color)
    ax.set_ylabel("Fraction of pixels below intensity")
    return


# Function to plot united histogram and cumulative distribution plot
def plot_united_histogram_and_cdf(image):
    width, height = plt.rcParams['figure.figsize']
    fig, (ax_image, ax_united) = plt.subplots(ncols=2, figsize=(2*width, height))

    # Plot image with united histogram
    kwargs = {'cmap': plt.cm.gray}
    ax_image.imshow(image, **kwargs)
    plot_histogram(image, ax=ax_united)
    ax_image.set_axis_off()
    match_axes_height(ax_image, ax_united)
    
    # Plot united cumulative distribution plot
    ax_cdf_united = ax_united.twinx()
    plot_cdf(image, ax=ax_cdf_united, color='black')

    # Plot individual RGB channel cumulative distribution plots
    colors = ['r', 'g', 'b']
    for channel, color in zip(iter_channels(image), colors):
        plot_cdf(channel, ax=ax_cdf_united, color=color)

    plt.tight_layout()
    plt.show()


def plot_rgb_histograms(image):
    width, height = plt.rcParams['figure.figsize']
    fig, axs = plt.subplots(1, 3, figsize=(3*width, height))

    # Plot individual RGB channel histograms
    colors = ['r', 'g', 'b']
    for i, (channel, color) in enumerate(zip(iter_channels(image), colors)):
        ax_hist = axs[i]
        _plot_histogram(ax_hist, channel, color=color)
        plot_cdf(channel, ax=ax_hist.twinx(), color=color)
        # pretty it up
        ax_hist.set_title(f'Channel {color.upper()}')
    plt.tight_layout()
    plt.show()



#%%
#### Detecting faced from the img_df and storing them in Face_detected folder ####

def detect_and_show_face_in_image(image_for_face_detection, face_cascade_classifier, sz):
    # Convert to gray
    gray = cv2.cvtColor(image_for_face_detection, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))    

    faces_list = []
    for (x, y, w, h) in faces:
        # Sanity test
        if h <= 0 or w <= 0:
            pass
        crop_img = image_for_face_detection[y:y+h, x:x+w]  # Corrected the order of slicing
        faces_list.append(crop_img)
    return faces_list

# get classifiers
face_cascade_classifier = cv2.CascadeClassifier('../model/haarcascades/haarcascade_frontalface_default.xml')
if face_cascade_classifier.empty():
    print('Missing face classifier xml file')


        
def detect_df_faces():
    for file_path in img_df["Filepath"]:
        
        image_for_face_detection = cv2.imread(file_path) 
        sz = (image_for_face_detection.shape[1], image_for_face_detection.shape[0])
    
        # do face detection
        detected_faces = detect_and_show_face_in_image(image_for_face_detection, face_cascade_classifier, sz)
        
        directory, name = os.path.split(file_path)
        class_name = directory.split("\\")[-1]
        # Display all detected faces
        for i, face in enumerate(detected_faces):
            new_path = 'Face_detected/' + class_name +'/Face-'+str({i+1})+'_'+ name
            print(new_path)
            cv2.imwrite(new_path, face)
        
        
detect_df_faces()



#%%

### functions for diffrernt versions of pre-proccesing datasets ###

def GaussianBlur_by_kernel(image, kernel_size):
    if image is None:
        print("Input image is None.")
        return image
    else:    
        filtered_image = cv2.GaussianBlur(image, (kernel_size,kernel_size),0)
    return filtered_image


def Contrast_enhancement(image):
    num_channels = image.shape[2]
    if (num_channels == 3):
        R, G, B = cv2.split(image)
        # Convert to 8-bit unsigned integer
        B = B.astype(np.uint8)
        G = G.astype(np.uint8)
        R = R.astype(np.uint8)
        
        # Apply histogram equalization to each channel
        equalized_B = exposure.equalize_hist(B)
        equalized_G = exposure.equalize_hist(G)
        equalized_R = exposure.equalize_hist(R)
        
        # Convert back to 8-bit unsigned integer
        equalized_B = (equalized_B * 255).astype(np.uint8)
        equalized_G = (equalized_G * 255).astype(np.uint8)
        equalized_R = (equalized_R * 255).astype(np.uint8)
        
        Image_contrast = cv2.merge((equalized_R, equalized_G, equalized_B))

    else:
        image = image.astype(np.uint8)
        image_equ = exposure.equalize_hist(image)
        image_equ = (image_equ * 255).astype(np.uint8)
        RGB_image_contrast = image_equ
        
    return Image_contrast



def Apply_thresholding(image, thresholding_Type ,dark_threshold, bright_threshold):
    num_channels = image.shape[2]
    if (num_channels == 3):
        (r, g, b) = cv2.split(image)
        
        if(thresholding_Type == 'Dark'):
            #apply Dark thresholding to each channel separately
            ret1, threshold_b_Dark = cv2.threshold(b, dark_threshold, 255, cv2.THRESH_TOZERO)
            ret2, threshold_g_Dark = cv2.threshold(g, dark_threshold, 255, cv2.THRESH_TOZERO)
            ret3, threshold_r_Dark = cv2.threshold(r, dark_threshold, 255, cv2.THRESH_TOZERO)
            Img_thresholding_Dark = cv2.merge((threshold_r_Dark, threshold_g_Dark, threshold_b_Dark))
            Img_thresholding = Img_thresholding_Dark
    
     
        if(thresholding_Type == 'Bright'):
            #apply Bright thresholding to each channel separately
            ret1, mask_threshold_b = cv2.threshold(b, bright_threshold, 255, cv2.THRESH_BINARY)
            threshold_b_bright = cv2.bitwise_or(b, mask_threshold_b)
            ret2, mask_threshold_g = cv2.threshold(g, bright_threshold, 255, cv2.THRESH_BINARY)
            threshold_g_bright = cv2.bitwise_or(g, mask_threshold_g)
            ret3, mask_threshold_r = cv2.threshold(r, bright_threshold, 255, cv2.THRESH_BINARY)
            threshold_r_bright = cv2.bitwise_or(r, mask_threshold_r)
            Img_thresholding_Bright = cv2.merge((threshold_r_bright, threshold_g_bright, threshold_b_bright))
            Img_thresholding = Img_thresholding_Bright
            
    
        if(thresholding_Type == 'Dark and Bright'):
            ret1, threshold_b_Dark = cv2.threshold(b, dark_threshold, 255, cv2.THRESH_TOZERO)
            ret2, threshold_g_Dark = cv2.threshold(g, dark_threshold, 255, cv2.THRESH_TOZERO)
            ret3, threshold_r_Dark = cv2.threshold(r, dark_threshold, 255, cv2.THRESH_TOZERO)
            ret1, mask_threshold_b = cv2.threshold(threshold_b_Dark, bright_threshold, 255, cv2.THRESH_BINARY)
            threshold_b_both = cv2.bitwise_or(threshold_b_Dark, mask_threshold_b)
            ret2, mask_threshold_g = cv2.threshold(threshold_g_Dark, bright_threshold, 255, cv2.THRESH_BINARY)
            threshold_g_both = cv2.bitwise_or(threshold_g_Dark, mask_threshold_g)
            ret3, mask_threshold_r = cv2.threshold(threshold_r_Dark, bright_threshold, 255, cv2.THRESH_BINARY)
            threshold_r_both = cv2.bitwise_or(threshold_r_Dark, mask_threshold_r)
            Img_thresholding_Both = cv2.merge((threshold_r_both, threshold_g_both, threshold_b_both))
            Img_thresholding = Img_thresholding_Both

    else:
        Img_thresholding = image
        
    return Img_thresholding 


def Canny(Image, min_value, max_value):
    my_image_gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY) #convert to grayScale
    canny_edges = cv2.Canny(my_image_gray, min_value, max_value)  
    return canny_edges


# merge all preprocessing funcations
def Merge_preprocess_image(image,kernel_size, thresholding_Type ,dark_threshold, bright_threshold, min_value, max_value):
    gussianBlur_image = GaussianBlur_by_kernel(image, kernel_size)
    contrast_image = Contrast_enhancement(gussianBlur_image)
    thresholding_image = Apply_thresholding(contrast_image, thresholding_Type ,dark_threshold, bright_threshold)
    canny_image = Canny(thresholding_image, min_value, max_value)
    output_Img = canny_image
    return output_Img


#%%

#Histogram

def match_axes_height(ax_src, ax_dst):
    """Match the axes height of two axes objects."""
    plt.draw()
    dst = ax_dst.get_position()
    src = ax_src.get_position()
    ax_dst.set_position([dst.xmin, src.ymin, dst.width, src.height])

def _plot_histogram(ax, image, alpha=0.3, **kwargs):
    hist, bin_centers = exposure.histogram(image)
    ax.fill_between(bin_centers, hist, alpha=alpha, **kwargs)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('# Pixels', labelpad=-1)

def iter_channels(color_image):
    for channel in np.rollaxis(color_image, -1):
        yield channel

def plot_histogram_and_cdf(image, ax=None, **kwargs):
    ax = ax if ax is not None else plt.gca()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a single twin axis for CDF
    ax_twin = ax.twinx()

    for channel, channel_color in zip(iter_channels(image), 'rgb'):
        _plot_histogram(ax, channel, color=channel_color, label=f'{channel_color.upper()} Histogram')

        # Calculate and plot the CDF
        img_cdf, bins = exposure.cumulative_distribution(channel)
        ax_twin.plot(bins, img_cdf, color=channel_color, linestyle='--', label=f'{channel_color.upper()} CDF')

    ax.legend(loc='upper left')
    ax_twin.legend(loc='center right')  # Adjusted legend position

    # Set the y-axis limits for CDF plot
    ax_twin.set_ylim(0, 1)

    ax.set_xlabel('Intensity')
    ax.set_ylabel('# Pixels')

    return ax, ax_twin


def imshow_with_histogram_and_cdf(image, title='', **kwargs):
    width, height = plt.rcParams['figure.figsize']
    fig, (ax_image, ax_hist) = plt.subplots(ncols=2, figsize=(2 * width, height))

    kwargs.setdefault('cmap', plt.cm.gray)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to the range [0, 1]
    normalized_image = image_rgb / 255.0
    # Clip pixel values to the valid range [0, 1]
    normalized_image = np.clip(normalized_image, 0, 1)

    ax_image.imshow(normalized_image, **kwargs)
    plot_histogram_and_cdf(image, ax=ax_hist)

    ax_image.set_axis_off()
    match_axes_height(ax_image, ax_hist)
    fig.suptitle(title)

    return ax_image, ax_hist


image = cv2.imread(r'C:\Users\peleg\Desktop\Image_Processing\image_processing_course THE ARCHIVE WITH TUTORIALS AND ENV-20240106\Assignment3\assets\animal.jpeg',0)
image_for_hist = image.astype(np.float32)
img_new = Apply_thresholding(image_for_hist, 'Dark and Bright', 85, 170)
ax_image, ax_hist = imshow_with_histogram_and_cdf(img_new, 'Cumulative histogram vs. histogram for RGB Bands')

plt.show()


#%%

#### tarining model 


def CNN_Classifer(train_df_original, test_df_original, val_df_original, preprocessing_type):

    all_df_original = [train_df_original, test_df_original, val_df_original]
    column_names = ['Filepath', 'Label']
        
    train_df_preprocessing = pd.DataFrame(columns=column_names)
    test_df_preprocessing = pd.DataFrame(columns=column_names)
    val_df_preprocessing = pd.DataFrame(columns=column_names)
       
    # loop that create train, test, val datasets for each PP
    for i in range(len(all_df_original)): 
        for img_original_path in all_df_original[i]['Filepath']:
            directory, name = os.path.split(img_original_path)
            class_name = directory.split("\\")[-1]
            
            img_original = cv2.imread(img_original_path, 1)
  
            if(preprocessing_type == 'Without_Preprocessing'):
                train_df_preprocessing = train_df_original
                test_df_preprocessing = test_df_original
                val_df_preprocessing = val_df_original

            elif(preprocessing_type == 'GaussianBlur_by_kernel'):
                image_version1_GaussianBlur_by_kernel = GaussianBlur_by_kernel(img_original, 15)
                export_path = 'Data1_GaussianBlur_by_kernel/' + class_name +'/'+ name
                cv2.imwrite(export_path, image_version1_GaussianBlur_by_kernel)
                new_row = {'Filepath':export_path, 'Label':class_name}
                
            elif(preprocessing_type == 'Contrast_enhancement'):
                image_version2_Contrast_enhancement = Contrast_enhancement(img_original)
                export_path = 'Data2_Contrast_enhncement/' + class_name +'/'+ name
                cv2.imwrite(export_path, image_version2_Contrast_enhancement)
                new_row = {'Filepath':export_path, 'Label':class_name}
                
            elif(preprocessing_type == 'Apply_thresholding'):
                image_version3_Apply_thresholding = Apply_thresholding(img_original, 'Dark and Bright', 70, 200)
                export_path = 'Data3_Apply_thresholding/' + class_name +'/'+ name
                cv2.imwrite(export_path, image_version3_Apply_thresholding)
                new_row = {'Filepath':export_path, 'Label':class_name}

            elif(preprocessing_type == 'Canny'):
                image_version4_Canny =  Canny(img_original, 70, 200)
                export_path = 'Data4_Canny/' + class_name +'/'+ name
                cv2.imwrite(export_path, image_version4_Canny)
                new_row = {'Filepath':export_path, 'Label':class_name}

            elif(preprocessing_type == 'Merge_preprocess_image'):
                image_version5_Merge_preprocess_image =  Merge_preprocess_image(img_original, 7, 'Dark and Bright' ,70, 200, 70, 200)
                export_path = 'Data5_Merge_preprocess_image/' + class_name +'/'+ name
                cv2.imwrite(export_path, image_version5_Merge_preprocess_image)
                new_row = {'Filepath':export_path, 'Label':class_name}
                
       
            if (preprocessing_type != "Without_Preprocessing"):       
                if(i == 0):
                    train_df_preprocessing.loc[len(train_df_preprocessing.index)] = new_row
                if(i ==1):
                    test_df_preprocessing.loc[len(test_df_preprocessing.index)] = new_row
                if(i == 2):
                    val_df_preprocessing.loc[len(val_df_preprocessing.index)] = new_row
   # END loop that create train, test, val datasets for each PP

    
    
    train_images = []; test_images = []; val_images = []
    
    for i in range(len(train_df_preprocessing)):
        img = cv2.imread(train_df_preprocessing.iloc[i]["Filepath"])
        img = cv2.resize(img,(224, 224))
        train_images.append(img)
    
    for i in range(len(test_df_preprocessing)):
        img = cv2.imread(test_df_preprocessing.iloc[i]["Filepath"])
        img = cv2.resize(img, (224, 224))
        test_images.append(img)
    
    for i in range(len(val_df_preprocessing)):
        img = cv2.imread(val_df_preprocessing.iloc[i]["Filepath"])
        img = cv2.resize(img, (224, 224))
        val_images.append(img)




    
    model = Sequential()
    model.add(Conv2D(10,(3,3),activation='relu',input_shape=(224,224,3)))
    model.add(Conv2D(10,(3,3),activation='relu'))
    model.add(MaxPool2D((2,2),padding='valid'))
    
    model.add(Conv2D(20,(3,3),activation='relu'))
    model.add(Conv2D(20,(3,3),activation='relu'))
    model.add(MaxPool2D((2,2),padding='valid'))
    
    model.add(Conv2D(50,(3,3),activation='relu'))
    model.add(Conv2D(50,(3,3),activation='relu'))
    model.add(MaxPool2D((2,2),padding='valid'))
    
    model.add(Conv2D(20,(3,3),activation='relu'))
    model.add(Conv2D(20,(3,3),activation='relu'))
    model.add(MaxPool2D((2,2),padding='valid'))
    
    model.add(Conv2D(10,(3,3),activation='relu'))
    model.add(Conv2D(10,(3,3),activation='relu'))
    model.add(MaxPool2D((2,2),padding='valid'))
    
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(3,activation='softmax'))
    
    # model.summary()

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    



    all_labels = pd.concat([train_df_preprocessing["Label"], test_df_preprocessing["Label"], val_df_preprocessing["Label"]])

    # Fit LabelEncoder on all labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    # Encode labels to integers
    train_labels_encoded = label_encoder.transform(train_df_preprocessing["Label"])
    val_labels_encoded = label_encoder.transform(val_df_preprocessing["Label"])
    test_labels_encoded = label_encoder.transform(test_df_preprocessing["Label"])
    
    # Convert encoded labels to one-hot encoded format
    train_labels = to_categorical(train_labels_encoded, num_classes=3)
    val_labels = to_categorical(val_labels_encoded, num_classes=3)
    test_labels = to_categorical(test_labels_encoded, num_classes=3)
    


    # Train the model
    history = model.fit(x=np.array(train_images), y=train_labels,
                        validation_data=(np.array(val_images), val_labels),
                        epochs=20, 
                        verbose=1)

    
    plt.plot(history.history['accuracy'],label='Train Accuracy',c='g')
    plt.plot(history.history['val_accuracy'],label='Val Accuracy',c='r')
    plt.grid()
    plt.legend()
    plt.title(f"Accuracy by Epoch for {preprocessing_type}")
    plt.show()
    
    score = model.evaluate(np.array(test_images),test_labels,verbose=1)

    

    # Confusion matrix result
    from sklearn.metrics import classification_report, confusion_matrix
    Y_pred = model.predict(np.array(test_images), verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    # print("y_pred: ", y_pred)

    for ix in range(3):
        print(ix, confusion_matrix(np.argmax(test_labels, axis=1),y_pred)[ix].sum())
    cm = confusion_matrix(np.argmax(test_labels, axis=1),y_pred)
    print(cm)
    
    
    # Extract values from confusion matrix
    a, b, c = cm[0]
    d, e, f = cm[1]
    g, h, i = cm[2]

    # Calculate accuracy
    accuracy = (a+e+i) / (a+b+c+d+e+f+g+h+i)

    # Calculate precision
    p0 = a/(a+b+c)
    p1 = e/(d+e+f)
    p2 = i/(g+h+i)

    # Calculate recall
    r0 = a/(a+d+g)
    r1 = e/(b+e+h)
    r2 = i/(c+f+i)
    
    # Calculate f1
    f0 = 2*((p0*r0)/(p0+r0))
    f1 = 2*((p1*r1)/(p1+r1))
    f2 = 2*((p2*r2)/(p2+r2))
   
    #total
    t0 = a+b+c
    t1 = d+e+f
    t2 = g+h+i
    t = t0+t1+t2
    
    avg_precision = p0*(t0/t) + p1*(t1/t) + p2*(t2/t)
    avg_recall = r0*(t0/t) + r1*(t1/t) + r2*(t2/t)
    avg_f1 = f0*(t0/t) + f1*(t1/t) + f2*(t2/t)

    print("****** Model Summary: ******")
    print(f"{preprocessing_type} Metrics:")
    print("Accuracy: ", accuracy)
    print(f"Precision: [{p0, p1, p2}], Weighted Avg. Precision: {avg_precision}")
    print(f"Recall: [{r0, r1, r2}], Weighted Avg. Recall: {avg_recall}")
    print(f"F1-score: [{f0, f1, f2}], Weighted Avg. F1-score: {avg_f1}")


    return model, accuracy, avg_precision, avg_recall, avg_f1



Accuracy_list = []
Precision_list = []
Recall_list = []
F1_list = []
model_list = []


faces_detected_df = data_dir_to_df("Face_detected")
img_train , test_df_original = train_test_split(faces_detected_df,test_size=0.2)
train_df_original, val_df_original = train_test_split(img_train,test_size=0.2)



preprocessing_type_list = ['Without_Preprocessing', 'GaussianBlur_by_kernel', 'Contrast_enhancement', 'Apply_thresholding', 'Canny', 'Merge_preprocess_image']
for preprocessing_type in preprocessing_type_list:
    model, accuracy, avg_precision, avg_recall, avg_f1 = CNN_Classifer(train_df_original, test_df_original, val_df_original, preprocessing_type)
    Accuracy_list.append(accuracy)
    Precision_list.append(avg_precision)
    Recall_list.append(avg_recall)
    F1_list.append(avg_f1)
    model_list.append(model)
    # print("Preprocess: ", preprocessing_type)
    # print("Accuracy: ", accuracy)
    # print("Precision: ", avg_precision)
    # print("Recall: ", avg_recall)
    # print("F1: ", avg_f1)
    # print(model.summary())
    
    

print("Summary:")
print (Accuracy_list)
max_index_accuracy = Accuracy_list.index(max(Accuracy_list))
print(max_index_accuracy)
best_model_all_version = model_list[max_index_accuracy]
print(best_model_all_version)
preprocessing_of_best_model = preprocessing_type_list[max_index_accuracy]
print(preprocessing_of_best_model)


print("Accuracy:" , Accuracy_list)
print("Precision:" , Precision_list)
print("Recall:" , Recall_list)
print("F1:" , F1_list)


# top_acc_0_7256_with_gauss = best_model_all_version


#### Our Results
# best_model_all_version.summary()
# best_accuracy_all_version = ['0.6226415038108826', '0.4716981053352356', '0.49056604504585266', '0.5471698045730591', '0.4528301954269409', '0.43396225571632385']
# best_model = '<keras.engine.sequential.Sequential object at 0x000001AC2FDC6370>'
# best_model_summary = best_model_all_version.summary()
# preprocessing_selected_of_best_model = 'preprocessing_of_best_model'






#%%

def predict_image_class(img, model, preprocessing_of_best_model):
    
    img = cv2.resize(img, (224, 224))    
    
    if(preprocessing_of_best_model == 'GaussianBlur_by_kernel'):
        img = GaussianBlur_by_kernel(img, 15)

    if(preprocessing_of_best_model == 'Contrast_enhancement'):
        img = Contrast_enhancement(img)
        
    if(preprocessing_of_best_model == 'Apply_thresholding'):
        img = Apply_thresholding(img, 'Dark and Bright', 70, 200)   
        
    if(preprocessing_of_best_model == 'Canny'):
        img = Canny(img, 70, 200)    
    
    if(preprocessing_of_best_model == 'Canny'):
        img = Merge_preprocess_image(img, 7, 'Dark and Bright' ,70, 200, 70, 200)
  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.  # Normalize the image data
    
    model = best_model_all_version
    # Predict the class of the image
    predictions = model.predict(img_array)
    print(predictions)
    
    class_labels = ['Angry', 'Happy', 'Sad']  # Define your class labels here
    # Interpret the prediction results
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_class_index]
    return predicted_label

### Example
# path = 'Data/Angry/12.jpg'  #### Check funcation predict_image_class
# image = cv2.imread(path, 1)
# image = cv2.resize(image, (224, 224))
# predicted_label = predict_image_class(image, best_model_all_version, preprocessing_of_best_model)    
# print(predicted_label)
    
def detect_and_recognize_face_label_In_Image(path, model, preprocessing_of_best_model):   
    desired_width = 224*2
    desired_height = 224*2
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame = cv2.imread(path)
    frame_image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    frame = cv2.resize(frame, (desired_width, desired_height))
    frame_image_gray = cv2.resize(frame_image_gray, (desired_width, desired_height))
    
    faces_test = face_cascade.detectMultiScale(frame_image_gray, scaleFactor=1.1, minNeighbors=5)
    
    class_labels = ['Angry', 'Happy', 'Sad']  # Define your class labels here
    i = 0
    for (x, y, w, h) in faces_test:
         face = frame_image_gray[y:y+h, x:x+w]
         
         # Resize the ROI to match the size of the training images
         face = cv2.resize(face, (desired_width, desired_height))
         cv2.imwrite(f'Outputs/face{x}.png',face)
         face_img = cv2.imread(f'Outputs/face{x}.png')
        
         label_pred = predict_image_class(face_img, model, preprocessing_of_best_model)  
    
         cv2.putText(frame, label_pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 2)
    
        # Draw a rectangle around the detected face
         rec = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
         i = i+1
    
    cv2.imshow('Detected Faces', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

### Example
path = "C:\\Users\\peleg\\Desktop\\Image_Processing\\image_processing_course THE ARCHIVE WITH TUTORIALS AND ENV-20240106\\Project\\us.jpeg"
faces = detect_and_recognize_face_label_In_Image(path, model, "GaussianBlur_by_kernel")




