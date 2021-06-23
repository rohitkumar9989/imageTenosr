import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import xlsxwriter

class Image_Tensor ():
    def __init__(self, directory, data_frame_dire_save):
        """
        Image Tensor is an methos which does not use the tensorflow's 
        convolutional layers,

        This model uses the csv formats for the saving of the models
        Please make sure that the model is xlsx file should be assigned in your 
        directory


        Args:
            directory ([str]): The directory ewhere the models images are saved 
                                and the images where it has to be decoded
                                **Note**: If another directory exists inside the model `self.directory`
                                then this may impose an error
            data_frame_dire_save ([str]): Your xlsx directory where the models data is saved
        """


        self.dataframe=data_frame_dire_save
        self.dir=directory
        self.get_images()
        #self.image_prediction()
    def convert_image_tensor (self, image):
        image=tf.io.read_file(image)
        img=tf.image.decode_image(image)
        image=tf.image.resize(img, size=(70, 70))
        return image/255.

    def get_images(self):
        """
        This method is usefull for hetting the images and passing to the 
        `self.convert_image_tensor` method for the conversion i=of the methos into tensors
        """
        self.mainlist=[]
        try:
            for self.dirpaths, self.dirnames, self.filenames in os.walk(self.dir):
                self.class_names=len(self.dirnames)
                for i in range (int(class_names)):
                    try:
                        for dirpaths, dirnames, filenames in os.walk(dirnames[i]):
                            file_names=int(len(filenames))
                            for m in range (file_names):
                                image=self.convert_image_tensor(filenames[m])
                                arrayed=np.array(image)
                                list=arrayed.tolist()
                                image_list=[]

                                for i in range (70):
                                    for m in range (70):
                                        for k in range (3):
                                            image_list.append(list[i][m][k])
                                self.mainlist.append(image_list.append(self.dirnames))
                                list.clear()
                    except Exception as e:
                        print ("The file doesnt have any images") 
        except Exception as e:
            print ('Another direcory in the main directory') 

        all_results = pd.DataFrame(self.mainlist)

        writer = pd.ExcelWriter('demo.xlsx', engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        all_results.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file.
        
        writer.save()  
        print ("A file already exists. Please Delete the file")        
                         
                 
    def open_excel_sheet (self, path):
        self.path=path
        data=pd.read_excel(self.dataframe, engine='openpyxl')
        x=data.iloc[:,:-1].values
        y=data.iloc[:,-1].values
        X_train,X_test,Y_train, Y_test=train_test_split(x,
                                                        y)
        return X_train, X_test, Y_train, Y_test


    def image_prediction(self, show_loss_curve=False, learning_rate=False):
        """AI is creating summary for image_prediction

        Args:
            show_loss_curve (bool, optional): [shows the performance of the model]. 
                                                Defaults to False.
            learning_rate (bool, optional): [This method is usefull for the looking at the losscurves and
                                                and manually setting it into the code for more presicion for the model]. 
                                                Defaults to False.

        Returns:
            `model`: Returns the model which has been trained
        """
        def make_model ():
            model=tf.keras.models.Sequential([
                tf.keras.layers.Dense(units=100,
                                        activation='relu'),
                tf.keras.layers.Dense(units=100, 
                                        activation='relu'),
                tf.keras.layers.Dense(90, activation='relu'),
                tf.keras.layers.Dense(90, activation='relu'),
                tf.keras.layers.Dense(50, activation='relu'),
                tf.keras.layers.Dense(50, activation='relu'),
                tf.keras.layers.Dense(50, activation='relu'),
                tf.keras.layers.Dense(20, activation='relu'),
                tf.keras.layers.Dense(20, activation='relu'),
                tf.keras.layers.Dense(20, activation='relu'),
                tf.keras.layers.Dense(len(os.listdir(self.dir)), activation='softmax')
            ])
            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                            optimizer=tf.keras.optimizers.Adam(),
                            metrics='accuracy')
            X_train, X_test, Y_train, y_test=self.open_excel_sheet(path=self.dataframe)
            model.fit(X_train, 
                        Y_train, 
                        epochs=100, 
                        steps_per_epoch=len(X_train),
                        callback=tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4*10**(epochs/200))
                        )
            return model, epochs

        if show_loss_curve==True:
            if learning_rate==False:
                model_main, epochs=make_model()
                history=model_main
                pd.DataFrame(history.history).plot()


        if learning_rate==True:
            if show_loss_curve==False:
                model_main, epochs=make_model()
                history=model_main
                lrs=1e-4*10**(np.arange(0, epochs)/200)
                plt.semilogx(lrs, history.history['loss'])


        if learning_rate==False:
            if show_loss_curve==False:
                model_main=make_model()

        #Predict the main model results with the test


        while True:
            model_main.evaluate(X_test, Y_test)
            numps=model_main.numpy()
            if numps[1]>=80:
                return model_main
                break
            else:
                continue

        
    
Image_Tensor(directory='C:\\Users\\rohit\\Downloads\\Data_for_blind',data_frame_dire_save='zero.xlsx')
