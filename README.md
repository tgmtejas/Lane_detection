# Lane_detection
Lane Detection using End to End Deep Learning
Pre-processed data: </br>
https://drive.google.com/open?id=14hos3u3be7U5TJTWmayaOgwz89q9qVYH </br>
Code to load data: </br>

x_train = [] </br>
y_train = [] </br>
with h5py.File(os.path.join(os.getcwd(),'Lane_data2.h5'), 'r') as f: </br>
&nbsp; &nbsp;  x_train = f['RGB'][()] </br>
&nbsp; &nbsp;    y_train = f['Segmented'][()] </br>
#Convert both lists to Numpy</br>
    
x_train --> RGB images </br>
y_train --> Segmented images </br>
