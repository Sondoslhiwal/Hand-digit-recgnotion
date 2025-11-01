from tensorflow.keras.models import sequential
from tensorflow.keras.dataset import minst
from tensorflow.keras.layers import Dense,Dropout,Flatten 
from sklearn.model_selection import train_test_split

(x_train,y_train),(x_test,y_test)=minst.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
