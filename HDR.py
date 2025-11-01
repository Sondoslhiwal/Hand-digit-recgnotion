from tensorflow.keras.models import sequential
from tensorflow.keras.dataset import minst
from tensorflow.keras.layers import Dense,Dropout,Flatten 
from sklearn.model_selection import train_test_split

(x_train,y_train),(x_test,y_test)=minst.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
Brainstorm_Model=Sequential([
   Flatten(input_shape=(28,28)),
   Dense(128,activation='relu'),
   Dropout(0.2),
   Dense(10,activation='softmax')])


Brainstorm_Model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
X =Brainstorm_Model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=20)

print(Brainstorm_Model.evaluate(x_test,y_test))
