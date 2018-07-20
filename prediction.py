import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


#Importing data from csv file into pandas dataframe

df = pd.read_csv('inspectdata.csv',usecols=[4,5,6,8,10,12])

# Taking log log of KM Done
df['Odometer'] = np.log(np.log(df['KM Done']))
df_variant = df['Variant'].str.get_dummies()
#df_state = df['State'].str.get_dummies()
df = pd.concat([df,df_variant],axis=1)
df = df.drop(['KM Done','Variant','State'],axis=1)

#Creating Correlation matrix 

matrix = df.corr()
f, ax = plt.subplots(figsize=(8, 6))
#sns.heatmap(matrix, vmax=0.7, square=True,annot=True,)
sns.pairplot(df[['Mfg. Year','O/S','Odometer','Final Price']])
plt.show()
# Ask Price function 

X = df[['Mfg. Year', 'O/S', 'Odometer','LX','LX BS II','LX MINOR','LXI','LXI BS','LXI BS II','LXI BS III','LXI BS IV','LXI CNG','LXI GREEN','LXI GREEN BS III','LXI MINOR','LXI MINOR GREEN','PRIMEA','VXI','VXI ABS BS III','VXI AT','VXI BS II','VXI BS III','VXI MINOR']]
#X = df[['Mfg. Year', 'O/S', 'Odometer Log Log','DL','UP','PB','HR','WB']]
y = df['Final Price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

X_normalizer = StandardScaler()
X_train = X_normalizer.fit_transform(X_train)
X_test = X_normalizer.transform(X_test)

y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)
print(X_train)
model = MLPRegressor(hidden_layer_sizes=(24,24), random_state=42,activation ='tanh',learning_rate_init = 0.01)
model.fit(X_train, y_train.ravel())

# Drawing Line of Best Fit through our cluster 

y_pred = model.predict(X_test)

y_pred_inv = y_normalizer.inverse_transform(y_pred)
y_test_inv = y_normalizer.inverse_transform(y_test)

# Build a plot
plt.scatter(y_pred_inv, y_test_inv)
plt.xlabel('Prediction')
plt.ylabel('Final Price')

diagonal = np.linspace(0, 300000, 100)
plt.plot(diagonal, diagonal, '-r')
plt.xlabel('Predicted Final Price')
plt.ylabel('Final Price')
plt.title('Line of Best fit')
plt.show()

def prediction(year,odometer,owner,lx,lxbii,lxmin,lxi,lxib,lxibii,lxibiii,lxibiv,lxic,lxig,lxigbiii,lximin,lximg,pri,vxi,vxiabiii,vxiat,vxibii,vxibiii,vximin):
    my_car = pd.DataFrame([
        {
            'Construction Year': year,
            'Odometer': odometer,
            'O/S': owner,
            'LX': lx,
            'LX BS II': lxbii,
            'LX MINOR': lxmin,
            'LXI': lxi,
            'LXI BS': lxib,
            'LXI BS II': lxibii,
            'LXI BS III': lxibiii,
            'LXI BS IV': lxibiv,
            'LXI CNG': lxic,
            'LXI GREEN': lxig,
            'LXI GREEN BS III': lxigbiii,
            'LXI MINOR': lximin,
            'LXI MINOR GREEN': lximg,
            'PRIMEA': pri,
            'VXI': vxi,
            'VXI ABS BS III': vxiabiii,
            'VXI AT': vxiat,
            'VXI BS II': vxibii,
            'VXI BS III': vxibiii,
            'VXI MINOR': vximin
        }
    ])

    my_car['Odometer Log Log'] = np.log(np.log(my_car['Odometer']))
    X_custom = my_car[['Construction Year','O/S', 'Odometer Log Log','LX','LX BS II','LX MINOR','LXI','LXI BS','LXI BS II','LXI BS III','LXI BS IV','LXI CNG','LXI GREEN','LXI GREEN BS III','LXI MINOR','LXI MINOR GREEN','PRIMEA','VXI','VXI ABS BS III','VXI AT','VXI BS II','VXI BS III','VXI MINOR']]
    X_custom = X_normalizer.transform(X_custom)
    y_pred = model.predict(X_custom)
    price_prediction = y_normalizer.inverse_transform(y_pred)
    print('Predicted ask price: %.2f' % price_prediction)

os.system('cls')
year = int(input('Please enter Mfg Year : '))
odometer = int(input('Please enter odometer reading : '))
owner = int(input('Please enter No of owners : '))
car_model = input('Please enter car model : ')
#state_code = input('Enter State code : ')

dl = 0 
hr = 0
wb = 0 
pb = 0 
up = 0

lx = 0
lxbii = 0
lxmin = 0
lxi = 0
lxib = 0
lxibii = 0
lxibiii = 0
lxibiv = 0
lxic = 0
lxig = 0
lxigbiii = 0
lximin = 0
lximg = 0
pri = 0
vxi = 0
vxiabiii = 0
vxiat = 0
vxibii = 0
vxibiii = 0
vximin = 0


if car_model =='LX':
    lx=1

elif car_model == 'LX BS II':
    lxbii = 1

elif car_model == 'LX MINOR':
    lxmin = 1

elif car_model == 'LXI':
    lxi = 1

elif car_model == 'LXI BS':
    lxib = 1

elif car_model == 'LXI BS II':
    lxibii = 1

elif car_model == 'LXI BS III':
    lxibiii = 1

elif car_model == 'LXI BS IV':
    lxibiv = 1 

elif car_model == 'LXI GREEN':
    lxig = 1

elif car_model == 'LXI CNG':
    lxic =1

elif car_model == 'LXI GREEN BS III':
    lxigbiii = 1

elif car_model =='LXI MINOR':
    lximin = 1

elif car_model == 'LXI MINOR GREEN':
    lximg = 1

elif car_model == 'PRIMEA':
    pri = 1

elif car_model == 'VXI':
    vxi = 1 

elif car_model == 'VXI ABS BS III':
    vxiabiii = 1

elif car_model == 'VXI AT':
    vxiat = 1

elif car_model == 'VXI BS II':
    vxibii = 1

elif car_model == 'VXI BS III':
    vxiabiii = 1

elif car_model == 'VXI MINOR':
    vximin = 1

'''
if state_code == 'dl':
    dl = 1
elif state_code == 'up':
    up = 1     
elif state_code == 'hr':
    hr = 1   
elif state_code == 'wb':
    wb = 1
elif state_code == 'pb`':
    pb = 1            
'''

prediction(year,odometer,owner,lx,lxbii,lxmin,lxi,lxib,lxibii,lxibiii,lxibiv,lxic,lxig,lxigbiii,lximin,lximg,pri,vxi,vxiabiii,vxiat,vxibii,vxibiii,vximin)
