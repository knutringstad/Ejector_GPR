import numpy as np

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import pandas as pd 



# Select features to use for prediction
design_space = "geometry"

if design_space=="geometry":
    features = ["DmotiveOut", "Dmix", "Lmix", "alphadiff", "DdiffOut"]
    # Read data into dataframe
    df = pd.read_csv('Database_200pnt_reducedrange_proper.csv')
    
else:
    features = ["Pm", "Plift", "hm", "hs"]
    # Read data into dataframe
    df = pd.read_csv('Database_Performance_600pnt.csv')
    



# Drop crashed simulations 
df = df.drop(df.loc[df["CrashIndicator"]==1].index)
df = df.drop(df.loc[df["uni_vel"]==0].index)
df = df.drop(df.loc[abs(df["mfr_err"])>0.0001].index)

df_ejector = df.copy() 
df_ejector["Plift"] = df_ejector["Po"]- df_ejector["Ps"]
df_ejector["ER"] = df_ejector["mfr_s"] / df_ejector["mfr_m"]


# Select outputs to predict, choices =  ["eff", "ER", "mfr_m", "mfr_s", "uni_vel", "uni_alpha", "ds1"]
output = ["eff"]
y = df_ejector[output]
x = df_ejector[features]

# Split into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15,random_state=42)

# Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#Gaussian Process regression
#Load or retrain
lengthScale = np.ones(len(features))*0.5
std_estimate= 0.0000 #found by optimizer function
kernel = 0.2 * RBF(length_scale=lengthScale, length_scale_bounds=(1e-3, 3e1))  + WhiteKernel(noise_level=1e-2)

gp = GaussianProcessRegressor(kernel=kernel,alpha=std_estimate**2,n_restarts_optimizer=10).fit(x_train, y_train)  #run fitting
print(gp.kernel_) 

pred = gp.predict(x_test)

err = mean_squared_error(y_test,pred)
print("MSE : %f"%err)







#Displaying data

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plotting_variable_x=0
plotting_variable_y=2

#### Test data vs prediction plotted in scatterplot

ax.scatter(x_test[:,plotting_variable_x],x_test[:,plotting_variable_y], pred, c='b', s=50, zorder=10, edgecolors=(0, 0, 0))
ax.scatter(x_test[:,plotting_variable_x],x_test[:,plotting_variable_y], y_test, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
# ax.scatter(x_train[:,plotting_variable_x],x_train[:,plotting_variable_y], y_train, c='g', s=50, zorder=10, edgecolors=(0, 0, 0))

plt.show()

