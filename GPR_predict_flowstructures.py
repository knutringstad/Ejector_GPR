
from pickle import FALSE
from matplotlib.pyplot import scatter
from numpy.core.fromnumeric import var


def GPRflowstructure():
    from joblib import dump, load
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel ,RationalQuadratic,ExpSineSquared
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    from matplotlib import cm
    
    
    
    folder="PathDataAnalysed"
    variable="velocity"
    nozzle="suction"
    particle_list=[2]
    for particle in particle_list:


        filename="%s/%s_%s_particle%d_pathlines.csv" %(folder,variable,nozzle,particle)
        df = pd.read_csv(filename,index_col=0)

        features=['Dmix','Lmix','X']
        output = variable


        df_2=df
        
        predictionTest=False
        if predictionTest:
            df=df.loc[ (df["Dmix"]!=0.004) | (df["Lmix"]!=0.03) ]




        trainBool=True
        if trainBool == True:
            sample_size =0.1
            seed = 2

            x_train, x_test, y_train, y_test = train_test_split(df[features],df[output], test_size=sample_size,random_state=seed)

            x_train = np.array(x_train)
            x_test = np.array(x_test)

            # Scaling
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            lengthScale = np.ones(len(features))*0.5
            std_estimate= 0.000

            kernel = ConstantKernel(1e+2) * RBF(length_scale=lengthScale)  + WhiteKernel(noise_level=1e-2) 

            Normalize_bool=True

            gp = GaussianProcessRegressor(kernel=kernel,alpha=std_estimate**2,n_restarts_optimizer=2,normalize_y=Normalize_bool).fit(x_train, y_train)  #run fitting

            gp_dump = dump(gp,'gp_flow.joblib')
            sc_dump = dump(sc,'sc_flow.joblib')
            x_test_dump = dump(x_test,'x_test_flow.joblib')
            x_train_dump = dump(x_train,'x_train_flow.joblib')
            y_test_dump = dump(y_test,'y_test_flow.joblib')
            y_train_dump = dump(y_train,'y_train_flow.joblib')
            print("Model trained")
        else:
            print("Loading pretrained gaussian process")
            gp= load('gp_flow.joblib') 
            sc = load('sc_flow.joblib')
            x_test = load('x_test_flow.joblib')
            x_train = load('x_train_flow.joblib')
            y_test = load('y_test_flow.joblib')
            y_train = load('y_train_flow.joblib')




        kern = gp.kernel_
        print(kern)
        pred = gp.predict(x_test)
        err=mean_absolute_error(y_test,pred)
        err_square=mean_squared_error(y_test,pred)

        print(err)

        plottype="singlevar"
        if plottype =="surf":

            var1=0
            var2=2



            Ndim = len(features)        

            N_points=40
            scaled_minmax=1.6
            x_vec = np.linspace(-scaled_minmax, scaled_minmax, N_points)
            y_vec = np.linspace(-scaled_minmax, scaled_minmax, N_points)

            mat = np.meshgrid(x_vec, y_vec)
            X_,Y_ = np.meshgrid(x_vec, y_vec)

            numPoints = len(mat[0].flatten('C'))
            testPoints=np.zeros((Ndim,numPoints))


            
            if var1==0:
                freevarval=0.04 
                freevar= np.ones(numPoints)*freevarval  #SET THE FREE VAR HERE
            else:
                freevarval=0.003
                freevar= np.ones(numPoints)*freevarval

            vecunscaled = np.array([freevar,freevar,freevar])
            vecscaled=sc.transform(vecunscaled.T).T

            for j in range((Ndim)):
                if j == var1:
                    testPoints[j]=mat[0].flatten('C')
                elif j==var2:
                    testPoints[j]=mat[1].flatten('C')
                else:
                    testPoints[j] = vecscaled[j]






            scaled=np.array([x_vec,x_vec,x_vec])
            unscaled=sc.inverse_transform(scaled.T)


            X_,Y_ = np.meshgrid(unscaled[:,var1], unscaled[:,var2])



            pred_grid, std_grid = gp.predict(testPoints.T, return_std=True)

            pred_grid= np.reshape(pred_grid, (N_points,N_points),order='C')
            std_grid= np.reshape(std_grid, (N_points,N_points),order='C')


            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # ax.set_zscale('log')

            surf = ax.plot_surface(X_,Y_, pred_grid[:,:],alpha=0.6)

            plt.tight_layout()

            # surf = ax.plot_surface(X_,Y_, pred_grid[:,:] )

            
            if (not var1==1 and not var2==1):
                scatter_df=df_2.loc[ (df_2["Lmix"]== freevarval) ]
            else:
                scatter_df=df_2.loc[ (df_2["Dmix"]== freevarval) ]

            plotline3d=True
            if plotline3d:
                first= [0,38,38+43,38+45+45+1,38+45+45+41+13]
                last = [38,38+43,38+45+45+1,38+45+45+41+13,38+45+45+45+45+22]
                for i in range(5):
                    ax.plot3D(scatter_df[features[var1]][first[i]:last[i]], scatter_df[features[var2]][first[i]:last[i]], scatter_df[variable][first[i]:last[i]], 'red')

            ax.scatter(scatter_df[features[var1]],scatter_df[features[var2]], scatter_df[variable], c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
           

            ax.set_ylabel('Pathline length - c [m]')
            ax.set_xlabel(r'Mixing diameter - $D_{{mix}}$ [m]')
            ax.set_zlabel(r'Velocity magnitude - $\|\vec{u}\|$ [m/s]')
            ax.set_zlim(0, 80)
            ax.set_ylim(0.0, 0.13)
            
        elif plottype=="singlevar":

            Ndim = len(features)        

            N_points=40
            x_vec = np.linspace(0, 0.13, N_points)

            dmix=0.004
            lmix=0.03
            # mix_list=[0.003,0.0035,0.004,0.0045,0.005]
            mix_list=[0.01,0.02,0.03,0.04,0.05]
            # mix_list=[dmix]
            for lmix in mix_list:
                

                dmixvec= np.ones(N_points)*dmix 
                lmixvec= np.ones(N_points)*lmix

                vecunscaled = np.array([dmixvec,lmixvec,x_vec])
                vecscaled=sc.transform(vecunscaled.T).T

                testPoints=np.zeros((Ndim,N_points))

                testPoints[0] = vecscaled[0]
                testPoints[1] = vecscaled[1]
                testPoints[2] = vecscaled[2]

                pred, std= gp.predict(testPoints.T, return_std=True)
                
                scatter_df=df_2.loc[ (df_2["Dmix"]== dmix) ]
                scatter_df=scatter_df.loc[ (scatter_df["Lmix"]== lmix) ]


                if variable=="pressure": #scale to bar
                    y_scatter=np.divide(scatter_df[variable],1e+5)
                    pred=np.divide(pred,1e+5)
                    std=np.divide(std,1e+5)
                    ymin = 29
                    ypos_label=ymin+0.3
                    y_ax=ymin
                elif variable=="velocity": #scale
                    y_scatter=np.divide(scatter_df[variable],1)
                    pred=np.divide(pred,1)
                    std=np.divide(std,1)
                    ypos_label=3
                    y_ax=0

                plt.plot(x_vec,pred)
                # plt.fill_between(x_vec, pred - std, pred + std, color='darkblue', alpha=0.2)
                plt.fill_between(x_vec, pred - std, pred + std,  alpha=0.2)


                # plt.scatter(scatter_df["X"],y_scatter)



                if variable=="pressure":
                    plt.ylabel("Pressure [bar]")
                    plt.ylim(ymin,39)
                elif variable=="velocity":
                    plt.ylabel(r"Velocity magnitude - $\| \vec{u} \|$ [m/s]")
                    plt.ylim(0,100)

                
                plt.annotate('', xy=(0.0, y_ax), xytext=(0.015, y_ax), arrowprops=dict(arrowstyle='<->',facecolor='black'), annotation_clip=False            )
                plt.annotate('Suction nozzle', xy=(0.0, ypos_label), xytext=(-0.002, ypos_label), annotation_clip=False            )
                plt.annotate('', xy=(0.02, y_ax), xytext=(0.06, y_ax), arrowprops=dict(arrowstyle='<->',facecolor='black'), annotation_clip=False            )
                plt.annotate('Mixing chamber', xy=(0.026, ypos_label), xytext=(0.03, ypos_label), annotation_clip=False            )
                plt.annotate('', xy=(0.06, y_ax), xytext=(0.125, y_ax), arrowprops=dict(arrowstyle='<->',facecolor='black'), annotation_clip=False            )
                plt.annotate('Diffuser', xy=(0.075, ypos_label), xytext=(0.073, ypos_label), annotation_clip=False            )
                
                plt.tight_layout

                plt.xlabel("Pathline distance - c [m]")

                
    # legends= [r'$D_{{mix}}$=3 [mm], $\eta=0.21$ [-]',r'$D_{{mix}}$=3.5 [mm], $\eta=0.32$ [-]',r'$D_{{mix}}$=4 [mm], $\eta=0.34$ [-]',r'$D_{{mix}}$=4.5 [mm], $\eta=0.25$ [-]',r'$D_{{mix}}$=5 [mm], $\eta=0.05$ [-]']
    legends= [r'$L_{{mix}}$=10 [mm], $\eta=0.40$ [-]',r'$L_{{mix}}$=20 [mm], $\eta=0.42$ [-]',r'$L_{{mix}}$=30 [mm], $\eta=0.37$ [-]',r'$L_{{mix}}$=40 [mm], $\eta=0.34$ [-]',r'$L_{{mix}}$=50 [mm], $\eta=0.30$ [-]']
    plt.legend(legends)
    plt.show()

GPRflowstructure()