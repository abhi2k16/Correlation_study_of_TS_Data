import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import spearmanr #for nonlinear corelation 
#plt.style.use('seaborn')
#%% import data
# Parameter to be analys:N2_rate, Depth, IPRS_RAW,WOB_DH,TOB_DH,CT_WGT,CIRC_PRS,WH_PRS,BVEL
# FLWI, GTF_RAW, VIB_LAT, SHK_LAT,TEMP_DNI_RAW, ATEMP_RAW, PTEMP_RAW, DAGR_Temp
parameters = ['N2_RATE','APRS_RAW','IPRS_RAW','WOB_DH','CT_WGT',
              'CIRC_PRS','WH_PRS','BVEL','FLWI','GTF_RT_RAW','VIB_LAT','SHK_LAT','HDTH',
              'TEMP_DNI_RAW', 'ATEMP_RAW', 'PTEMP_RAW', 'DAGR_Temp','DEPT','INCL_RT_RAW',
              'AZIM_RT_RAW'] 
#data = pd.read_csv("Pandas_dataframe_O_1011724_56-7.csv")
data = pd.read_csv("C:/Users/AKumar340/OneDrive - SLB/2024/CTD_EventDetection/Data/O.1011724.56-5.csv")
N2data = data[['TIME','N2_RATE']]
APRS_RAWdata = data[['TIME','APRS_RAW']]
#%% Plot the observed data 
start_time = 0 #In second
end_time = 140000 #In second
fig, axs = plt.subplots(4,4, figsize = (14,10))

axs[0,0].plot(data.APRS_RAW[start_time:end_time])
axs[0,0].set_title('APRS_RAW')

axs[0,1].plot(data.IPRS_RAW[start_time:end_time])
axs[0,1].set_title('IPRS_RAW')

axs[0,2].plot(data.WOB_DH[start_time:end_time])
axs[0,2].set_title('WOB_DH')

axs[0,3].plot(data.AZIM_RT_RAW[start_time:end_time])
axs[0,3].set_title('AZIM_RT_RAW')

axs[1,0].plot(data.CT_WGT[start_time:end_time])
axs[1,0].set_title('CT_WGT')

axs[1,1].plot(data.CIRC_PRS[start_time:end_time])
axs[1,1].set_title('CIRC_PRS')

axs[1,2].plot(data.WH_PRS[start_time:end_time])
axs[1,2].set_title('WH_PRS')

axs[1,3].plot(data.BVEL[start_time:end_time])
axs[1,3].set_title('BVEL')

axs[2,0].plot(data.FLWI[start_time:end_time])
axs[2,0].set_title('FLWI')

axs[2,1].plot(data.GTF_RT_RAW[start_time:end_time])
axs[2,1].set_title('GTF_RT_RAW')

axs[2,2].plot(data.VIB_LAT[start_time:end_time])
axs[2,2].set_title('VIB_LAT')

axs[2,3].plot(data['SHK_LAT'][start_time:end_time])
axs[2,3].set_title('SHK_LAT')

axs[3,0].plot(data['TEMP_DNI_RAW'][start_time:end_time])
axs[3,0].set_title('TEMP_DNI_RAW')

axs[3,1].plot(data['ATEMP_RAW'][start_time:end_time])
axs[3,1].set_title('ATEMP_RAW')

axs[3,2].plot(data['PTEMP_RAW'][start_time:end_time])
axs[3,2].set_title('PTEMP_RAW')

axs[3,3].plot(data['DAGR_Temp'][start_time:end_time])
axs[3,3].set_title('DAGR_Temp')
axs[3,2].set(xlabel = 'Time (sec)')
plt.subplots_adjust(left=None, bottom=None, right=None,
        top=None, wspace=0.3, hspace=0.3)
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#      ax.label_outer()
#plt.savefig('parameter_of_interest.tif')
plt.show()
#%% Data cleaning: fill NaN with adjecent value
for parameter in parameters:
    if data[parameter].isnull().values.any() == True:
        print(data[data[parameter].isnull()].index)
        print(len(data[data[parameter].isnull()]),parameter)
#%%
for column in data.columns:
    if column in parameters:
        if data[column].isnull().values.any() == True:
            print(column)
            data[column]=data[column].bfill()
#%% PTEMP_RAW unwanted data removal
for k in data[data['PTEMP_RAW'] < 0].index:
    j = 0
    while data['PTEMP_RAW'][k] < 0:
        k = k+1
        j = j+1
        if data['PTEMP_RAW'][k] > 0: 
            data.loc[k-j,'PTEMP_RAW'] = data.loc[k,'PTEMP_RAW']
#%% IPRS_RAW unwanted data removal
for k in data[data['IPRS_RAW'] < 0].index:
    j = 0
    while data['IPRS_RAW'][k] < 0:
        k = k+1
        j = j+1
        if data['IPRS_RAW'][k] > 0: 
            data.loc[k-j,'IPRS_RAW'] = data.loc[k,'IPRS_RAW']
#%% Measuring correlation by calculating Pearson Coeficent
class correlation:
    def __init__(self, parameters):
        self.parameters = parameters
    def normalize_data(self,x):
        self.x = x
        normData = (x-np.min(x))/(np.max(x)-np.min(x))
        return normData
    def scatter_plot(self, n_step, time_step, fig_size, sub_plot_row, sub_plot_col):
        # n_step: # of step
        # time_step : Time in second
        # sub_plot_row : plt.subplots(sub_plot_row,Sub_plot_col, fig_size = (_,_)) 
        # Sub_plot_col :
        time = np.arange(time_step/60,(time_step/60)*n_step+1,time_step/60)
        for i, parameter in enumerate(self.parameters):
            if parameter == 'N2_RATE':
                pass
            else:
                plt.figure(figsize=fig_size)
                for j in range(n_step):
                    x = data.N2_RATE[j*time_step:(j+1)*time_step].values
                    y = data[parameter][j*time_step:(j+1)*time_step].values
                    norm_x = self.normalize_data(x)
                    norm_y = self.normalize_data(y)
                    plt.subplot(sub_plot_row,sub_plot_col,j+1)
                    plt.scatter(norm_x,norm_y,marker=".")
                    plt.title(str(int(time[j]))+' minute')
                plt.suptitle(parameter,fontsize=16)
                plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.3, hspace=0.3)
                plt.show()
    def corr_plot(self, n_step,time_step, fig_size):
        plt.figure(figsize=(12,9))
        for j,parameter in enumerate(self.parameters):
            if parameter == 'N2_RATE':
                pass
            else:
                corr_coefficients = []
                for i in range(n_step):
                    x = N2data.N2_RATE[i*time_step:(i+1)*time_step].values
                    y = data[parameter][i*time_step:(i+1)*time_step].values
                    correlation_coefficient, p_value = stats.pearsonr(x, y)
                    corr_coefficients.append(correlation_coefficient)
                plt.subplot(4,3,j)
                plt.plot(np.arange(time_step/60,(time_step/60)*n_step+1,time_step/60)
                         ,np.array(corr_coefficients),'-')
                plt.title(parameter)
                plt.ylabel('Corr. Coefficient')
        plt.xlabel('time (minute)')
        plt.subplots_adjust(left=None, bottom=None, right=None,
                top=None, wspace=0.3, hspace=0.3)
        #plt.savefig('pearson_corr_5hrs_step_20min.tif')
        plt.show()
    def corr_param(self,max_steps, time_steps,fig_size, sub_plot_row, sub_plot_col):
        # Max steps is time in minutes
        for i, parameter in enumerate(self.parameters):
            if parameter == 'N2_RATE':
                pass
            else:
                plt.figure(figsize = fig_size)
                for k,time_step in enumerate(time_steps):                
                    corr_coefficients = []
                    p_values = []
                    n_step = int(max_steps/time_step)
                    for j in range(n_step):
                        x = data.N2_RATE[j*time_step*60:(j+1)*time_step*60].values
                        y = data[parameter][j*time_step*60:(j+1)*time_step*60].values
                        correlation_coefficient, p_value = stats.kendalltau(x, y)
                        corr_coefficients.append(correlation_coefficient)   
                        p_values.append(p_value)
                    plt.subplot(sub_plot_row,sub_plot_col,k+1)
                    plt.plot(np.arange(time_step,(time_step)*n_step+1,time_step)
                             ,np.array(corr_coefficients),'-')
                    plt.ylabel('Corr. Coefficient')
                    plt.xlabel('time (minute)')
                    plt.title(str(time_step)+' MINUTES')
                plt.suptitle(parameter,fontsize = 16)
                plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.3, hspace=0.3)
            plt.show()
    # Plotting Corr for different parameters with different time step  
    def corr_slid(self,start_time, end_time, time_steps,fig_size, sub_plot_row, sub_plot_col):
        #time_steps = [_,_,_] values in minutes
        # end_time in second
        for i, parameter in enumerate(self.parameters):
            if parameter == 'N2_RATE':
                pass
            else:
                fig, axs = plt.subplots(sub_plot_row, sub_plot_col, figsize = fig_size)
                for k,time_step in enumerate(time_steps):   
                    # Convert time_step from minutes to second
                    time_step_sec = time_step*60
                    x_N2Rate = data.N2_RATE[start_time:end_time].values
                    corr_coefficients = []
                    p_values = []
                    #n_step = int(max_steps/time_step)
                    for j in range(int(len( x_N2Rate[0:end_time-time_step_sec])/60)):
                        x = data.N2_RATE[start_time +j*60:start_time +(j+1)*60 + time_step_sec-60].values
                        y = data[parameter][start_time+j*60:start_time +(j+1)*60 + time_step_sec-60].values
                        correlation_coefficient, p_value = stats.spearmanr(x, y)
                        corr_coefficients.append(correlation_coefficient)   
                        p_values.append(p_value)
                    axs[k+1].plot(
                        np.arange(start_time/60,end_time/60,((end_time-start_time)/60)/len(corr_coefficients)), 
                             corr_coefficients)
                    axs[k+1].set_ylabel('Corr. Coefficient')
                    z = data[parameter][start_time:end_time]
                    axs[k].plot(np.arange(start_time,end_time,1)/60,z,label = parameter)
                    axs_twin = axs[k].twinx()
                    axs_twin.plot(np.arange(start_time,end_time,1)/60,x_N2Rate, label='N2_Rate',color='darkorange')
                    axs_twin.set_ylabel('N2 rate',color='darkorange')
                    axs_twin.legend()
                    axs_twin.tick_params(axis='y',colors ='darkorange')
                    axs_twin.spines['right'].set_color('darkorange')
                    axs[k].set_ylabel(parameter)
                    axs[k+1].set_xlabel('time (minute)')
                    axs[k].legend()
                fig.suptitle(parameter,fontsize = 16)
            plt.show()
    def corr_slid_timeLag(self, end_time, time_steps,fig_size, sub_plot_row, sub_plot_col,TimeLags):
        #time_steps = [_,_,_] values in minutes
        # timeLag in minutes
        for i, parameter in enumerate(self.parameters):
            if parameter == 'N2_RATE':
                pass
            else:
                plt.figure(figsize = fig_size)
                for k,time_step in enumerate(time_steps):   
                    # Convert time_step from minutes to second
                    corr_coefficients1 = []
                    corr_coefficients2 = []
                    corr_coefficients3 = []
                    corr_coefficients4 = []
                    corr_coefficients5 = []
                    for l,timeLag in enumerate(TimeLags):
                        time_step_sec = time_step*60
                        x_N2Rate = data.N2_RATE[0:end_time].values
                        #corr_coefficients = []
                        #p_values = []
                        #n_step = int(max_steps/time_step)
                        for j in range(int(len( x_N2Rate[0:end_time-time_step_sec])/60)):
                            x = data.N2_RATE[j*60 :(j+1)*60 + time_step_sec-60].values
                            y = data[parameter][j*60 + timeLag*60 :(j+1)*60 + time_step_sec-60+timeLag*60].values
                            correlation_coefficient, p_value = stats.pearsonr(x, y)
                            corr = 'corr_coefficients'+str(l+1)
                            corr = eval(corr)
                            corr.append(correlation_coefficient)   
                            #p_values.append(p_value)
                    plt.subplot(sub_plot_row,sub_plot_col,k+1)
                    plt.plot(corr_coefficients1,linewidth=1.5)
                    plt.plot(corr_coefficients2,linewidth=1.2)
                    plt.plot(corr_coefficients3,linewidth=1.5)
                    plt.plot(corr_coefficients4,linewidth=1.5)
                    plt.plot(corr_coefficients5,linewidth=1.5)
                    legends = map(str, TimeLags)
                    plt.legend(list(legends),loc='lower left')
                    plt.ylabel('Corr. Coefficient')
                    plt.xlabel('time (minute)')
                    plt.title(str(time_step)+' MINUTES')
                #plt.suptitle(parameter+' '+str(timeLag)+' Minute Time Lag',fontsize = 16)
                plt.suptitle(parameter, fontsize = 16)
                plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=None, wspace=0.3, hspace=0.3)
            plt.show()
#%%
parameters = ['N2_RATE','VIB_LAT','SHK_LAT','CIRC_PRS','IPRS_RAW','APRS_RAW'] 
corr = correlation(parameters)
TimeLags = [2,4,6,8,10]
start_time = 300*60
end_time =600*60
#corr.corr_param(max_steps=1200, time_steps=[5,10,15,20], fig_size =(12,8), sub_plot_row=2, sub_plot_col=2)
corr.corr_slid(start_time=start_time, end_time=end_time, time_steps=[2], fig_size =(8,4), sub_plot_row=2, sub_plot_col=1)
#end_time =36000
#corr.corr_slid_timeLag(end_time =135000, time_steps=[5,10,15,20], fig_size =(12,8), 
#                      sub_plot_row=2, sub_plot_col=2,TimeLags=TimeLags)
#%%corr.corr_plot(n_step =30, time_step=300, fig_size=(12,9))
#%% Time Series Data Plotting of Parameter With N2 RATE 
corr = correlation(parameters)
end_time = 2250*60
start_time = 0
#plt.figure(figsize = (8,6))
for i, parameter in enumerate(corr.parameters):
    if parameter == 'N2_RATE':
        pass
    elif parameter == 'AZIM_RT_RAW':
    #else:
        parameter1 = 'SHK_LAT'
        x = data[parameter1][start_time:end_time].values
        y = data[parameter][start_time:end_time].values 
        fig, ax1 = plt.subplots(4,1,figsize = (8,8))
        ax1[0].plot(np.arange(0,len(y),1)/60, x)
        ax2 = ax1[0].twinx()
        ax2.plot(np.arange(0,len(y),1)/60,y,'-',color='darkorange')
        ax1[0].set_ylabel(parameter1)
        ax2.set_ylabel(parameter)
        #ax1[0].set_xlabel('Time (Minutes)')
        #ax1[0].set_xlim(0,2400)
        ax1[0].legend([parameter1])
        ax2.legend([parameter])
        ax3 = ax1[1].twinx()
        ax1[1].plot(np.arange(0,len(y),1)/60, x)
        ax3.plot(np.arange(0,len(y),1)/60,y,'-',color='darkorange')
        ax1[1].set_ylabel(parameter1)
        ax3.set_ylabel(parameter)
        #ax1[1].set_xlabel('Time (Minutes)')
        ax1[1].set_xlim(0,400)
        ax4 = ax1[2].twinx()
        ax1[2].plot(np.arange(0,len(y),1)/60, x)
        ax4.plot(np.arange(0,len(y),1)/60,y,'-',color='darkorange')
        ax1[2].set_ylabel(parameter1)
        ax3.set_ylabel(parameter)
        ax1[2].set_xlabel('Time (Minutes)')
        ax4.set_ylabel(parameter)
        ax1[2].set_xlim(400,1200)
        ax5 = ax1[3].twinx()
        ax1[3].plot(np.arange(0,len(y),1)/60, x)
        ax5.plot(np.arange(0,len(y),1)/60,y,'-',color='darkorange')
        ax1[3].set_ylabel(parameter1)
        ax3.set_ylabel(parameter)
        ax1[3].set_xlabel('Time (Minutes)')
        ax5.set_ylabel(parameter)
        ax1[3].set_xlim(1200,1600)
        fig.suptitle(parameter1+' with '+ parameter)
    plt.show()
#%% Multi axis plot 
data = pd.read_csv("C:/Users/AKumar340/OneDrive - SLB/2024/CTD_EventDetection/Data/O.1011724.56-7.csv")
parameters_2 = ['AZIM_RT_RAW','INCL_RT_RAW','WH_PRS','DEPT','CT_WGT','BVEL','GTF_RT_RAW']
fig, ax1 = plt.subplots(figsize = (20,10))

y = np.arange(0,len(data['SHK_LAT']),1)/60

ax1.plot(y,data[parameters_2[0]], color = 'blue',alpha=0.9)
ax1.set_ylabel(parameters_2[0])
ax1.set_xlabel('Time(Minutes)')
ax2 = ax1.twinx()
ax2.plot(y,data['SHK_LAT'],color = 'green',alpha=0.9)
ax2.set_ylabel('SHK_LAT', color = 'green' )
ax3 = ax1.twinx()
ax3.plot(y,data[parameters_2[1]], color = 'red',alpha=0.5)
ax3.set_ylabel(parameters_2[1],color = 'red')
ax3.spines['right'].set_position(('outward',40))
ax4 = ax1.twinx()
ax4.plot(y,data[parameters_2[2]],color = 'darkorange')
ax4.set_ylabel(parameters_2[2],color = 'darkorange')
ax4.spines['right'].set_position(('outward',80))
ax5 = ax1.twinx()
ax5.plot(y,data[parameters_2[3]],color = 'deeppink',alpha=1,linewidth = 3)
ax5.set_ylabel(parameters_2[3], color = 'deeppink')
ax5.spines['right'].set_position(('outward',120))
ax6 = ax1.twinx()
ax6.plot(y,data[parameters_2[4]],color = 'brown',alpha=0.5)
ax6.set_ylabel(parameters_2[4], color = 'brown')
ax6.spines['right'].set_position(('outward',160))
ax7 = ax1.twinx()
ax7.plot(y,data[parameters_2[5]],color = 'purple')
ax7.set_ylabel(parameters_2[5], color = 'purple')
ax7.spines['right'].set_position(('outward',220))
ax8 = ax1.twinx()
ax8.plot(y,data[parameters_2[6]],color = 'black',alpha=0.5)
ax8.set_ylabel(parameters_2[6], color = 'black')
ax8.spines['right'].set_position(('outward',260))
ax1.tick_params(axis = 'y', colors = 'blue')
ax2.tick_params(axis = 'y', colors = 'green')
ax3.tick_params(axis = 'y', colors = 'red')
ax4.tick_params(axis = 'y', colors = 'darkorange')
ax5.tick_params(axis = 'y', colors = 'deeppink')
ax6.tick_params(axis = 'y', colors = 'brown')
ax7.tick_params(axis = 'y', colors = 'purple')
ax8.tick_params(axis = 'y', colors = 'black')

ax2.spines['right'].set_color('green')
ax3.spines['right'].set_color('red')
ax4.spines['right'].set_color('darkorange')
ax5.spines['right'].set_color('deeppink')
ax6.spines['right'].set_color('brown')
ax7.spines['right'].set_color('purple')
ax8.spines['right'].set_color('black')

plt.show()
#%% Noise Quantification
from scipy import signal
from scipy.spatial.distance import jensenshannon, euclidean
def NoiseQuantification(x,time_step, end_time):
    x_noise = x
    b, a = signal.butter(3, 0.05)
    #x_noisless = signal.filtfilt(b=b, a=a, x = x)
    y = data.N2_RATE[0:end_time].values # N2 Rate data
    y_norm = corr.normalize_data(y)
    x_noisless = signal.savgol_filter(x_noise, window_length = 200, polyorder = 3)
    plt.figure(figsize= (8,6))
    plt.subplot(311)
    plt.plot(np.arange(0,len(x_noise),1)/60,corr.normalize_data(x_noise))
    plt.plot(np.arange(0,len(x_noise),1)/60,corr.normalize_data(x_noisless))
    plt.plot(np.arange(0,len(y_norm),1)/60,y_norm)
    plt.ylabel('Normalized Scale')
    plt.legend(['Noisy data','Filtered data','N2 Rate'])
    #n_step = int(len(x_noise[0:end_time-time_step])/time_step)
    p_correlations = []
    d_jensenshannons = []
    for i in range(int(len(x_noise[0:end_time-time_step])/60)):
        x_noise_seg = x_noise[i*60  :(i+1)*60 + time_step]
        x_noisless_seg = x_noisless[i*60 :(i+1)*60 + time_step]
        p_corr = signal.correlate(x_noise_seg, x_noisless_seg)
        p_correlations.append(p_corr)
        jensenshannon_d = jensenshannon(x_noise_seg, x_noisless_seg)
        d_jensenshannons.append(jensenshannon_d)
    plt.subplot(312)
    #plt.plot(np.arange(0,len(p_correlations)*5,5),np.array(p_correlations))
    plt.plot(np.arange(5,len(p_correlations)+5,1),np.array(p_correlations))
    plt.ylabel('Corr Coefficient')
    plt.subplot(313)
    plt.plot(np.arange(5,len(p_correlations)+5,1),np.array(d_jensenshannons))
    plt.ylabel('JS Divergence')
    plt.xlabel('Time (Minute)')
    plt.suptitle('VIB_LAT')
    plt.show()
end_time= 6000
time_step = 10
x = data.VIB_LAT[0:end_time].values
NoiseQuantification(x =x , time_step = time_step, end_time= end_time)
#NoiseQuantification(x = data.VIB_LAT[0:36000].values, time_step = 300, end_time= 36000)
#%% NoiseQuantification by measuring Coeffiecient of Variation
#from scipy.stats import variation
corr = correlation(parameters)
def QuartileCOD(x): #Quartile Coefficient of Dispersion
    x.sort()
    q1, q3 = np.percentile(x,[23,75])
    return (q3 - q1)/(q1+q3)
    
def Coeff_ofvariance(x):
    mean_x = np.mean(x)
    var_x = np.std(x)
    coef_ofVarianse = var_x/mean_x
    return coef_ofVarianse
def NoiseQuantificationCV(x,time_step, end_time):
    x_noise = x
    #b, a = signal.butter(3, 0.05)
    #x_noisless = signal.filtfilt(b=b, a=a, x = x)
    y = data.N2_RATE[0:end_time].values # N2 Rate data
    y_norm = corr.normalize_data(y)
    y_iprs_norm = corr.normalize_data(data.IPRS_RAW[0:end_time])
    x_noisless = signal.savgol_filter(x_noise, window_length = 200, polyorder = 3)
    plt.figure(figsize= (12,9))
    plt.subplot(411)
    plt.plot(np.arange(0,len(y_norm),1)/60,y_norm)
    plt.plot(np.arange(0,len(x_noise),1)/60,corr.normalize_data(x_noise))
    plt.plot(np.arange(0,len(x_noise),1)/60,corr.normalize_data(x_noisless))
    plt.ylabel('Normalized Scale')
    plt.legend(['N2 Rate','Noisy data','Filtered data'], loc = 'best')
    #n_step = int(len(x_noise[0:end_time-time_step])/time_step)
    Coeff_ofvar_noise = []
    Coeff_ofvar_noiseless = []
    for i in range(int(len(x_noise[0:end_time-time_step])/60)):
        x_noise_seg = x_noise[i*60  :(i+1)*60 + time_step]
        x_noisless_seg = x_noisless[i*60 :(i+1)*60 + time_step]
        coefOfvar_noise = QuartileCOD(x=x_noise_seg)
        Coeff_ofvar_noise.append(coefOfvar_noise)
        coefOfvar_noiseless = QuartileCOD(x=x_noisless_seg)
        Coeff_ofvar_noiseless.append(coefOfvar_noiseless)
    plt.subplot(412)
    #plt.plot(np.arange(0,len(p_correlations)*5,5),np.array(p_correlations))
    plt.plot(np.arange(5,len(Coeff_ofvar_noise)+5,1),np.array(Coeff_ofvar_noise))
    plt.plot(np.arange(5,len(Coeff_ofvar_noiseless)+5,1),np.array(Coeff_ofvar_noiseless))
    plt.plot(np.arange(0,len(y_norm),1)/60,y_iprs_norm)
    plt.ylabel('Coeff of Var.')
    plt.xlabel('Time (Minute)')
    plt.subplot(413)
    #plt.plot(np.arange(0,len(p_correlations)*5,5),np.array(p_correlations))
    plt.plot(np.arange(0,len(y_norm),1)/60,y_norm)
    plt.plot(np.arange(5,len(Coeff_ofvar_noise)+5,1),np.array(Coeff_ofvar_noise))
    plt.plot(np.arange(5,len(Coeff_ofvar_noiseless)+5,1),np.array(Coeff_ofvar_noiseless))
    plt.legend(['N2 Rate','COV'])
    plt.ylabel('Coeff of Var.')
    plt.xlabel('Time (Minute)')
    plt.xlim(450,700)
    plt.subplot(414)
    plt.plot(np.arange(0,len(y_norm),1)/60,y_norm)
    plt.plot(np.arange(5,len(Coeff_ofvar_noise)+5,1),np.array(Coeff_ofvar_noise))
    plt.plot(np.arange(5,len(Coeff_ofvar_noiseless)+5,1),np.array(Coeff_ofvar_noiseless))
    plt.legend(['N2 Rate','COV'])
    plt.ylabel('Coeff of Var.')
    plt.xlabel('Time (Minute)')
    plt.xlim(0,400)
    plt.ylim(0,0.7)
    plt.suptitle('VIB_LAT')
    plt.show()
    
end_time= 135000
time_step = 12
x = data.SHK_LAT[0:end_time].values
NoiseQuantificationCV(x =x , time_step = time_step, end_time= end_time)
#%% PTEMP_RAW unwanted data removal
for k in data[data['PTEMP_RAW'] < 0].index:
    j = 0
    while data['PTEMP_RAW'][k] < 0:
        k = k+1
        j = j+1
        if data['PTEMP_RAW'][k] > 0: 
            data.loc[k-j,'PTEMP_RAW'] = data.loc[k,'PTEMP_RAW']
#%% plot IPRS_RAW with other parameter for outlier analysis
parameters = ['TEMP_DNI_RAW', 'ATEMP_RAW', 'PTEMP_RAW', 'DAGR_Temp'] 
lw = 4
for i, parameter in enumerate(parameters):
    if parameter == 'IPRS_RAW':
        pass
    else:
        fig, ax1 = plt.subplots(4,1,figsize = (12,12))
        y = data[parameter][0:end_time]
        ax1[0].plot(np.arange(0,len(y),1)/60,y,'-',color='darkorange',linewidth=2, label=parameter)
        ax1[0].set_ylabel(parameter)
        ax2 = ax1[0].twinx()
        ax2.plot(np.arange(0,len(y),1)/60,data.IPRS_RAW[0:end_time],label = 'IPRS_RAW')
        ax2.set_ylabel('IPRS_RAW')
        ax2.legend(loc = 'best')
        ax1[0].legend(loc='center right')
        ax1[1].plot(np.arange(0,len(y),1)/60,y,'-',color='darkorange',linewidth=lw)
        ax1[1].set_ylabel(parameter)
        ax2 = ax1[1].twinx()
        ax2.plot(np.arange(0,len(y),1)/60,data.IPRS_RAW[0:end_time])
        ax2.set_ylabel('IPRS_RAW')
        ax1[1].set_xlim(-1,100)
        ax1[2].plot(np.arange(0,len(y),1)/60,y,'-',color='darkorange',linewidth=lw)
        ax1[2].set_ylabel(parameter)
        ax2 = ax1[2].twinx()
        ax2.plot(np.arange(0,len(y),1)/60,data.IPRS_RAW[0:end_time])
        ax2.set_ylabel('IPRS_RAW')
        ax2.set_xlabel('Time (Minute)')
        ax1[2].set_xlim(180,240)
        ax1[3].plot(np.arange(0,len(y),1)/60,y,'-',color='darkorange',linewidth=lw)
        ax1[3].set_ylabel(parameter)
        ax2 = ax1[3].twinx()
        ax2.plot(np.arange(0,len(y),1)/60,data.IPRS_RAW[0:end_time])
        ax2.set_ylabel('IPRS_RAW')
        ax1[3].set_xlabel('Time (Minute)')
        ax1[3].set_xlim(1100,1250)
    fig.suptitle(parameter)
    fig.subplots_adjust(left=None, bottom=None, right=None,
            top=None, wspace=0.3, hspace=0.3)
    plt.show()
#%% overall correlation measurment
data = pd.read_csv("C:/Users/AKumar340/OneDrive - SLB/2024/CTD_EventDetection/Data/O.1011724.56-7.csv")
data1 = data[parameters]
#%%
import seaborn as sns
corr_mat = data1.corr()
print(corr_mat)

plt.figure(figsize=(14, 10))
sns.heatmap(corr_mat, annot=True, cmap='cubehelix', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()
#%%

            








