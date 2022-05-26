from cmath import sin
import pickle
from re import X
from statistics import mean
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from matplotlib.pyplot import MultipleLocator, figure, ginput
def is_nan_or_inf(value):
    return np.isnan(value) or np.isinf(value)

def xgboost_aggregate(maxt,seeds, algorithm, last_str, color, label, marker= None, linewidth= None):

    def lexico_compare(tem_y0, tem_y1, incumbent_metric1, incumbent_metric2, toler):
        if tem_y0 < incumbent_metric1:
            return 1
        elif tem_y0>= incumbent_metric1 and tem_y0<= incumbent_metric1+toler:
            if tem_y1 < incumbent_metric2:
                return 1
            else:
                return -1
        else:
            return -1

    x_all = []
    y_all = []
    for index,seed in enumerate(seeds):
        path_algorithm  = algorithm
        if last_str != None:
            path = os.path.join("./results/adult/","seed-"+str(seed)+"_"+"algorithm-"+"CFO"+last_str,"result.pckl")
        else:
            path = os.path.join("./results/adult/","seed-"+str(seed)+"_"+"algorithm-"+str(path_algorithm),"result.pckl")
        f = open(path,"rb")
        result = pickle.load(f)
        f.close()
        if algorithm in ["ours","CFO"]:
            origin =result
            cfo_x = []
            cfo_y= []
            for key,value in origin.items():
                if value!= {} and value["wall_clock_time"] is not None and value["train_time"] is not None and value["val_loss"] is not None:
                    cfo_x.append(value["wall_clock_time"])
                    if "incumbent_comparision" in value.keys() and value["incumbent_comparision"] != None:
                        if algorithm == "CFO":
                            cfo_y.append([value["incumbent_comparision"]["val_loss"],value["incumbent_comparision"]["train_time"]])
                        else:
                            cfo_y.append([value["incumbent_comparision"]["val_loss"],value["incumbent_comparision"]["train_time"]])     
                    else:
                        cfo_y.append([value["val_loss"],value["train_time"]])
        elif algorithm in ["parego","qehvi","random"]:
            origin =result
            cfo_x = []
            cfo_y= []
            for key,value in origin.items():
                cfo_x.append(value["wall_clock_time"])
                cfo_y.append([value["val_loss"],value["train_time"]])
        x_all.append(cfo_x)
        y_all.append(cfo_y)

    if algorithm in ["parego","qehvi","random","CFO"]:
        new_y_all=[]
        for seed_index,seed_item in enumerate(y_all):
            incumbent_metric1 = None    
            incumbent_metric2 = None
            f1_best = None
            new_seed_item = []
            for intem_index,intem_value in enumerate(seed_item):
                if incumbent_metric1 == None:                 
                    incumbent_metric1 = intem_value[0]
                    incumbent_metric2 = intem_value[1]
                    f1_best = intem_value[0]
                else:
                    if lexico_compare(intem_value[0],intem_value[1],incumbent_metric1,incumbent_metric2,maxt) == 1:
                        incumbent_metric1 = intem_value[0]
                        incumbent_metric2 = intem_value[1]
                    if incumbent_metric1 < f1_best:
                        f1_best = incumbent_metric1
                new_seed_item.append([incumbent_metric1,incumbent_metric2])         
            new_y_all.append(new_seed_item)
        y_all = new_y_all
    initial1_mean_all = []
    initial2_mean_all = []
    if algorithm in ["CFO"]:
        for index in range(len(y_all)):
            initial1_mean_all.append(y_all[index][0][0])
            initial2_mean_all.append(y_all[index][0][1])
    else:
        for index in range(len(y_all)):
            initial1_mean_all.append(y_all[index][0][0])
            initial2_mean_all.append(y_all[index][0][1])
    time_step_all = list(range(0, 5001, 1))
    y_final = []
    for seed_index, seed_y in enumerate(y_all):
        seed_x = x_all[seed_index] 
        tem_y_final=[]
        flag_exp = 0 
        flag_set = 0
        while flag_set<5001:
            while flag_exp<len(seed_x) and seed_x[flag_exp]<=flag_set:
                flag_exp=flag_exp+1
            tem_y_final.append(seed_y[min(flag_exp,len(seed_x)-1)])
            flag_set=flag_set+1
        y_final.append(tem_y_final)

    
    x_plot = time_step_all
    y_plot_all = np.mean(np.array(y_final),axis=0)
    y_std = np.std(np.array(y_final),axis=0)
    std1 = y_std[:,0]
    std2 = y_std[:,1]
    y_plot1= y_plot_all[:,0]
    y_plot2 = y_plot_all[:,1]
    fill1_high=[]
    fill1_low=[]
    fill2_high=[]
    fill2_low=[]
    seed_size = len(y_all)

    for index in range(len(x_plot)):
        fill1_high.append(y_plot1[index]+3.291*std1[index]/np.sqrt(seed_size))
        fill1_low.append(y_plot1[index]-3.291*std1[index]/np.sqrt(seed_size))
        fill2_high.append(y_plot2[index]+3.291*std2[index]/np.sqrt(seed_size))
        fill2_low.append(y_plot2[index]-3.291*std2[index]/np.sqrt(seed_size))
    y_plot2[0] =np.max(initial2_mean_all) 
    return x_plot, y_plot1,y_plot2,fill1_low,fill1_high,fill2_low,fill2_high


fig=plt.figure(1)


set = 0.001
listuse_2 = [1,2,3,4,5]
xg_parego_x_plot,xg_parego_y_plot1, xg_parego_y_plot2,xg_parego_fill1_low,xg_parego_fill1_high,xg_parego_fill2_low,xg_parego_fill2_high =xgboost_aggregate(set,listuse_2,"qehvi",None,'red',"qehvi",linewidth=5)
xg_our_x_plot,xg_our_y_plot1, xg_our_y_plot2,xg_our_fill1_low,xg_our_fill1_high,xg_our_fill2_low,xg_our_fill2_high =xgboost_aggregate(set,[1,2,3,4,5], "ours", "_C-0.0-0.0-_toler-0.001-1.0-",'red',"ours",linewidth=5)


ax2 = fig.add_subplot(1,1,1)
plt.tick_params(labelsize=11)
ax2.ticklabel_format(style='sci', scilimits=(-1,2),axis='y')
ax2.add_patch(
     Rectangle(
        (0., 0.07717),
        5000,
        0.001,
        fill=False,
        color = "red",
        label = "Optimality Tolerance Range",
        # linestyle = "dashed",
        linewidth = 2.0,   
     ) ) 
plt.plot(np.array(xg_parego_x_plot),np.array(xg_parego_y_plot1), linestyle = 'dashed',color = "darkorange",label = "MO-HPO",marker="d",linewidth=2.5,markersize=5,markevery=markerevery2)
plt.plot(np.array(xg_our_x_plot),np.array(xg_our_y_plot1), color = "tab:blue",label = "LexicoFlow",marker="d",linewidth=2.5,markersize=5,markevery=markerevery2)
plt.ylabel("loss = 1- roc_auc",fontsize = 20)
plt.xlabel("HPO wall-clock-time (s)",fontsize = 20)
plt.title("XGboost 1st Objective",fontsize = 20)
plt.legend(fontsize=16)
plt.ylim((0.077, 0.1))

plt.grid(True, which="major", ls="-")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax = plt.gca()
ax.set_facecolor('#F6F5F5')
plt.tight_layout()


plt.savefig("intro_a.png", dpi=None, facecolor='w', edgecolor='w',
          orientation='portrait', papertype=None, format=None,
          transparent=False, bbox_inches=None, pad_inches=0.0,
          frameon=None, metadata=None)

plt.show()

