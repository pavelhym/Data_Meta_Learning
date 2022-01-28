import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#all df
all_df = pd.read_csv("all_df.csv")

plt.figure(figsize=(8,6), dpi= 80)
sns.violinplot(x='predicted_class', y='real_quality', data=all_df, scale='width', inner='quartile')

plt.figure(figsize=(8,6), dpi= 80)
sns.boxplot(x='predicted_class', y='real_quality', data=all_df, notch=False)


plt.figure(figsize=(15, 10))
# Create violin plots without mini-boxplots inside.
ax = sns.violinplot(x='predicted_class', y='real_quality', data=all_df,
                    color='mediumslateblue', 
                    cut=0, inner=None)
# Clip the right half of each violin.
for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width/2, height,
                       transform=ax.transData))
# Create strip plots with partially transparent points of different colors depending on the group.
num_items = len(ax.collections)
sns.stripplot(x='predicted_class', y='real_quality', data=all_df,
              palette=['blue', 'deepskyblue'], alpha=0.4, size=7)
# Shift each strip plot strictly below the correponding volin.
for item in ax.collections[num_items:]:
    item.set_offsets(item.get_offsets() + 0.15)
# Create narrow boxplots on top of the corresponding violin and strip plots, with thick lines, the mean values, without the outliers.
sns.boxplot(x='predicted_class', y='real_quality', data=all_df, width=0.25,
            showfliers=False, showmeans=True, 
            meanprops=dict(marker='o', markerfacecolor='darkorange',
                           markersize=10, zorder=3),
            boxprops=dict(facecolor=(0,0,0,0), 
                          linewidth=3, zorder=3),
            whiskerprops=dict(linewidth=3),
            capprops=dict(linewidth=3),
            medianprops=dict(linewidth=3))
plt.legend(frameon=False, fontsize=15, loc='lower left')
#add_cosmetics(xlabel='Color', ylabel='Price, USD')






#smape

df_smape = pd.read_csv("for_plot_smape.csv")


plt.figure(figsize=(8,6), dpi= 80)
sns.boxplot(x='level', y='smape',hue = 'quality', data=df_smape, notch=False)

plt.figure(figsize=(8,6), dpi= 80)
sns.violinplot(x='level', y='smape',hue = 'quality', data=df_smape, notch=False)


#test

df_test = pd.read_csv("for_plot_test.csv")

#test training
plt.figure(figsize=(8,6), dpi= 80)
sns.boxplot(x='predicted_class', y='real_quality', data=df_test, notch=False)

#test classificator

plt.figure(figsize=(8,6), dpi= 80)
sns.boxplot(x='classificator_class', y='real_quality', data=df_test, notch=False)


#confusion matrix

from sklearn.metrics import confusion_matrix


cm = confusion_matrix(df_test["predicted_class"], df_test["classificator_class"])


cm_df = pd.DataFrame(cm,
                     index = ['1','2','3','4', '5'], 
                     columns = ['1','2','3','4', '5'])




#Plotting the confusion matrix
plt.figure(figsize=(5,5))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()