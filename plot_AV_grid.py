import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd

file = 'sweep_av_grids_3rev'
# file = 'sweep_av_grids_3rev_PENonly'
data = pd.read_pickle(f'results/AVgrid/{file}.pkl')
sns.set_style('white')
print(data.head)

df = data.loc[(data['AV'] >25) ]
                        
sns.violinplot(data=df, x="AV", y="RMSE", hue="Grid Type")
plt.show()

plt.figure(figsize=(3,3))
sns.lineplot(
    data=df,
    x="AV", y="RMSE",hue="Grid Type", style="Grid Type",
    markers=False, dashes=False
)
plt.ylim([0,11])
# df = data.loc[(data['AV'] ==60) ]
# print(df['RMSE'].iloc[0])
# plt.plot(60, df['RMSE'].iloc[0], '*',c='r')
plt.ylabel('RMSE (deg)')
plt.xlabel('Angular Velocity (deg/s)')
sns.despine()
plt.savefig(f'results/AVgrid/{file}.svg', bbox_inches='tight')
plt.savefig(f'results/AVgrid/{file}.png', bbox_inches='tight')

plt.show()


sns.lineplot(
    data=df, x="AV", y="RMSE", hue="Grid Type", err_style="bars", errorbar=("se", 2),
)
plt.show()

