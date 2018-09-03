import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../dataset/pima-indians-diabetes.csv',
                 names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI",
                          "pedigree", "age", "class"])

"""print(df[['pregnant','class']].groupby(['pregnant'], as_index=False).mean().sort_values(by='pregnant', ascending=True))""" # 발생할 확률 알아보기

"""
plt.figure(figsize=(12,12))
#heatmap 두 항목의 패턴을 파악해 서로 비슷한 방향으로 갈수록 1을 출력함
sns.heatmap(df.corr(), linewidths=0.1,vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=False)# vmax : 밝기 , cmap : matplotlib 색상의 설정값 불러옴
""" # 가장 중요한 역할을 하는 항목 알아보기

grid = sns.FacetGrid(df, col="class")
grid.map(plt.hist,'plasma', bins=10)
plt.show()
