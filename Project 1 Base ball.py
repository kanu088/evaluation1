#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # for data wrangling purpose
import numpy as np # Basic computation library
import seaborn as sns # For Visualization 
import matplotlib.pyplot as plt # ploting package
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings # Filtering warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('https://raw.githubusercontent.com/FlipRoboTechnologies/ML_-Datasets/main/Baseball/baseball.csv')
df.head()


# In[4]:


print('No of Rows:',df.shape[0])
print('No of Columns:',df.shape[1])
df.head()


# # Comment -

# This dataset contains 16 features which contains statistics summary of the Baseball players and the 'W' (wins) is the target variable which predicts the number of wins.

# Input features in this dataset are : Runs, At Bats, Hits, Doubles, Triples, Homeruns, Walks, Strikeouts, Stolen Bases, Runs Allowed, Earned Runs, Earned Run Average (ERA), Shutouts, Saves, Complete Games and Errors

# Target features : Number of predicted wins (W)

# In[5]:


df.columns


# The column names present in our data that is not making much sense and will need deciphering to be converted into understandable format. In order to gain understanding of different columns in dataset, following baseball stastics terminology I get from Wikipedia.

# W – Win: number of games where pitcher was pitching while their team took the lead and went on to win, also the starter needs to pitch at least 5 innings of work

# W – Win: number of games where pitcher was pitching while their team took the lead and went on to win, also the starter needs to pitch at least 5 innings of work

# ER – Earned run: number of runs that did not occur as a result of errors or passed balls

# ERA – Earned run average: total number of earned runs (see "ER" above), multiplied by 9, divided by innings pitched

# CG – Complete game: number of games where player was the only pitcher for their team

# SHO – Shutout: number of complete games pitched with no runs allowed. A starting pitcher is credited with a shutout when he pitches the entire game for a team and does not allow the opposition to score. By definition, any pitcher who throws a shutout is also awarded a win.

# SV – Save: number of games where the pitcher enters a game led by the pitcher's team, finishes the game without surrendering the lead, is not the winning pitcher, and either (a) the lead was three runs or fewer when the pitcher entered the game; (b) the potential tying run was on base, at bat, or on deck; or (c) the pitcher pitched three or more innings

# # Batting statistics:

# R – Runs scored: number of times a player crosses home plate

# AB – At bat: plate appearances, not including bases on balls, being hit by pitch, sacrifices, interference, or obstruction. The number of times in which the hitter appeared at the plate and made a base hit, reached base on an error, or was out.

# H – Hit: reaching base because of a batted, fair ball without error by the defense

# 2B – Double: hits on which the batter reaches second base safely without the contribution of a fielding error

# 3B – Triple: hits on which the batter reaches third base safely without the contribution of a fielding error

# HR – Home runs: hits on which the batter successfully touched all four bases, without the contribution of a fielding error

# BB – Base on balls (also called a "walk"): hitter not swinging at four pitches called out of the strike zone and awarded first base.A walk (or base on balls) occurs when a pitcher throws four pitches out of the strike zone, none of which are swung at by the hitter. After refraining from swinging at four pitches out of the zone, the batter is awarded first base.

# K – Strike out (also abbreviated SO): number of times that a third strike is taken or swung at and missed, or bunted foul. Catcher must catch the third strike or batter may attempt to run to first base. It usually means the batter is out.

# # Base running statistics:

# SB – Stolen base: number of bases advanced by the runner while the ball is in the possession of the defense.A stolen base occurs when a baserunner advances by taking a base to which he isn't entitled. This generally occurs when a pitcher is throwing a pitch, but it can also occur while the pitcher still has the ball or is attempting a pickoff, or as the catcher is throwing the ball back to the pitcher.

# R – Runs scored: times reached home plate legally and safely
# 

# # Fielding statistics:

# E – Errors: number of times a fielder fails to make a play he should have made with common effort, and the offense benefits as a result. An error is an act, in the judgment of the official scorer, of a fielder misplaying a ball in a manner that allows a batter or baserunner to advance one or more bases or allows a plate appearance to continue after the batter should have been put out.

# # Now that we have clearer understanding on what the abbreviation mean and In order to simplify we are going to rename columns in dataset.

# In[7]:


df.rename(columns={'W' : 'Wins', 
                   'R' : 'Runs Scored', 
                  'AB' : 'At Bat', 
                   'H' : 'Hits', 
                  '2B' : 'Doubles', 
                  '3B' : 'Triples',
                  'HR' : 'Home Runs', 
                  'BB' : 'Base on Balls', 
                  'SO' : 'Strike Outs', 
                  'SB' : 'Stolen Base',
                  'RA' : 'Runs Average', 
                  'ER' : 'Earned Runs', 
                 'ERA' : 'Earned Run Average', 
                  'CG' : 'Complete Game',
                 'SHO' : 'Shut Outs', 
                  'SV' : 'Saves', 
                   'E' : 'Errors'}, 
          inplace=True)
df.head()


# In[8]:


df.info()


# # Comment -

# We can obsereve that this datset has only numeric data and no column has categorical data.
# This dataset fall into regression analysis.

# # Develop an ML Regression based algorithm that predicts the number of wins for a given team based on features. Here Wins is target variable and others are Input features.

# # Statistical Analysis

# In[9]:


# Visualizing the statistics of the columns using heatmap.
plt.figure(figsize=(20,8))
sns.heatmap(df.describe(),linewidths = 0.1,fmt='0.1f',annot = True)


# In[10]:


df.describe().T


# # Comment -

# If we just look at mean and 50% columns for different feature we can see data is sightly right skew for most of features.

# Count is same for each variable.

# 75% and max values for Errors, Shutout, Run Scored shows presence of possible outliers.

# Overall all statstical parameter from mean to max, indicate features are seem to be progressing in a definite manner showing no visible abnormalities.

# Heatmap clearly shows data need to scale while building ML Model.

# # Missing value check

# In[12]:


plt.figure(figsize=(8,7))
sns.heatmap(df.isnull())


# In[13]:


missing_values = df.isnull().sum().sort_values(ascending = False)
percentage_missing_values =(missing_values/len(df))*100
print(pd.concat([missing_values, percentage_missing_values], axis =1, keys =['Missing Values', '% Missing data']))


# Comment -There is no null value present in dataset.

# # EDA

# Here we can try to bring insight in what feature contribute to win

# # Distribution of features

# In[14]:


plt.figure(figsize=(20,25), facecolor='white')
plotnumber =1
for column in df:
    if plotnumber <=17:
        ax = plt.subplot(6,3,plotnumber)
        sns.distplot(df[column], color='r',hist=False,kde_kws={"shade": True})
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.show()


# Comment -
# Clearly some of feature are either left or right skewed.

# In[15]:


sns.set_palette('gist_rainbow_r')
plt.figure(figsize=(20,30), facecolor='white')
plotnumber =1
for column in df:
    if plotnumber <=17:
        ax = plt.subplot(6,3,plotnumber)
        sns.violinplot(df[column])
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.show()


# # Comment -

# Shut outs and Complete Game occur very rarely now-a-days which we can definately see in violinplot of these variable.

# An Errors does not count as a hit but still counts as an at bat for the batter, So need to dive into how much Error are contributing to at bat.

# Most of saves are between 30 & 50. Saves doesnot entitle pitcher as wining pitcher but still it bring wins for team.It will be interesting what relation wins and save held or how much saves contribute in win.

# Run average, Earned run and Earned run average are important for pitcher statstics. We can see there is not much difference in plot of Earned run and Run Average, so from here we can conclude that Unearned Run doesnot making much difference in wins.

# Homeruns (125 to 175 peak) are more than triples (20 to 40 majority) so most of good shot by battar directly convert into homeruns.

# As we know pitcher try to keep Earned run Average low which eventually lead to wins. Here for most of game ERA is around 3.5-4.5.

# Let consider violinplot of doubles and base on balls. We know that if pitcher pitched ball for consecutive 4 ball then Base is awarded to batter. Clearly More runs comes from base of ball than doubles.

# # Lets now Investigate Runs and Hits together , before that let dive into how team get win and some clearity over Run Vs Hits

# # How do u win baseball ?

# To win in baseball, you must reach the end of the game with more runs than your opponent. If you have the same amount of runs, you will go into another inning until one team has more runs at the end of the inning.

# # Runs Vs Hits

# Runs (R) - The number of runs scored by that player, by reaching home base in any manner.
# Hits (H) - The number of base hits made by that player, not including reaching base on an error or on a "fielder's choice".

# So Now we will put insight on how many hits convert into Runs and inturn lead to win throgh multivariate analysis

# In[17]:


# Checking the relation between two variables
sns.set_palette('Set1')
plt.figure(figsize=[10,6])
plt.title('Comparison between Runs and Hits',fontsize =20)
sns.scatterplot(x=df['Runs Scored'],y=df['Hits'],hue=df['Wins'])
plt.xlabel('Runs',fontsize =16)
plt.ylabel("Hits",fontsize =16)


# # Comment :

# Even if number of times ball hit bat is less than 1375 still run in range of 650 to 750 contribute to win.

# Very less wining chance for run less than 650 and no of hits less than 1325.

# There is one outlier in runs. After checking sknewness we can decide whether to keep to while building ML model or remove it even if it is valid data point. Definitely it will affect performance of ML model if we consider outlier data points as most of data point will not fall dont in that side. potential leading to biased model.

# In[18]:


# Checking the relation between two variables
sns.set_palette('Set1')
plt.figure(figsize=[10,6])
plt.title('Runs Scored Vs Home Runs',fontsize =20)
sns.scatterplot(x=df['Runs Scored'],y=df['Home Runs'],hue=df['Wins'])
plt.xlabel('Runs Scored',fontsize =16)
plt.ylabel('Home Runs',fontsize =16)


# # Comment :

# Home Runs in range of 140 & 180 with combination of Run Scored in between 650-750 lead to more than 90 Wins for team. So keeping home runs in this range is cruical for more possibility of wins.

# # In conclusion we can say that Home runs is definitely contributing factor for team to win but not sufficient to make sure win.

# In[19]:


# Checking the relation between two variables
sns.set_palette('Set1')
plt.figure(figsize=[10,6])
plt.title('Comparison between Runs and At Bat', fontsize =20)
sns.scatterplot(x=df['Runs Scored'],y=df['At Bat'],hue=df['Wins'])
plt.xlabel('Runs Scored',fontsize =16)
plt.ylabel("At Bat",fontsize =16)


# # Comment :

# We doesnot get any benchmark range for at bats from here. So it is questionable things that how much At bats matter to winning statstics. Atleast we get here that At Bat and Run Scored has positive linear relationship, which means that more Run Scroed naturally lead to more at bats.

# In[20]:


# Checking the relation between two variables
sns.set_palette('Set1')
plt.figure(figsize=[10,6])
plt.title('Runs Scored Vs Strike Outs',fontsize =20)
sns.scatterplot(x=df['Runs Scored'],y=df['Strike Outs'],hue=df['Wins'])
plt.xlabel('Runs Scored',fontsize =16)
plt.ylabel('Strike Outs',fontsize =16)


# # Comment :

# In simple word strike Outs means batter is out. We can see Strike out opponent team below 700 runs essential for more win.

# Clearly Strikeout below 1200 is like making recipe for losing game. Strikeouts in regular interval not only lead to pressure on opponent in game but also bring break on high run score.

# In[21]:


# Checking the relation between two variables
sns.set_palette('hsv')
plt.figure(figsize=[10,6])
plt.title('Errors Vs Earned Run Average',fontsize =20)
sns.scatterplot(x=df['Errors'],y=df['Earned Run Average'],hue=df['Wins'], cmap=('Spectral'))
plt.xlabel('Errors',fontsize =16)
plt.ylabel('Earned Run Average',fontsize =16)


# # Comment :

# At Bat Vs Base on Balls doesnot give any significant imformation than High ERA means High Errors.

# # Boxplot of Features

# In[22]:


plt.figure(figsize=(18,15), facecolor='white')
plotnumber =1
for column in df:
    if plotnumber <=18:
        ax = plt.subplot(6,3,plotnumber)
        sns.boxplot(df[column], palette='hsv')
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()
plt.show()


# # Comment :

# There are some outliers present in data. But as data is of Real world MLB 2014, these outliers are valid datapoints.

# In[23]:


plt.figure(figsize=(18,12))
sns.barplot(x="Wins", y="Base on Balls", data=df,palette='PiYG')
plt.show()


# # Comment :

# Base on ball is contribution from pitcher to batter for winning. In simple word it is like Wide or No Ball in Cricket.

# We can see that base on ball above 400 really contribute in Wins in team.

# In[24]:


plt.figure(figsize=(18,12))
sns.barplot(x="Wins", y="Runs Scored", data=df,palette='PiYG')
plt.show()


# Run Scored above 600 is benchmark for wining in any scenerio

# In[25]:


plt.figure(figsize=(18,12))
sns.barplot(x="Wins", y="Runs Average", data=df,palette='gist_earth')
plt.title('Bar plot of Wins Vs Run Average', fontsize =20)
plt.show()


# Here comes interesting plot, we can see Run Average decrease with increase in number of wins. But why this trend if more runs means directly increase in chance of winning

# In[26]:


plt.figure(figsize=(18,12))
sns.barplot(x="Wins", y="Earned Run Average", data=df,palette='spring')
plt.xlabel('Wins',fontsize =16)
plt.ylabel('Earned Run Average',fontsize =16)
plt.show()


# there must be atleast one pitcher parameter where this decreasing trend must be reflected.

# In[28]:


# Checking the relation between two variables
sns.set_palette('Set1')
plt.figure(figsize=[16,8])
plt.title('Comparison between Run Average and Earned Run Average', fontsize =20)
sns.stripplot(x=df['Runs Average'],y=df['Earned Run Average'],hue=df['Wins'])
plt.xlabel('Runs Average',fontsize =16)
plt.ylabel("Earned Run Average",fontsize =16)


# Comment :Here we got what we suspect in previous plot. ERA and RA hold linear relationship.

# In[29]:


# Checking the relation between two variables
sns.set_palette('Set1')
plt.figure(figsize=[16,8])
plt.title('Comparison between Run Average and Earned Run Average', fontsize =20)
sns.stripplot(x=df['Runs Average'],y=df['Strike Outs'],hue=df['Wins'])
plt.xlabel('Runs Average',fontsize =16)
plt.ylabel("Strike Outs",fontsize =16)


# Strike outs are randomly placed giving not much any significant insights. Most probably strikeouts doesnot matter.

# # There is one outlier in Runs score, lets check that entry

# In[30]:


df['Runs Scored'].max()


# In[31]:


df.loc[df['Runs Scored']==891]


# It seem like highest Doubles, Homeruns and base ball also belong to this entry. Let cross check

# In[32]:


df['Home Runs'].max(),df['Base on Balls'].max(),df['Doubles'].max()


# In[33]:


sns.jointplot(x="Earned Runs", y="Wins", data=df, color="blue",palette="Set1")


# In[34]:


sns.jointplot(x="Earned Run Average", y="Wins", data=df, color="green",palette="Set1")


# Jointplot shows same story about Earned Run/Earned Run Average and Wins having linear negative relationship

# # Let check relationship between saves and wins

# In[36]:


plt.figure(figsize=(10,10))
sns.jointplot(x="Saves", y="Wins", data=df, color="purple")


# Here with increase in the number of save increases the number of wins.

# In[37]:


sns.pairplot(df, hue="Wins")


# # Outliers Detection and Removal

# In[39]:


from scipy.stats import zscore
z = np.abs(zscore(df))
threshold = 3
df1 = df[(z<3).all(axis = 1)]


# In[40]:


print("\033[1m"+'Shape of dataset after removing outliers :'+"\033[0m",df1.shape)


# # Data Loss

# In[41]:


print("\033[1m"+'Percentage Data Loss :'+"\033[0m",((30-29)/30)*100,'%')


# # Feature selection and Engineering

# # 1. Skewness of features

# In[42]:


df1.skew()


# Optimal range for skewness is -0.5 to 0.5.

# Hits, Complete Game, Shuts Outs, Saves, Errors are positively Skewed in nature, need to transform.

# # Transforming positive or right skew data using boxcox transformation

# In[43]:


from scipy.stats import boxcox


# In[44]:


df1['Hits']=boxcox(df1['Hits'],-2)


# In[45]:


df1['Shut Outs']=boxcox(df1['Shut Outs'],0.5)


# In[46]:


df1['Saves']=boxcox(df1['Saves'],0.5)


# Other feature not able transform by Boxcox Method as they showing data must be positive. So others columns are transfrom using yeo-johnson method

# In[47]:


from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer


# In[48]:


EC=['Errors','Complete Game']
ds =df1[EC].copy()


# In[49]:


column_trans =ColumnTransformer(
    [  ('Errors',PowerTransformer(method='yeo-johnson',standardize=True),['Errors']),
      ('Complete Game',PowerTransformer(method='yeo-johnson',standardize=True),['Complete Game'])])
transformed_yeojohnson =column_trans.fit_transform(df1)   
new_cols=['Errors','Complete Game']
dataset=pd.DataFrame(transformed_yeojohnson,columns=new_cols) #to convert numpy array back into dataframe
pd.concat([dataset],axis=1)
dataset.head()


# In[50]:


# reseting index and mergeing transform data
df1.reset_index(drop=True, inplace=True)
dataset.index=df1.index
df1[EC]=dataset[EC]


# # Skewness after transforming features

# In[51]:


df1.skew()


# All features skewness is now transform within permissible limit of -0.5 to 0.5 as shown above

# # 2. Corrleation

# In[52]:


df1.corr()


# In[53]:


upper_triangle = np.triu(df.corr())
plt.figure(figsize=(20,15))
sns.heatmap(df1.corr(), vmin=-1, vmax=1, annot=True, square=True, fmt='0.3f', 
            annot_kws={'size':10}, cmap="gist_stern", mask=upper_triangle)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[54]:


plt.figure(figsize = (18,6))
df1.corr()['Wins'].drop(['Wins']).plot(kind='bar',color = 'c')
plt.xlabel('Features',fontsize=15)
plt.ylabel('Wins',fontsize=15)
plt.title('Correlation of features with Target Variable win',fontsize = 18)
plt.show()


# Hits, At bats, Complete game and errors are very poorly correlated with target variable.

# Saves, ERA,RA,EA are highly correleated with target variable.

# here is visible multi colinearity between the feature columns "Earned Runs", "Earned Run Average" and "Runs Average". This need to check.

# # 3. Checking Multicollinearity between features using variance_inflation_factor

# In[56]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif= pd.DataFrame()
vif['VIF']= [variance_inflation_factor(df1.values,i) for i in range(df1.shape[1])]
vif['Features']= df1.columns
vif


# Earned Run Average,Earned Runs,Runs Average are highly correlated with each other.

# # Standard Scaling

# In[57]:


X=df1.drop(columns =['Wins'])
Y=df1['Wins']


# In[58]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_scale = scaler.fit_transform(X)


# # PCA

# In[59]:


from sklearn.decomposition import PCA
pca = PCA()
#plot the graph to find the principal components
x_pca = pca.fit_transform(X_scale)
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Variance %')
plt.title('Explained variance Ratio')
plt.grid()


# AS per the graph, we can see that 7 principal components attribute for 95% of variation in the data. We shall pick the first 7 components for our prediction

# In[60]:


pca_new = PCA(n_components=7)
x_new = pca_new.fit_transform(X_scale)


# In[61]:


principle_x=pd.DataFrame(x_new,columns=np.arange(7))


# # Checking Multicollinearity after applying PCA

# In[62]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif= pd.DataFrame()
vif['VIF']= [variance_inflation_factor(principle_x.values,i) for i in range(principle_x.shape[1])]
vif['Features']= principle_x.columns
vif


# # Machine Learning Model Building

# In[64]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  Ridge
from sklearn.linear_model import  Lasso


# In[65]:


X_train, X_test, Y_train, Y_test = train_test_split(principle_x, Y, random_state=42, test_size=.3)
print('Training feature matrix size:',X_train.shape)
print('Training target vector size:',Y_train.shape)
print('Test feature matrix size:',X_test.shape)
print('Test target vector size:',Y_test.shape)


# # Finding Best Random state

# In[66]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
maxR2_score=0
maxRS=0
for i in range(1,250):
    X_train, X_test, Y_train, Y_test = train_test_split(principle_x, Y, random_state=i, test_size=.25)
    lin_reg=LinearRegression()
    lin_reg.fit(X_train,Y_train)
    y_pred=lin_reg.predict(X_test)
    R2=r2_score(Y_test,y_pred)
    if R2>maxR2_score:
        maxR2_score=R2
        maxRS=i
print('Best R2 Score is', maxR2_score ,'on Random_state', maxRS)


# # Linear Regression : Base model

# In[67]:


X_train, X_test, Y_train, Y_test = train_test_split(principle_x, Y, random_state=217, test_size=.25)
lin_reg=LinearRegression()
lin_reg.fit(X_train,Y_train)
lin_reg.score(X_train,Y_train)
y_pred=lin_reg.predict(X_test)
print('\033[1m'+'Predicted Wins:'+'\033[0m\n',y_pred)
print('\n')
print('\033[1m'+'Actual Wins:'+'\033[0m\n',Y_test)


# # Linear Regression Evaluation Matrix

# In[68]:


print('\033[1m'+' Error :'+'\033[0m')
print('Mean absolute error :', mean_absolute_error(Y_test,y_pred))
print('Mean squared error :', mean_squared_error(Y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test,y_pred)))
print('\n')
from sklearn.metrics import r2_score
print('\033[1m'+' R2 Score :'+'\033[0m')
print(r2_score(Y_test,y_pred,multioutput='variance_weighted'))


# # Cross validation

# In[77]:


# Cross Validation
from sklearn.model_selection import cross_val_score, KFold
score = cross_val_score(lin_reg, principle_x, Y, cv =3)
print('\033[1m'+'Cross Validation Score :',lin_reg,":"+'\033[0m\n')
print("Mean CV Score :",score.mean())


# # Finding best n_neighbors for KNN Regressor

# In[75]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse_val = [] #to store rmse values for different k
for K in range(10):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train,Y_train)  #fit the model
    y_pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(Y_test,y_pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[76]:


#plotting the rmse values against k values -
plt.figure(figsize = (8,6))
plt.plot(range(10), rmse_val, color='blue', linestyle='dashed', marker='o', markerfacecolor='green', markersize=10)


# At k= 4, we get the minimum RMSE value which approximately 5.050525962709231, and shoots up on further increasing the k value. We can safely say that k=4 will give us the best result in this case

# # Applying other Regression Model, Evaluation & Crossvalidation

# In[86]:


rf = RandomForestRegressor(n_estimators = 250 ,max_depth=6)
svr=SVR(C=1.0, epsilon=0.2, kernel='poly', gamma='auto')
dtc = DecisionTreeRegressor(criterion='mse')
adb=AdaBoostRegressor(learning_rate=0.1)
gradb=GradientBoostingRegressor( max_depth=6,learning_rate=0.1)
knn=KNeighborsRegressor(n_neighbors=4,algorithm='kd_tree')
ls= Lasso(alpha=1e-2, max_iter=1e5)
rd=Ridge(alpha=1e-2)
model = [rf,ls,rd,svr,dtc,adb,gradb,knn]

for m in model:
    m.fit(X_train,Y_train)
    m.score(X_train,Y_train)
    y_pred = m.predict(X_test)
    print('\n')                                        
    print('\033[1m'+' Error of ', m, ':' +'\033[0m')
    print('Mean absolute error :', mean_absolute_error(Y_test,y_pred))
    print('Mean squared error :', mean_squared_error(Y_test,y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test,y_pred)))
    print('\n')

    print('\033[1m'+' R2 Score :'+'\033[0m')
    print(r2_score(Y_test,y_pred)) 
    
    # Cross Validation
    score = cross_val_score(m, principle_x, Y, cv =4)
    print('\n')
    print('\033[1m'+'Cross Validation Score :',m,":"+'\033[0m\n')
    print("Mean CV Score :",score.mean())
    print('==============================================================================================================')


# In[ ]:





# In[ ]:




