# UCI Heart Disease Problem
[link to data](https://www.kaggle.com/redwankarimsony/heart-disease-data)
><img src="https://i.imgur.com/vtFkSQC.png"  width="200" align="left"> Heart problems can have many different causes: poor diet, obesity, co-morbidities, complications after surgeries, smoking etc. It is worth paying attention to this aspect of our lives and make appropriate changes. To live as long as possible. 

------

# Data

[Link to preprocessing the dataset.](https://github.com/m0gr1m/UCI_Heart_Disease/blob/main/main_analysis.ipynb)

[Link to models.](https://github.com/m0gr1m/UCI_Heart_Disease/blob/main/models.ipynb)

------

# Motivation

Heart disease has been present in my family for a long time so I know how important it is to detect possible disease early and take appropriate treatment. The dataset I dealt with contained over a dozen variables, many of which had missing data, which I dealt with in my master's thesis. 

## Main goal 

+ Building a classification model that will predict heart disease.

------

# Summary

The dataset consisted of 14 independent variables and 1 dependent variable (*num*), which describes the predicted class: heart disease. It is not a binary variable, but consists of five digits from 0 to 4. It can be guessed that it describes different stages of heart disease: 0 - none, 1:4 - sick person (*from mild to severe condition*). Among the 14 independent variables, eight of them are of qualitative type, so it will be necessary to use appropriate encoding, which will significantly affect the size of the dataset. 

**Missing data were a major problem.** Three variables describing: the slope of the peak exercise ST segment, the number of major vessels stained by fluoroscopy, and an inherited blood disorder that causes the body to have less hemoglobin than normal had respectively: 33.59%, 66.41% and 52.83% missing values. The image below shows the location of missing data across the dataset (*dark color - available value*, *white color - missing data*). 

<img src="https://i.imgur.com/Kd4FCE1.png"  width="600">

For two variables: *ca* and *thal*, the gaps are too large to consider leaving them in the model. Any missing data in quantitative variables are relatively few, so I used imputation using the kNN algorithm (*number of neighbors: 3*). Data gaps in qualitative variables are, in my opinion, gaps that should not be imputed, so to make the best use of the remaining information **I decided to create two datasets**: df1 - contains the *slope* variable and has all data gaps removed (*we lost 41.20% of the entire dataset*), df2 - I removed the *slope* variable and the remaining data gaps (*we lost 15.87% of the dataset*).

**Extreme outliers** were removed from the dataset based on three times IQR (1.5 * IQR *is standardly used*). The data sets prepared in this way were saved and then used to build the classification models. 

**Model building phase** <br />
Since most classification algorithms could not obtain convergence with the current class numbers of the dependent variable I decided to combine the labels: 2, 3, 4 into one: 2.

<img src="https://i.imgur.com/YDPywGK.png"  width="600">
<img src="https://i.imgur.com/a4vhN88.png"  width="600">

Using the *make_column_transformer* function, we determined the appropriate transformation of the data and then built a loop that would indicate the best model from the list: *kNN*, *DecisionTree*, *Logistic Regression* and *NuSVC*.
Since the problem we are considering is a medical one, we changed the evaluation of the models to recall, so as to find out how the model performs in finding samples of a given class. 

Of the four models, **Logistic Regression** turned out to be the best (*in both cases: for df1 and df2*), so using the *GridSearchCV* function, we optimized the parameters and then trained two models whose recall is respectively: 70% and 72% for df1 and df2. On the test set, on the other hand, the recall looks as follows: 61.61% and 69.48% for df1 and df2, respectively. 

------

# Conclusions

Removing the *slope* variable that had more than 30% missing data and then removing the remaining cases that had missing data resulted in a better model. It can be assumed that variables that are largely contain NaN values should be removed during the procedure of building machine learning models. 
As can also be read from the confusion_matrix ([available in the file at the very bottom](https://github.com/m0gr1m/UCI_Heart_Disease/blob/main/models.ipynb)) the most disturbing is the first class, which is assigned the largest number of incorrect predictions from both the second and null classes; this may suggest that this class should be included in the severely diseased individuals (*class 2*), as additional testing in their case is needed and the model could gain a better result. 
