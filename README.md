# Spaceship Titanic Prediction

## Objective <br>
The dataset was downloaded from Kaggle, [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data?select=train.csv)
, on 18 October 2024. <br>
This dataset is part of an open Kaggle competition, 
where the task is to predict whether a passenger was transported to an 
alternate dimension during the Spaceship Titanic's collision with the
spacetime anomaly. <br>
The data originally comes in 2 separate datasets, *train.csv* and *test.csv*
. Each dataset contains a set of personal records recovered from the ship's
 damaged computer system. There are 13 columns of personal records, and the 
 14th column is the target.

## Technology Used <br>
1. Python
2. Pandas
3. Matplotlib
4. Seaborn
5. 

## Results <br>
The features listed both on feature importance and SHAP are congruent with 
EDA analysis. However, the order and strength shown on Feature Importance was
 surprising. 
<br><br>
**SHAP and Feature Importance** <br>
As EDA showed us, CryoSleep followed by luxury amenities have
huge distinction between the 2 classes in SHAP. But in feature importance, 
these follow an opposite order. This maybe that CryoSleep is not important 
for splits in the decision tree, but is critical in specific context. 
CryoSleep is correlated with several other features, possibly CryoSleep are 
affected by those.<br>
It was also surprising to see Age, that were not given much  
attention/important, is high on feature importance.<br><br>

**Conclusion:**<br>
This is the first iteration of this project, with submission score 0.79962. 
A part of the project, which is not seen, is that LabelEncoder was first 
used, but changed to OrdinalEncoder after hyperparameter tuning proved 
the latter better, by a few percent.<br>
For the second iteration, I would spend more time on the following 
improvements:<br>

1. More thorough EDA and feature extraction/creation, because the final model 
interpretation had a few surprising details. For example, luxury amenities 
features can create a 'Total Spending' feature, which might have more impact.
2. Deeper analysis of encoding and imputation methods.
3. Try if PCA or Boruta would provide better feature selection.
4. Try CatBoost as the data has quite some categorical features, and other 
models.
5. Divide dataset into one more set for a final evaluation before submission.
<br><br>


**How do I use it:**
```shell
curl --location 'https://spaceship-prediction-953e7e237ee4.herokuapp.com/predict' \
--header 'Content-Type: application/json' \
--data '{
    "features": [
        ["0029_01", "Europa", true, "B/2/P", "55 Cancri e", 21.0, false, 0.0, 0.0, 0.0, 0.0, 0.0, "Aldah Ainserfle"],
        ["0029_01", "Europa", true, "B/2/P", "55 Cancri e", 21.0, false, 0.0, 0.0, 0.0, 0.0, 0.0, "Aldah Ainserfle"]
    ],
    "threshold": 0.96
}
'
```

output:
`
{
  "prediction": [
    1,
    1
  ],
  "probability": [
    0.9741316826843636,
    0.9741316826843636
  ]
}
`
