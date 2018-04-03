import numpy as np
import pandas as pd
import math

features = ['age',  #: continuous.
            'workclass',
            #: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
            'fnlwgt',  #: continuous.
            'education',
            #: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
            'education-num',  #: continuous.
            'marital-status',
            #: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
            'occupation',
            #: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
            'relationship',  #: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
            'race',  #: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
            'sex',  #: Female, Male.
            'capital-gain',  #: continuous.
            'capital-loss',  #: continuous.
            'hours-per-week',  #: continuous.
            'native-country',
            #: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.]
            'target']
category = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

train_data = pd.read_csv('adult-new.data', header=None, names=features)
train_target = train_data.target
train_data = train_data[features[:-1]]

test_data = pd.read_csv('adult-new.test', header=None, names=features)
test_target = test_data.target
test_data = test_data[features[:-1]]

train_data_dummies = pd.get_dummies(train_data, columns=category)
test_data_dummies = pd.get_dummies(test_data, columns=category)

for col in train_data_dummies.columns:
    if col not in test_data_dummies.columns:
        test_data_dummies[col] = 0

test_female_dummies = test_data_dummies[test_data_dummies['sex_ Female'] == 1]
test_male_dummies = test_data_dummies[test_data_dummies['sex_ Male'] == 1]
female = list(test_female_dummies.index.values)
male = list(test_male_dummies.index.values)

train_target = train_target.replace(' >50K', 1)
train_target = train_target.replace(' <=50K', 0)
test_target = test_target.replace(' >50K', 1)
test_target = test_target.replace(' <=50K', 0)
test_female_target = test_target[female]
test_male_target = test_target[male]

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

models = {'LR': LogisticRegression(),
          'RF': RandomForestClassifier(n_estimators=150),
          'GB': GradientBoostingClassifier()}

n_features = int(math.sqrt(train_data_dummies.shape[1]))

parameter_choice = {'LR': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
                    'RF': {'max_features': [n_features - 5, n_features, n_features + 5, n_features + 10]},
                    'GB': {'max_depth': [3, 5, 7]}}

grid_result = {}

for model in models:
    grid = GridSearchCV(models[model], parameter_choice[model], return_train_score=True)
    grid.fit(train_data_dummies, train_target)

    grid_result[model] = {'best_params_': grid.best_params_,
                          'best_score_': grid.best_score_,
                          'best_estimator_': grid.best_estimator_,
                          'cv_results': pd.DataFrame(grid.cv_results_)}

for model in grid_result:
    print(model, 'train score', 1 - grid_result[model]['best_estimator_'].score(train_data_dummies, train_target))
    print(model, 'test score', 1 - grid_result[model]['best_estimator_'].score(test_data_dummies, test_target))


def rate(classifier, testdata, testtarget):
    prediction = classifier.predict(testdata)
    wholeset = set(range(len(testdata)))
    #     print(wholeset)
    predict_positive = {i for i in np.nonzero(prediction)[0]}
    real_positive = {i for i in np.nonzero(testtarget)[0]}
    true_positive_rate = len(predict_positive & real_positive) / len(real_positive)

    predict_negative = wholeset - predict_positive
    real_negative = wholeset - real_positive

    false_positive_rate = len(real_negative & predict_positive) / len(real_negative)
    #     print('False Positive Rate of:',false_positive_rate)
    false_negative_rate = 1 - true_positive_rate
    #     print('False Negative Rate:',false_negative_rate)
    return (false_positive_rate, false_negative_rate)


for model in grid_result:
    print('female_data', model, 'False Positive, False Negative',
          rate(grid_result[model]['best_estimator_'], test_female_dummies, test_female_target))

for model in grid_result:
    print('male_data', model, 'False Positive, False Negative',
          rate(grid_result[model]['best_estimator_'], test_male_dummies, test_male_target))
