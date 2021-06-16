# 23. 결정 트리로 악성/양성 유방암 분류하기

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

# 데이터 셋 불러 오기
cancer_data = load_breast_cancer()

# 데이터 셋을 살펴보기 위한 코드
#print(cancer_data.DESCR)

# Pandas 데이터프레임 만들기
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])

# train, test set 만들기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 결정 트리 모델 만듬
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = model.score(X_test, y_test)

# 출력 코드
print(predictions)
print(score)