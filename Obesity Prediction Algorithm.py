## Import Library
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report


## Define File Path
path = r"C:\Users\user\Desktop\Projects\Obesity Predictor App\obesity new.csv"


## Define and read dataset 
df = pd.read_csv(path)


## Check the shape of the dataset
df.shape


## Check the columns of the dataset
df.columns


## Checking the missing values in the dataset
df.isna().sum()


## Checking the duplicated rows in the dataset
df.duplicated().sum()


## Removing the duplicated rows in the dataset
df =  df.drop_duplicates()


## Checking the unique values in the dataset
df.nunique()


## Converting the columns to lower case 
df.columns = df.columns.str.lower()


## Checking the value count for gender column
df['gender'].value_counts()


## Checking datatype of gender variable
df.gender.dtype


## replacing the values in the gender variable
df['gender'] = df['gender'].replace({1:'female', 0:'male'})


## converting the gender data type
df['gender'] = df['gender'].astype('category')


## Renaming the fammily history variables 
df = df.rename(columns={'family_history_with_overweight':'family_history'}) 


## Identifying the dependent Variable 
df['nobeyesdad'].value_counts()


## Splitting the data into Dependent and Independent variable
x = df.drop(['nobeyesdad'], axis = 1)
y = df['nobeyesdad']


## Undersampling the data
scaler = RandomUnderSampler()

resampled_x, resampled_y = scaler.fit_resample(x,y)


## Splitting the variables into numeric and categorical columns
num_cols = resampled_x.select_dtypes(include=['number']).columns
cat_cols = resampled_x.select_dtypes(include=['object', 'category']).columns


## Building the numeric pipeline 
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


## Building the categorical columns using Ordinal Encoder 
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])


## Building the categorical columns using OneHotEncoder
cat_pipe2 = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
])


## Creating a processor to cobine the columns 
processor = ColumnTransformer(
    transformers=[
        ('num_pipe', num_pipe, num_cols),
        ('cat_pipe', cat_pipe, cat_cols)
    ],
    remainder = 'passthrough'
)


processor2 = ColumnTransformer(
    transformers=[
        ('num_pipe', num_pipe, num_cols),
        ('cat_pipe2', cat_pipe2, cat_cols)
    ],
    remainder = 'passthrough'
)


## Creating and Building the training and testing sets
x_train, x_test, y_train, y_test = train_test_split(resampled_x, resampled_y, test_size = 0.2, random_state=45)


x_train.shape

x_test.shape


## Building the Random Forest Pipeline
rf_pipe = Pipeline([
    ('processor', processor),
    ('rf_model', RandomForestClassifier())
])
rf_model = rf_pipe.fit(x_train, y_train)

rf_pipe2 = Pipeline([
    ('processor2', processor2),
    ('rf_model2', RandomForestClassifier())
])
rf_model2 = rf_pipe2.fit(x_train, y_train)

rf_predict = rf_model.predict(x_test)


## Evaluating the Random Forest Model Prediction
confusion_matrix(y_test, rf_predict)

f1_score(y_test, rf_predict2)

classification_report(y_test, rf_predict2)

# Extract the preprocessing step (ColumnTransformer) from the pipeline
preprocessor = rf_model2.named_steps['processor2']

# Get the transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Access the RandomForestClassifier step from the pipeline
rf_classifier = rf_model2.named_steps['rf_model2']

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Combine feature names with their importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False).round(2)

print(importance_df.head(20))


# Extract the preprocessing step (ColumnTransformer) from the pipeline
preprocessor = rf_model2.named_steps['processor2']

# Get the transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Access the RandomForestClassifier step from the pipeline
rf_classifier = rf_model2.named_steps['rf_model2']

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Combine feature names with their importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False).round(2)

print(importance_df.head(20))



xgb_pipe = Pipeline([
    ('processor', processor2),
    ('xgb_model', XGBClassifier())
])
xgb_model = xgb_pipe.fit(x_train, y_train)


# Extract the preprocessing step (ColumnTransformer) from the pipeline
preprocessor = xgb_model.named_steps['processor']

# Get the transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Access the RandomForestClassifier step from the pipeline
xgb_classifier = xgb_model.named_steps['xgb_model']

# Get feature importances
xgb_feature_importances = xgb_classifier.feature_importances_

# Combine feature names with their importances
xgb_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_feature_importances
}).sort_values(by='Importance', ascending=False).round(2)

print(xgb_importance_df.head(20))


xgb_predict = xgb_model.predict(x_test)

classification_report(y_test, xgb_predict)

f1_score(y_test, xgb_predict)

confusion_matrix(y_test, xgb_predict)