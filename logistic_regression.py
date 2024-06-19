# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
data = pd.read_csv(url)

# Counting the number of survivors (1) and non-survivors (0)
survived_counts = data['Survived'].value_counts()

print(survived_counts)
# Filling missing values
data['Age'].fillna(data['Age'].median(), inplace=True)

# Data preprocessing
data = data.dropna(subset=['Survived', 'Pclass', 'Sex'])  # Drop rows with missing target or features
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Convert categorical variable to numeric

# Selecting features and target
X = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

# Initializing the Logistic Regression model
model = LogisticRegression(class_weight='balanced') 
#balance the dataset cz we have: 
# Survived
# 0    545
# 1    342
# Thus we have to balance our data,
# this augmented the accuracy by approximatly 1%

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plotting the results
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
