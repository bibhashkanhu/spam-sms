import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv("C:\\Users\\bibha\\Downloads\\spam.csv")

# Drop duplicates
data.drop_duplicates(inplace=True)

# Check for null values
print(data.isnull().sum())

# Replace category labels
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Separate features and labels
mess = data['Message']
cat = data['Category']

# Split the data into training and testing sets
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)

# Initialize the CountVectorizer
cv = CountVectorizer(stop_words='english')

# Transform the training data
features = cv.fit_transform(mess_train)

# Create the model
model = MultinomialNB()

# Train the model
model.fit(features, cat_train)

# Test the model
features_test = cv.transform(mess_test)
print(model.score(features_test, cat_test))

# Predict data
message = cv.transform(['WINNER!! As a valued network customer you have been selected to receivea Â£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.'])

result = model.predict(message)  # Pass the transformed message to the predict method
print(result)