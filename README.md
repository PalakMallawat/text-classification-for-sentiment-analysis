# text-classification-for-sentiment-analysis

## Dataset: 
Amazon reviews dataset which contains real reviews for jewelry products sold on Amazon. The dataset is downloadable at:
https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz

## Dataset Preparation
We create a three-class classification problem according to the ratings. The original dataset is large. To this end, let ratings with the values of 1 and 2 form class 1, ratings with the value of 3 form class 2, and ratings with the values of 4 and 5 form class 3. To avoid the computational burden, we selected 20,000 random reviews from each rating class and create a balanced dataset to perform the required tasks on the downsized dataset

## Dataset cleaning
Use some data cleaning steps to preprocess the dataset you created. For example:
- converted all reviews into lowercase.
- removed the HTML and URLs from the reviews
- removed non-alphabetical characters
- removed extra spaces
- performed contractions on the reviews, e.g., won’t → will not.

## Preprovessing
Used NLTK package to process the dataset:
- removed the stop words
- performed lemmatization

## Feature Extraction
Used TF-IDF for feature extraction

## Training models
- Perceptron
- SVM
- Logistic regression
- Multinomial Naive Bayes
