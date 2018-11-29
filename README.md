# news-classification
The machine learning model to differentiate amongst the category of news, by using headline and short description of news.
For downloading the dataset: https://www.dropbox.com/s/jfjjw66jo3ukuil/News_Category_Dataset.json.zip?dl=0
The dataset contains 125k news headlines and there are 31 different news categories.

Sample Data:
{
    "short_description": "She left her husband. He killed their children. Just another day in America.",
    "headline" : "There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV",
    "date" : "2018-05-26",
    "link": "https://www.huffingtonpost.com/entry/texas-amanda-painter-mass-shooting_us_5b081ab4e4b0802d69caad89"
    "authors": "Melissa Jeltsen",
    "category" : "CRIME"
}

The model used:
1. Multinomial naive Bayes | Accuracy = 34.45%
2. MLP classifier | Accuracy = 50.9%
3. SVM (Support vector machine) | Accuracy = 91.02%
