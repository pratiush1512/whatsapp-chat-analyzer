# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:35:55 2022

@author: DELL
"""

from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
extract = URLExtract()

## helper function to fetch statistics
def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

## helper function to fetch most busy users
def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

## helper function to create wordcloud
def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

## helper function to find most common words
def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

## helper function to get stats about emojis
def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user']==selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.distinct_emoji_list(message)])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

## function to get monthly timeline
def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline
## helper function to get daily timeline
def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

## helper function to get weekly activity map
def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

## helper function to get monthly activity map
def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

## helper function to get activity heatmap
def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap
## helper function for Sentiment Analysis
def sentiment_analysis(selected_user,df):
    flag = 0
    if selected_user != 'Overall':
        df = df[df['user']==selected_user]
    sentiments=SentimentIntensityAnalyzer()
    df["positive"]=[sentiments.polarity_scores(i)["pos"] for i in df["message"]]
    df["negative"]=[sentiments.polarity_scores(i)["neg"] for i in df["message"]]
    df["neutral"]=[sentiments.polarity_scores(i)["neu"] for i in df["message"]]
    
    x=sum(df["positive"])
    y=sum(df["negative"])
    z=sum(df["neutral"])
    if (x>y) and (x>z):
        flag = 1
    elif (y>x) and (y>z):
        flag = -1
    else:
        flag = 0
    values= [x,y,z]
    label = ['Positive','Negative','Neutral']
    return values,label,flag
            










