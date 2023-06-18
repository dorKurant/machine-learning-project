import matplotlib
import openpyxl as openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from prettytable import PrettyTable
import string
import collections
from collections import Counter
import seaborn as sns
import nltk
# nltk.download()
from nltk.tokenize import word_tokenize
from collections import Counter
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
import random
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,davies_bouldin_score
from gower import gower_matrix
from imblearn.over_sampling import SMOTE


# %matplotlib inline


def delete_duplicate(dt):
    dt = dt.astype('string')
    dt = dt.fillna("unknown")
    dt = dt.drop('job_id', axis='columns', inplace=False)
    dt = dt.drop_duplicates()
    dt = dt.drop_duplicates(subset=["company_profile", "description", "requirements"], keep="first")
    return dt


def balance_data(dt):
    dt['fraudulent'] = dt['fraudulent'].astype(int)
    filter_dt = dt[dt['fraudulent'] == 1]  # Filter rows where fraudulent is 1
    filtered_dt_0 = dt[dt['fraudulent'] == 0]  # Filter rows where fraudulent is 0
    unknown_counts = filtered_dt_0.apply(lambda row: row.tolist().count('unknown'), axis=1)
    filter_dt_0 = filtered_dt_0[unknown_counts <= 2]
    return pd.concat([filter_dt, filter_dt_0], axis=0).astype('string').sort_index()  # Concatenate the two filtered DataFrames

def fraud_percentage(td):
    total_count = len(td)
    fraudulent_count = td['fraudulent'].astype(int).sum()
    non_fraudulent_count = total_count - fraudulent_count
    fraud_percentage = (fraudulent_count / total_count) * 100
    non_fraud_percentage = (non_fraudulent_count / total_count) * 100
    print(f"Fraudulent=1 percentage: {fraud_percentage:.2f}%")
    print(f"Fraudulent=0 percentage: {non_fraud_percentage:.2f}%")


def word_fraudulent(df, training_data, attributs1,
                    attributs2):  # A function that returns what percentage of the entire value is fraudulent and how much is not
    training_data[attributs1] = training_data[attributs1].str.lower()
    training_data[attributs1] = training_data[attributs1].fillna("unknown")
    merge_fraudulent = pd.merge(training_data[attributs1], training_data[attributs2], left_index=True, right_index=True)
    for tf in merge_fraudulent[[attributs1, attributs2]].itertuples():
        if tf[2] == '1':
            words = word_tokenize(tf[1])
            for word in words:
                try:
                    df.loc[word, attributs2] += 1
                except:
                    continue
    return df


# a function that getting words as an input.
# shows the number of times a word appears in a job description overall
# and the number of times it appears in a fraudulent job offer
def inserted_words_description(df):
    word_list = input("Enter a comma-separated list of words: ").split(",")
    df['description'] = df['description'].astype('string')
    table = PrettyTable()
    table.field_names = ['The word', 'Number of times the word in description', 'Fraudulent = 1', 'Fraudulent = 0']
    for word in word_list:
        word_rows = df.loc[df['description'].str.contains(word)]
        word_count = (df['description'].str.contains(word)).sum()
        # count how many rows have a "fraudulent" value of 1 or 0
        fraudulent_1_count = word_rows.loc[word_rows['fraudulent'] == 1].shape[0]
        fraudulent_0_count = word_rows.loc[word_rows['fraudulent'] == 0].shape[0]
        table.add_row([word, word_count, fraudulent_1_count, fraudulent_0_count])
    print(table)


def denger_words_analistic(td, attribute, target, frequency_to_show, fraud_ratio_to_show):
    data = td[attribute]
    data = data.astype('string')
    data = data.fillna("unknown")
    tokenized_words = [word_tokenize(text) for text in data]
    all_words = [word.lower() for sublist in tokenized_words for word in sublist]
    word_freq = Counter(all_words)
    word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['frequency'])
    word_freq_df[target] = 0
    word_freq_df = word_fraudulent(word_freq_df, td, attribute, target)
    punctuation_marks = [".", ",", ";", ":", "!", "?", "'", '*', '+', '/', "(", ")", "[", "]", "{", "}", "-", "--",
                         "---", "&"]
    linking_words = ["and", "but", "or", "so", "because", "although", "while", "since", "if", "when", "as", "until",
                     "unless", "than", "for", "with", "to", "be", "is", "that", "a", "may", "just", "only", "both",
                     "either", "neither",
                     "nor", "not", "whether", "yet", "even", "also", "never", "off", "of", "the", "an", "via", "on",
                     "it"]
    word_freq_df = word_freq_df[~word_freq_df.index.isin(punctuation_marks)]
    word_freq_df = word_freq_df[~word_freq_df.index.isin(linking_words)]
    word_freq_df['fraud_ratio'] = word_freq_df[target] / word_freq_df['frequency']
    word_freq_df = word_freq_df.sort_values('fraud_ratio', ascending=False)
    word_freq_df = word_freq_df.loc[word_freq_df['frequency'] > frequency_to_show]
    word_freq_df = word_freq_df.loc[word_freq_df['fraud_ratio'] > fraud_ratio_to_show]
    word_freq_df.index.name = 'word'
    word_freq_df['fraud_ratio_rounded'] = round(word_freq_df['fraud_ratio'], 1)
    return word_freq_df


def analist_generic_word_in_text(td, attribute, target, frequency_to_show,
                                 fraud_ratio_to_show):  # A function that accepts a text features and returns a graph with its distribution against the target variable
    word_freq_df = denger_words_analistic(td, attribute, target, frequency_to_show, fraud_ratio_to_show)
    print(td[attribute].describe())
    plt.figure(figsize=(25, 10))
    p = sns.histplot(data=word_freq_df, x='word', y='frequency', hue='fraud_ratio_rounded', fill=True, palette="tab10",
                     legend='left', element='step')
    sns.move_legend(p, "upper left")
    plt.xticks(rotation=90)
    plt.show()


def sub_word_in_string(td, attribute, target, frequency_to_show, fraud_ratio_to_show):
    word_freq_df = denger_words_analistic(td, attribute, target, frequency_to_show, fraud_ratio_to_show)
    new_feature = 'Sus words in ' + attribute
    td = td.astype('string')
    td = td.fillna("unknown")
    td[new_feature] = 0.0
    row = 0
    for index, row in td.iterrows():
        words = row[attribute].split()
        max_score = 0.0
        for word in words:
            if word in word_freq_df.index:
                if word_freq_df.loc[word, 'fraud_ratio'] > max_score:
                    max_score = word_freq_df.loc[word, 'fraud_ratio']
        td.loc[index, new_feature] = max_score
    print(td.groupby([new_feature, 'fraudulent']).fraudulent.sum())


def create_binary_graph(training_data, features,
                        target):  # Creating a graph for binary variables against the target variable
    new_table = training_data.groupby(features)[
        target].value_counts().unstack()  # create table with all the option and group how many fraudlent and un fraudlent have from every option
    new_table.columns = ['non-fraudulent', 'fraudulent']  # add column names
    new_table['rate'] = new_table['fraudulent'] / (
            new_table['non-fraudulent'] + new_table['fraudulent'])  # add rate column
    print(new_table)  # print the table with the statistic
    ax = new_table[['non-fraudulent', 'fraudulent']].plot(kind='bar', stacked=True)
    for i, r in enumerate(new_table.index):  # adding columns rate to the bar
        ax.text(i - 0.25, new_table.loc[r, 'non-fraudulent'], "{:.2f}".format(new_table.loc[r, 'rate']), fontsize=10)
    plt.show()


def create_categorization_table(td, features, target):
    data = pd.concat([td[features], td[target]], axis=1)
    data = data.astype('string')
    data = data.fillna("unknown")
    data = data.groupby(features)[target].value_counts().unstack()
    data = data.fillna(0)
    data.columns = ['non-fraudulent', 'fraudulent']
    data['rate'] = data['fraudulent'] / (data['non-fraudulent'] + data['fraudulent'])
    return data


def create_categorization_graph(td, features, target, frequency_to_show,
                                fraud_ratio_to_show):  # Creating a graph for categorical variables
    data = create_categorization_table(td, features, target)
    data = data[(data['fraudulent'] + data['non-fraudulent']) > frequency_to_show]
    data = data[data.rate > fraud_ratio_to_show]
    ax = data[['non-fraudulent', 'fraudulent']].plot(kind='bar', stacked=True)
    for i, r in enumerate(data.index):
        if data.loc[r, 'rate'] > fraud_ratio_to_show:
            ax.text(i - 0.25, data.loc[r, 'non-fraudulent'], "{:.2f}".format(data.loc[r, 'rate']), fontsize=8)
    print(data)
    plt.show()


def sub_word_in_string(td,rd,attribute, target, frequency_to_show, fraud_ratio_to_show):
    word_freq_df = denger_words_analistic(td, attribute, target, frequency_to_show, fraud_ratio_to_show)
    new_feature = 'Sus words in ' + attribute
    rd = rd.astype('string')
    rd = rd.fillna("unknown")
    rd[new_feature] = 0.0
    row = 0
    for index, row in rd.iterrows():
        words = row[attribute].split()
        max_score = 0.0
        for word in words:
            if word in word_freq_df.index:
                if word_freq_df.loc[word, 'fraud_ratio'] > max_score:
                    max_score = word_freq_df.loc[word, 'fraud_ratio']
        rd.loc[index, new_feature] = max_score
    # print(td.groupby([new_feature,'fraudulent']).describe())
    # create_categorization_graph(td,new_feature,'fraudulent',0,0.2)
    rd[new_feature] = rd[new_feature].apply(lambda
                                                x: "0-0.1" if x < 0.1 else "0.1-0.25" if x < 0.25 and x >= 0.1 else "0.25 - 0.5" if x >= 0.25 and x < 0.5 else "0.5 - 0.75" if x >= 0.5 and x < 0.75 else "0.75 - 1")
    # create_categorization_graph(td, new_feature, 'fraudulent', 0, 0)
    return rd


def aggregation_to_features(td, feature, target, frequency,rd):
    ag = create_categorization_table(td, feature, target)
    ag['danger'] = ag.rate
    ag['danger'] = ag.apply(lambda x: round(x.danger, 1) if x['non-fraudulent'] + x['fraudulent'] > frequency else 0,
                            axis=1)
    rd['industry and function score'] = rd.apply(lambda x: ag.loc[(ag.index.get_level_values(0) == x['industry']) & (
                ag.index.get_level_values(1) == x['function']), 'danger'].values[0] if (x['industry'], x[
        'function']) in ag.index else 0, axis=1)
    # create_categorization_graph(td, 'industry and function score', 'fraudulent', 0, 0)
    return rd


# making the graph and probabilities for the location, given an attribute (sub-column of location)
def location_graph(df2, attribute):
    fraud_counts = df2.groupby([attribute, 'fraudulent'])[attribute].count().unstack()
    # filter out rows with total count less than 2 and fraudulent count less than 1
    fraud_counts = fraud_counts[(fraud_counts.sum(axis=1) >= 2) & (fraud_counts[1] >= 1)]  # filtering
    if attribute == 'city':
        fraud_counts = fraud_counts[(fraud_counts.sum(axis=1) >= 4) & (fraud_counts[1] > 1)]  # city filtering
    ax = fraud_counts.plot(kind='bar', stacked=True, cmap='Accent')
    plt.xlabel(attribute)
    plt.ylabel('Number of Job Offers')
    plt.title('Fraudulent vs Non-Fraudulent Job Offers by ' + attribute)
    plt.legend(title='Fraudulent', labels=['Non-Fraudulent', 'Fraudulent'])
    for p in ax.patches:
        height = p.get_height()
        width = p.get_width()
        x, y = p.get_xy()
        if y <= 0:
            continue
        ax.text(x + width / 2, y + height / 2, int(height), ha="right", va="baseline")
    plt.show()


def show_fraud_perc_location(df, attribute):  # print attribute with fraudulent in location ratio
    fraud_counts = df.groupby([attribute, 'fraudulent'])[attribute].count().unstack()
    fraud_counts = fraud_counts.fillna(0)
    fraud_counts = fraud_counts[(fraud_counts.sum(axis=1) >= 2) & (fraud_counts['1'] >= 1)]  # filtering
    if attribute == 'city':
        fraud_counts = fraud_counts[(fraud_counts.sum(axis=1) >= 4) & (fraud_counts['1'] > 1)]  # city filtering
    fraud_perc = fraud_counts.apply(lambda x: x * 100 / x.sum(), axis=1)
    fraud_perc = fraud_perc.rename(columns={'0': 'Non-Fraudulent', '1': 'Fraudulent'})
    fraud_perc['Total'] = fraud_counts.sum(axis=1)
    # fraud_perc = fraud_perc['Fraudulent'].fillna(0)
    fraud_perc = fraud_perc.sort_values(by='Fraudulent', ascending=False)
    fraud_perc = fraud_perc[(fraud_perc['Total'] >= 2) & (fraud_perc['Fraudulent'] >= 1)]
    pd.set_option('display.max_rows', None)
    print(fraud_perc)
    return fraud_perc


# splitting the location column into 4 columns
def location_eda(df, show_graphs):
    df2 = df.copy().assign(country='', region='', city='', addit='')
    x = df2['location'].astype('string')
    df2['country'] = df2['country'].astype('string')
    df2['region'] = df2['region'].astype('string')
    df2['city'] = df2['city'].astype('string')
    df2['addit'] = df2['addit'].astype('string')
    index = 0
    x = x.fillna('unknown')
    for i in x:
        if i.count(',') == 0 and i != 'unknown':
            df2.loc[index, 'country'] = i
        if i.count(',') == 1:
            k = i.split(',')
            df2.loc[index, 'country'] = k[0]
            df2.loc[index, 'region'] = k[1]
        if i.count(',') > 1:
            k = i.split(',')
            if len(k) >= 3:
                df2.loc[index, 'country'], df2.loc[index, 'region'], df2.loc[index, 'city'] = k[:3]
            if len(k) > 3:
                df2.loc[index, 'addit'] = ','.join(k[3:])
        elif i == 'unknown':
            df2.loc[index, 'country'] = 'unknown'
            df2.loc[index, 'region'] = 'unknown'
            df2.loc[index, 'city'] = 'unknown'
        index += 1
        if index == len(df2['location']) - 1:
            break
    df2['country'] = df2['country'].fillna('unknown')
    df2['region'] = df2['region'].fillna('unknown')
    df2['city'] = df2['city'].fillna('unknown')
    df2 = df2.fillna("unknown")
    df2 = df2.replace('', "unknown")
    print(df2.country.describe())
    print(df2.region.describe())
    print(df2.city.describe())
    print(df2.addit.describe())
    print('number of samples with additional info: ', df2['addit'].count())
    if show_graphs == True:
        location_graph(df2, 'country')
        location_graph(df2, 'region')
        location_graph(df2, 'city')
        location_graph(df2, 'addit')
    return df2


def get_salary_range(training_data):
    salary_split = training_data.salary_range.astype('string')  # נשנה לסטרינג
    salary_split = salary_split.fillna("unknown")
    salary_split1 = pd.DataFrame(
        {'min': [], 'max': [], 'unknown': [], 'average': []})  # נשנה את הטבלה להיות מינימום מקסימום
    for s in salary_split:
        if s == "unknown":
            salary_split1.loc[len(salary_split1)] = [0, 0, 1, 0]
        elif '-' in s:
            min_salary, max_salary = s.split('-')
            try:
                min_salary = float(min_salary)
                max_salary = float(max_salary)
                salary_split1.loc[len(salary_split1)] = [min_salary, max_salary, 0, (max_salary + min_salary) / 2]
            except:
                salary_split1.loc[len(salary_split1)] = [0, 0, 1, (0 + 0) / 2]
        else:
            salary = float(s)
            salary_split1.loc[len(salary_split1)] = [salary, 0, 0, salary]
    return salary_split1


def salary_range_graph(training_data):
    salary_split = get_salary_range(training_data)
    merged_data = pd.merge(salary_split['average'], training_data.fraudulent, left_index=True, right_index=True)
    counts = merged_data.groupby(['average', 'fraudulent']).size().reset_index(name='count')
    counts = counts[counts.average != 0]
    sns.scatterplot(data=counts, x='count', y='average', hue='fraudulent')
    plt.xlabel('counts')
    plt.show()
    print(merged_data)


def company_profile_graph(training_data):
    print(training_data.company_profile.describe())
    print(training_data.company_profile.value_counts())
    company_profile_fradulent = training_data[['company_profile', 'fraudulent']]
    print(company_profile_fradulent.company_profile.astype(str))
    company_profile_fradulent['words_count'] = company_profile_fradulent['company_profile'].fillna("unknown").apply(
        lambda x: len(x.split()))
    print(company_profile_fradulent['words_count'].value_counts())
    agri = company_profile_fradulent.groupby(['words_count', 'fraudulent']).size().reset_index(name='count')
    print(agri)
    sns.scatterplot(x=agri['words_count'], y=agri['count'], hue=agri['fraudulent'])
    fraudulent_data = agri[agri['fraudulent'] == 1]
    sns.regplot(x='words_count', y='count', data=fraudulent_data, scatter=False, color="orange")
    plt.show()
    print(("company_profile_fradulent"))


def has_company_logo_and_has_questions_and_telecommuting(td):
    features = ['has_company_logo', 'has_questions', 'telecommuting']
    target = 'fraudulent'
    create_binary_graph(td, features, target)


def has_company_logo_and_has_questions(td):
    features = ['has_company_logo', 'has_questions']
    target = 'fraudulent'
    create_binary_graph(td, features, target)


def telecommuting_and_has_questions(td):
    features = ['telecommuting', 'has_questions']
    target = 'fraudulent'
    create_binary_graph(td, features, target)


def has_company_logo_and_telecommuting(td):
    features = ['telecommuting', 'has_company_logo']
    target = 'fraudulent'
    create_binary_graph(td, features, target)


def telecommuting_graph(td):
    features = ['telecommuting']
    target = 'fraudulent'
    create_binary_graph(td, features, target)


def has_company_logo(td):
    features = ['has_company_logo']
    target = 'fraudulent'
    create_binary_graph(td, features, target)


def has_questions(td):
    features = ['has_questions']
    target = 'fraudulent'
    create_binary_graph(td, features, target)


# The graph of the employment type column
def employment_type_graph(df):
    fraud_counts = df.groupby(['employment_type', 'fraudulent'])['job_id'].count().unstack()
    ax = fraud_counts.plot(kind='bar', stacked=True, cmap='coolwarm')
    plt.xlabel('Employment Type')
    plt.ylabel('Number of Job Offers')
    plt.title('Fraudulent vs Non-Fraudulent Job Offers by Employment Type')
    plt.legend(title='Fraudulent', labels=['Non-Fraudulent', 'Fraudulent'])
    for p in ax.patches:
        height = p.get_height()
        width = p.get_width()
        x, y = p.get_xy()
        ax.text(x + width / 2, y + height / 2, int(height), ha="center", va="center")
    plt.show()


def sus_location_feature(new_df, old_df,learning_data):
    new_df[['country', 'region', 'city']] = old_df[['country', 'region', 'city']]
    #learning_data=location_eda(learning_data,False)
    # drop the location column from new_df
    # new_df.drop('location', axis=1, inplace=True)

    # get a list of suspicious countries (fraudulent rate over 20%)
    sus_country_df = show_fraud_perc_location(learning_data, 'country')
    list_sus_country = sus_country_df.loc[sus_country_df['Fraudulent'] > 20].index.tolist()

    # get a list of suspicious regions
    sus_region_df = show_fraud_perc_location(learning_data, 'region')
    list_sus_region = sus_region_df.loc[sus_region_df['Fraudulent'] > 20].index.tolist()

    # get a list of suspicious cities
    sus_city_df = show_fraud_perc_location(learning_data, 'city')
    list_sus_city = sus_city_df.loc[sus_city_df['Fraudulent'] > 20].index.tolist()

    new_df['sus_country'] = new_df['country'].isin(list_sus_country).astype(int)
    new_df['sus_region'] = new_df['region'].isin(list_sus_region).astype(int)
    new_df['sus_city'] = new_df['city'].isin(list_sus_city).astype(int)

    return new_df


# CREATE NEW FEATURES
def n_words_in_string_feature(new_df, df, attribute):
    new_feature = 'n_words_in_' + attribute
    df[attribute] = df[attribute].fillna('unknown')
    df[attribute] = df[attribute].astype('string')

    def count_words(text):
        return len(text.strip().split())

    new_df[new_feature] = df[attribute].apply(count_words)
    new_df[new_feature] = new_df[new_feature].apply(
        lambda x: 'less than 100' if x <= 100 else ('between 100 and 200' if x <= 200 else 'above 200'))

    return new_df


def experince_education_weight_feature(new_df, old_df):
    exp_weights = {'Associate': 1,
                   'Director': 5,
                   'Entry level': 0,
                   'Executive': 4,
                   'Internship': 0,
                   'Mid-Senior level': 3,
                   'Not Applicable': 0,
                   'unknown': 0,
                   np.nan: 0}
    edu_weights = {'Associate Degree': 2,
                   "Bachelor's Degree": 3,
                   'Certification': 1,
                   'Doctorate': 5,
                   'High School or equivalent': 0,
                   "Master's Degree": 4,
                   'Professional': 4,
                   'Some College Coursework Completed': 1,
                   'Some High School Coursework': 0,
                   'Unspecified': 0,
                   'Vocational': 1,
                   'Vocational - Degree': 2,
                   'Vocational - HS Diploma': 1,
                   'unknown': 0,
                   np.nan: 0}

    new_df['exp_weight'] = old_df['required_experience'].apply(lambda x: exp_weights[x])
    new_df['edu_weight'] = old_df['required_education'].apply(lambda x: edu_weights[x])
    new_df['edu_exp_mismatch'] = abs(new_df['exp_weight'] - new_df['edu_weight'])
    return new_df


def calc_fischer_score(df, feature):
    contingency_table = pd.crosstab(df[feature], df['fraudulent'])
    chi2, p, dof, ex = chi2_contingency(contingency_table, correction=False)
    fischer_score = chi2 / (contingency_table.sum().min() * (contingency_table.shape[0] - 1))
    print('Fischer score for {}: {}'.format(feature, fischer_score))


def correlation_matrix(training_data):
    training_data.reset_index()
    training_data.replace(0, np.nan, inplace=True)
    training_data.dropna(axis=1, how='all', inplace=True)
    # Specify the features you want to include in the correlation matrix
    features = ['telecommuting', 'has_company_logo', 'has_questions', 'sus_country', 'sus_region', 'sus_city',
                'n_words_in_company_profile', 'n_words_in_description',
                'n_words_in_requirements', 'exp_weight', 'edu_weight', 'edu_exp_mismatch', 'Sus words in title',
                'Sus words in description',
                'Sus words in department', 'Sus words in benefits', 'Sus words in requirements',
                'industry and function score', 'fraudulent']
    # Select only the specified features
    td = training_data[features]
    td['telecommuting'] = td['telecommuting'].astype('category').cat.codes
    td['has_company_logo'] = td['has_company_logo'].astype('category').cat.codes
    td['has_questions'] = td['has_questions'].astype('category').cat.codes
    td['sus_country'] = td['sus_country'].astype('category').cat.codes
    td['sus_region'] = td['sus_region'].astype('category').cat.codes
    td['sus_city'] = td['sus_city'].astype('category').cat.codes
    td['n_words_in_description'] = td['n_words_in_description'].astype('category').cat.codes
    td['n_words_in_requirements'] = td['n_words_in_requirements'].astype('category').cat.codes
    td['exp_weight'] = td['exp_weight'].astype('category').cat.codes
    td['exp_weight'] = td['exp_weight'].astype('category').cat.codes
    td['edu_exp_mismatch'] = td['edu_exp_mismatch'].astype('category').cat.codes
    td['Sus words in title'] = td['Sus words in title'].astype('category').cat.codes
    td['Sus words in description'] = td['Sus words in description'].astype('category').cat.codes
    td['Sus words in department'] = td['Sus words in department'].astype('category').cat.codes
    td['Sus words in benefits'] = td['Sus words in benefits'].astype('category').cat.codes
    td['Sus words in requirements'] = td['Sus words in requirements'].astype('category').cat.codes
    td['industry and function score'] = td['industry and function score'].astype('category').cat.codes
    td['fraudulent'] = td['fraudulent'].astype('category').cat.codes
    corr_matrix = td.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

#------------------------------- PART 2 --------------------------
def ordinally_categorical(df):
    sus_words_order = {"0-0.1": '1', "0.1-0.25": '2', "0.25 - 0.5": '3', "0.5 - 0.75": '4', "0.75 - 1": '5'}
    df["Sus words in title"] = df["Sus words in title"].replace(sus_words_order)
    df["Sus words in description"] = df["Sus words in description"].replace(sus_words_order)
    df["Sus words in department"] = df["Sus words in department"].replace(sus_words_order)
    df["Sus words in benefits"] = df["Sus words in benefits"].replace(sus_words_order)
    df["Sus words in requirements"] = df["Sus words in requirements"].replace(sus_words_order)
    scale_mapper = {"less than 100": '1', "between 100 and 200": '2', "above 200": '3'}
    df["n_words_in_company_profile"] = df["n_words_in_company_profile"].replace(scale_mapper)
    df["n_words_in_description"] = df["n_words_in_description"].replace(scale_mapper)
    df["n_words_in_requirements"] = df["n_words_in_requirements"].replace(scale_mapper)
    df = df.astype(float)
    return df

#------------------------Decision Tree----------------------------------

def holdout(td, features, label, x_train, x_test, y_train, y_test): #Holdout function for Decision Tree
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=123)
    max_depth_list = np.arange(1, 50, 1)  # לבדוק אם יש בעיית זמן ריצה
    res = pd.DataFrame()

    for max_depth_check in max_depth_list:
        model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth_check, random_state=42)
        model.fit(x_train, y_train)
        res = res.append(
            {'max_depth': max_depth_check, 'train_acc': roc_auc_score(y_train, model.predict_proba(x_train)[:, 1]),
             'val_acc': roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])}, ignore_index=True)
    # plt.figure(figsize=(13,4))
    #   plt.plot(res['max_depth'],res['train_acc'],marker='o',markersize=4)
    #  plt.plot(res['max_depth'],res['val_acc'],marker='o',markersize=4)
    #  plt.legend(['Train accuracy','Validation accuracy'])
    # plt.show()
    res = res.sort_values('val_acc', ascending=False)
    return res.iloc[0][0]

def kfold(x_train, y_train): #Kfold function for decision tree
    max_depth_list = np.arange(1, 20, 1)  # לבדוק אם יש בעיית זמן ריצה
    kfold = KFold(n_splits=5, shuffle=True, random_state=123)
    res = pd.DataFrame()
    #features.reset_index(drop=True, inplace=True)  # Reset the index of the features DataFrame
    for train_idx, val_idx in kfold.split(x_train):
        for max_depth in max_depth_list:
            model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=123)
            # x=features.iloc[train_idx]
            model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
            train_acc = roc_auc_score(y_train.iloc[train_idx], model.predict_proba(x_train.iloc[train_idx])[:, 1])
            val_acc = roc_auc_score(y_train.iloc[val_idx], model.predict_proba(x_train.iloc[val_idx])[:, 1])
            # res=res.append({'max_depth':max_depth,'acc':acc},ignore_index=True)
            res = pd.concat([res,
                             pd.DataFrame({'max_depth': [max_depth], 'val_acc': [val_acc], 'train_acc': [train_acc]},
                                          index=[0])], ignore_index=True)
    res = res.sort_values('val_acc', ascending=False)
    print(res[['max_depth', 'val_acc', 'train_acc']].groupby(['max_depth']).mean().reset_index().sort_values('val_acc',
                                                                                                             ascending=False))
    return res.iloc[0][0]

def grid_search_k_fold(x_train, x_test, y_train, y_test): #grid search k fol for decision Tree
    classifier = DecisionTreeClassifier(random_state=123)
    param_grid = {'max_depth': np.arange(1, 20, 1),
                  'criterion': ['gini', 'entropy'],
                  'max_features': ['auto','sqrt', 'log2', None],
                   'min_samples_split': [2, 5, 10],
                   'splitter': ['best', 'random']
                  }
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=123),
                               param_grid=param_grid,
                               refit=True,
                               scoring='roc_auc',
                               cv=10)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    print(best_model)
    preds = best_model.predict_proba(x_test)[:, 1]
    test_auc = roc_auc_score(y_test, preds)  # Calculate AUC-ROC score
    print("test auc accuracy:", round(test_auc, 5))

def print_tree(x_train, y_train, depth):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=123)
    model.fit(x_train, y_train)
    plt.figure(figsize=(40, 40))
    plot_tree(model, filled=True, max_depth=3, class_names=['0', '1'], feature_names=x_train.columns, fontsize=6)
    plt.show()
    print(f"AUC-ROC: {roc_auc_score(y_true=y_train, y_score=model.predict_proba(x_train)[:, 1]):.2f}")
    importance_scores = model.feature_importances_
    importance_dict = dict(zip(x_train.columns, importance_scores))
    filtered_importances = {feature: importance for feature, importance in importance_dict.items() if importance > 0}
    sorted_importances = sorted(filtered_importances.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance}")

#-----------------ANN , NLP CLASSIFIER-----------------------
def ann(x_train,y_train,x_test,y_test, random_state=123, layers = (1500,1300), activation='relu', learning_rate_init=0.01, max_iter = 500): #basic ann for
    model = MLPClassifier(random_state=random_state,
                          max_iter=max_iter,
                          hidden_layer_sizes=layers,
                          activation=activation,
                          learning_rate_init=learning_rate_init)
    model.fit(x_train, y_train)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_test = y_test.astype(int)
    print(f"Accuracy for train: {roc_auc_score(y_true=y_train, y_score=model.predict_proba(x_train)[:, 1]):.3f}")
    print(f"Accuracy for test: {roc_auc_score(y_true=y_test, y_score=y_pred_proba):.3f}")
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_pred = y_pred.reshape(-1,1)
    cf_matrix = confusion_matrix(y_true = y_test,y_pred= y_pred)
    print(cf_matrix)


#Checking ann with multy hiden layers and small amount nuirons
def annMultylayers(x_train,y_train,x_test,y_test, random_state=123, activation='relu', learning_rate_init=0.001, max_iter = 500):
    num_hidden_layers = 2
    hidden_layer_sizes = [1500] * num_hidden_layers
    model = MLPClassifier(random_state=random_state,
                          max_iter=max_iter,
                          hidden_layer_sizes=hidden_layer_sizes,
                          activation=activation,
                          learning_rate_init=learning_rate_init)
    model.fit(x_train, y_train)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_test = y_test.astype(int)
    print(f"Accuracy for train: {roc_auc_score(y_true=y_train, y_score=model.predict_proba(x_train)[:, 1]):.3f}")
    print(f"Accuracy for test: {roc_auc_score(y_true=y_test, y_score=y_pred_proba):.3f}")
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_pred = y_pred.reshape(-1,1)
    #print(confusion_matrix(y_true=y_test, y_pred=model.predict(x_test)))
    cf_matrix = confusion_matrix(y_true = y_test,y_pred= y_pred)
    print(cf_matrix)

def layer_sizes_generator():#generator layers for ann
    num_layers = random.randint(1,4)
    layer_sizes = tuple(random.randint(20, 80) for _ in range(num_layers))
    return layer_sizes

def ann_hyperParameters_generate(x_train,y_train,x_test,y_test): #generator for different size of layers and nuirons
    #train_accs=[]
    #test_accs=[]
    combinations = set()
    res = pd.DataFrame()
    for size_ in range(1,50,1):
        layers = layer_sizes_generator()
        if layers not in combinations:
            combinations.add(layers)
        else:
            continue
        model = MLPClassifier(random_state=1,
                                 max_iter=100,
                                 hidden_layer_sizes=layers,
                                 activation='relu',
                                 learning_rate_init=0.001,
                                 alpha=0.0001)
        model.fit(x_train, y_train)
        y_test = y_test.astype(int)
        train_acc=roc_auc_score(y_true=y_train, y_score=model.predict_proba(x_train)[:, 1])
        test_acc=roc_auc_score(y_true=y_test, y_score=model.predict_proba(x_test)[:, 1])
        res = pd.concat([res,
                         pd.DataFrame({'size_': [layers], 'train_acc': [train_acc], 'test_acc': [test_acc]},
                                          index=[0])], ignore_index=True)
    res = res.sort_values('test_acc', ascending=False)
    print(res)

def ann_hyperParameters(x_train,y_train,x_test,y_test): # Hyper parameters grid search for ann
    from sklearn.metrics import classification_report
    mlp= MLPClassifier(max_iter=100)
    parameter_space = {
        'hidden_layer_sizes': [(50,50),(20,30),],
        'activation': ['logistic', 'relu'],
        #'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        #'learning_rate': ['constant', 'adaptive'],
    }
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=2)
    clf.fit(x_train, y_train)
    print('Best parameters found:\n', clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print('Results on the test set:')
    print(classification_report(y_test,clf.predict_proba(x_test)[:, 1]))

def ann_hyperParameters_4_layers(x_train,y_train,x_test,y_test): #hyper paramteters by n^4 Run time, lopX4
    #train_accs=[]
    #test_accs=[]
    res_relu = pd.DataFrame()
    res_logistic=pd.DataFrame()
    for size_ in range(20,120,30):
        for size_2 in range(20,120,30):
            for size_3 in range (20,120,30):
                for size_4 in range(20, 120, 30):

                    model = MLPClassifier(random_state=1,
                                          max_iter=100,
                                          hidden_layer_sizes=(size_,size_2,size_3,size_4),
                                          activation='relu',
                                          learning_rate_init=0.001,
                                          alpha=0.0001)
                    model.fit(x_train, y_train)
                    #y_test = y_test.astype(int)
                    train_acc=roc_auc_score(y_true=y_train, y_score=model.predict_proba(x_train)[:, 1])
                    test_acc=roc_auc_score(y_true=y_test, y_score=model.predict_proba(x_test)[:, 1])
                    res_relu = pd.concat([res_relu,
                                     pd.DataFrame({'size_': [size_],'size_2':[size_2],'size_3':[size_3],'size_4':[size_4], 'train_acc': [train_acc], 'test_acc': [test_acc]},
                                                  index=[0])], ignore_index=True)

                    #----------------------we will try logistic-------------------------

                    model2 = MLPClassifier(random_state=1,
                                          max_iter=100,
                                          hidden_layer_sizes=(size_, size_2, size_3,size_4),
                                          activation='logistic',
                                          learning_rate_init=0.001,
                                          alpha=0.0001)
                    model2.fit(x_train, y_train)
                    # y_test = y_test.astype(int)
                    train_acc = roc_auc_score(y_true=y_train, y_score=model2.predict_proba(x_train)[:, 1])
                    test_acc = roc_auc_score(y_true=y_test, y_score=model2.predict_proba(x_test)[:, 1])
                    res_logistic = pd.concat([res_logistic,
                                     pd.DataFrame({'size_': [size_], 'size_2': [size_2], 'size_3': [size_3],'size_4':[size_4],
                                                   'train_acc': [train_acc], 'test_acc': [test_acc]},
                                                  index=[0])], ignore_index=True)


    res_relu = res_relu.sort_values('test_acc', ascending=False)
    res_logistic = res_logistic.sort_values('test_acc', ascending=False)
    print(res_relu)

#-------------------------------SVM---------------------------------

def SVM_pca(x_train,y_train,x_test,y_test,c=0.005,dumantion=2): # SVM wich pca
    pca=PCA(n_components=dumantion,random_state=123)
    pca1=pca.fit_transform(x_train)
    pca2=pca.fit_transform(x_test)
    model = SVC(probability=True, kernel='linear',C=c,random_state=123)
    model.fit(pca1,y_train)
    print("intercept:",model.intercept_)
    print("the coef is:" ,model.coef_)
    model_train=model.predict_proba(pca1)[:, 1]
    model_test=model.predict_proba(pca2)[:, 1]
    print("trainig ruc auc score :",roc_auc_score(y_train,model_train))
    print("test ruc auc score :",roc_auc_score(y_test,model_test))
    print("SSR:",pca.explained_variance_ratio_)
    print("SSR_SUM:",pca.explained_variance_ratio_.sum())
    # print the model:
    plt.figure(figsize=(10, 10))
    plt.title("predictions")
    y_pred = (model_train >= 0.5).astype(int)
    sns.scatterplot(x=pca1[:,0],y=pca1[:,1],hue=y_pred)
    plt.show()
    print(5)

def SVM(x_train,y_train,x_test,y_test,c=0.005): #generic SVM for diffult data
    model = SVC(probability=True, kernel='linear',C=c)
    model.fit(x_train,y_train)
    print("intercept:",model.intercept_)
    print("the coef is:" ,model.coef_)
    model_train=model.predict_proba(x_train)[:, 1]
    model_test=model.predict_proba(x_test)[:, 1]
    print("training roc auc score :",roc_auc_score(y_train,model_train))
    print("test roc auc score :",roc_auc_score(y_test,model_test))

def SVM_hyperParameters(x_train,y_train,x_test,y_test): #hyperparameters for SVM
    model=SVC(probability=True,kernel = 'linear')#,C=0.005)
    parameters_grid = {
            "C":[0.005,0.05,0.1,1.0,2]
             }
    grid=GridSearchCV(model,parameters_grid,refit=True,
                               scoring='roc_auc',
                               cv=10)
    grid.fit(x_train, y_train)
    print("best parameters:",grid.best_params_)

#------------------------ K MEANS ---------------------------------
def k_means_gower(x_train,n_cluster=2):
    gower_dist = gower_matrix(x_train) #we firt calculate the distance betwen the features
    pca=PCA(n_components=2, random_state=123)
    training_data=pca.fit_transform(gower_dist)
    training_data = pd.DataFrame(training_data, columns=['feature1', 'feature2'])
    kmeans=KMeans(n_clusters=n_cluster,init='k-means++',random_state=123)
    kmeans.fit(training_data)
    training_data['cluster'] = kmeans.predict(training_data)
    sns.scatterplot(x='feature1', y='feature2', hue='cluster', data=training_data)
    plt.show()

def k_mean_clusters_check_by_graph(x_train):
    iner_list=[]
    dbi_list=[]
    sil_list=[]
    pca=PCA(n_components=2,random_state=123)
    x_train =pca.fit_transform(x_train)
    for n in tqdm(range(2,10,1)):
        kmeans=KMeans(n_clusters=n,max_iter=100,n_init=10,random_state=123)
        kmeans.fit(x_train)
        assignment=kmeans.predict(x_train)
        iner=kmeans.inertia_
        sil=silhouette_score(x_train,assignment,random_state=123)
        dbi=davies_bouldin_score(x_train,assignment)
        dbi_list.append(dbi)
        sil_list.append(sil)
        iner_list.append(iner)

    plt.plot(range(2,10,1),iner_list,marker='o')
    plt.title("intertia")
    plt.xlabel("number of clusters")
    plt.show()

    plt.plot(range(2, 10, 1), sil_list, marker='o')
    plt.title("silhouette")
    plt.xlabel("number of clusters")
    plt.show()

    plt.plot(range(2, 10, 1), dbi_list, marker='o')
    plt.title("Davies-bouldin")
    plt.xlabel("number of clusters")
    plt.show()

#-------------------- IMPROVE THE MODEL ------------------------------

def smote_over_sampling(x_train, y_train):
    unique, count = np.unique(y_train, return_counts=True)
    y_train_dict_value_count = { k:v for (k,v) in zip(unique, count)}
    print("proportion before over-sampling: ", y_train_dict_value_count)
    sm = SMOTE(random_state=42, sampling_strategy=0.5)
    x_train_res, y_train_res = sm.fit_resample(x_train,y_train)
    unique, count = np.unique(y_train_res, return_counts=True)
    y_train_smote_value_count = {k: v for (k, v) in zip(unique, count)}
    print("proportion after over-sampling: ", y_train_smote_value_count)
    return x_train_res, y_train_res

def ann_improved(x_train,y_train,x_test,y_test, random_state=123, layers = (1500,1300), activation='relu', learning_rate_init=0.01, max_iter = 500):
    model = MLPClassifier(random_state=random_state,
                          max_iter=max_iter,
                          hidden_layer_sizes=layers,
                          activation=activation,
                          learning_rate_init=learning_rate_init)
    model.fit(x_train, y_train)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_test = y_test.astype(int)
    print(f"Accuracy for train: {roc_auc_score(y_true=y_train, y_score=model.predict_proba(x_train)[:, 1]):.3f}")
    print(f"Accuracy for test: {roc_auc_score(y_true=y_test, y_score=y_pred_proba):.3f}")
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_pred = y_pred.reshape(-1,1)
    cf_matrix = confusion_matrix(y_true = y_test,y_pred= y_pred)
    print(cf_matrix)

def ann_one_layer(x_train,y_train,x_test,y_test):
    res = pd.DataFrame()
    for size_ in range(1,100,1):
        model = MLPClassifier(random_state=1,
                                 max_iter=100,
                                 hidden_layer_sizes=size_,
                                 activation='relu',
                                 learning_rate_init=0.001,
                                 alpha=0.0001)
        model.fit(x_train, y_train)
        y_test = y_test.astype(int)
        train_acc=roc_auc_score(y_true=y_train, y_score=model.predict_proba(x_train)[:, 1])
        test_acc=roc_auc_score(y_true=y_test, y_score=model.predict_proba(x_test)[:, 1])
        res = pd.concat([res,
                         pd.DataFrame({'size_': [size_], 'train_acc': [train_acc], 'test_acc': [test_acc]},
                                          index=[0])], ignore_index=True)
    res = res.sort_values('test_acc', ascending=False)
    print(res)

def ann_final_testing(td,rd, random_state=123, layers = (1500,1300), activation='relu', learning_rate_init=0.01, max_iter = 500):
    td_x = td.drop('fraudulent', axis=1)
    td_y = td['fraudulent']
    td_features = set(td_x.columns)
    rd_features = set(rd.columns)
    drop_columns_test = rd_features - td_features
    drop_columns_train = td_features - rd_features
    rd = rd.drop(drop_columns_test, axis = 1)
    td_x = td_x.drop(drop_columns_train, axis = 1)
    model = MLPClassifier(random_state=random_state,
                          max_iter=max_iter,
                          hidden_layer_sizes=layers,
                          activation=activation,
                          learning_rate_init=learning_rate_init)
    model.fit(td_x, td_y)
    y_pred_proba = model.predict_proba(rd)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    y_pred = y_pred.reshape(-1,1)
    return y_pred





# ------------------------MAIN-------------------------------------------------------------------------------------------


if __name__ == '__main__':
    #----------------------------READING THE DATA ---------------------------------------
    training_data = pd.read_csv('XY_train.csv')
    test_data=pd.read_csv('X_test.csv')

    # ---------------------------NEW TRAINING AND TEST DATA WITH NEW FEATURES------------------------

    #training_data = pd.read_csv('output3.csv')
    training_data = training_data.astype('string')
    training_data = training_data.fillna('unknown')
    test_data=test_data.astype('string')
    test_data=test_data.fillna('unknown')

    # ----------------------------- PART A, FEATURE GRAPH -----------------------------------------
    # --- title:
    # analist_generic_word_in_text(td=training_data,attribute= 'title',target= 'fraudulent',frequency_to_show=6,fraud_ratio_to_show=0.2)

    # --- location:
    # location_eda(training_data)

    # --- department:
    # analist_generic_word_in_text(td=training_data, attribute='department', target='fraudulent', frequency_to_show=5,fraud_ratio_to_show=0.05)

    # --- salary_range:
    # salary_range_graph(training_data)

    # --- company_profile:
    # company_profile_graph(training_data)

    # --- description:
    # analist_generic_word_in_text(td=training_data, attribute='description', target='fraudulent', frequency_to_show=60, fraud_ratio_to_show=0.25)
    # list_of_words_in_string(training_data,'description')
    # inserted_words_description(training_data) #graph_words_in_string(description_frequent_words, 'description')

    # --- requirements:
    # analist_generic_word_in_text(td=training_data, attribute='requirements', target='fraudulent', frequency_to_show=12,fraud_ratio_to_show=0.25)

    # --- benefits:
    # benefits_frequent_words = list_of_words_in_string(training_data,'benefits')
    # graph_words_in_string(training_data,benefits_frequent_words, 'benefits')

    # --- telecommuting:
    # telecommuting_graph(training_data)

    # --- has_company_logo:
    # has_company_logo(training_data)

    # --- has_questions:
    # has_questions(training_data)

    # --- has_company_logo_&_has_questions_&_telecommuting:
    # has_company_logo_and_has_questions(training_data)
    # has_company_logo_and_has_questions_and_telecommuting(training_data)
    # telecommuting_and_has_questions(training_data)
    # has_company_logo_and_telecommuting

    # --- employment_type:
    # employment_type_graph(training_data) #data for employment_type
    # create_categorization_graph(training_data, ['employment_type', 'required_education', 'required_experience'], 'fraudulent',6,0.05)

    # --- required_experience:
    # create_categorization_graph(training_data,['required_experience','required_education'],'fraudulent',10,0.05)

    # --- required_education AND industry:
    # create_categorization_graph(training_data,['industry','function'],'fraudulent',10,0.05)

    # ----------------------------- NEW FEATURES -----------------------------------------
    # Feature Extraction:

    new_df = training_data.copy()
    new_df = delete_duplicate(new_df)
    new_df = balance_data(new_df)
    fraud_percentage(new_df)
    new_df=new_df.reset_index()
    new_df = new_df.drop('index', axis='columns', inplace=False)
    new_df= sus_location_feature(new_df=new_df,old_df=location_eda(new_df, False),learning_data=location_eda(new_df, False)) #suspicious location
    new_df = n_words_in_string_feature(new_df, new_df, 'company_profile')
    new_df = n_words_in_string_feature(new_df, new_df, 'description')
    new_df = n_words_in_string_feature(new_df, new_df, 'requirements')
    new_df = experince_education_weight_feature(new_df, new_df)
    new_df=sub_word_in_string(td=new_df,rd=new_df,attribute= 'title',target= 'fraudulent',frequency_to_show=4,fraud_ratio_to_show=0)
    new_df=sub_word_in_string(td=new_df,rd=new_df,attribute= 'description',target= 'fraudulent',frequency_to_show=6,fraud_ratio_to_show=0)
    new_df=sub_word_in_string(td=new_df,rd=new_df,attribute= 'department',target= 'fraudulent',frequency_to_show=5,fraud_ratio_to_show=0)
    new_df=sub_word_in_string(td=new_df,rd=new_df,attribute= 'benefits',target= 'fraudulent',frequency_to_show=5,fraud_ratio_to_show=0)
    new_df=sub_word_in_string(td=new_df,rd=new_df,attribute= 'requirements',target= 'fraudulent',frequency_to_show=5,fraud_ratio_to_show=0)
    new_df=aggregation_to_features(new_df, ['industry','function'], 'fraudulent',frequency=2,rd=new_df)
    #new_df.to_excel('output3.xlsx', index=True)
    print(5)

    # ------------------------------ TEST  FEATURES----------------------------------------
    test_data= sus_location_feature(new_df=test_data,old_df=location_eda(test_data, False),learning_data=location_eda(new_df, False)) #suspicious location
    test_data = n_words_in_string_feature(test_data, test_data, 'company_profile')
    test_data = n_words_in_string_feature(test_data, test_data, 'description')
    test_data = n_words_in_string_feature(test_data, test_data, 'requirements')
    test_data = experince_education_weight_feature(test_data, test_data)
    test_data=sub_word_in_string(td=new_df,rd=test_data,attribute= 'title',target= 'fraudulent',frequency_to_show=4,fraud_ratio_to_show=0)##
    test_data=sub_word_in_string(td=new_df,rd=test_data,attribute= 'description',target= 'fraudulent',frequency_to_show=6,fraud_ratio_to_show=0)
    test_data=sub_word_in_string(td=new_df,rd=test_data,attribute= 'department',target= 'fraudulent',frequency_to_show=5,fraud_ratio_to_show=0)
    test_data=sub_word_in_string(td=new_df,rd=test_data,attribute= 'benefits',target= 'fraudulent',frequency_to_show=5,fraud_ratio_to_show=0)
    test_data=sub_word_in_string(td=new_df,rd=test_data,attribute= 'requirements',target= 'fraudulent',frequency_to_show=5,fraud_ratio_to_show=0)
    test_data=aggregation_to_features(new_df, ['industry','function'], 'fraudulent',frequency=2,rd=test_data)


    # ----------------------------- FEATURE SELECTION -----------------------------------------
    '''
    training_data=new_df
    correlation_matrix(training_data)

    textual_features = ['job_id', 'title', 'location', 'department', 'salary_range', 'company_profile', 'description', 'requirements', 'benefits']
    for feature in training_data.columns[:]:
        if feature == 'fraudulent':
            continue
        calc_fischer_score(training_data, feature)
    '''

    #-----------------------------------PART 2----------------------------------------------
    #-------- FEATURES INCODING,CATEGORICAL COODING, STANDARTIZETION AND SCALING------------
    #-------- TRAINING DATA:

    textual_features = ['title', 'location', 'department', 'salary_range', 'company_profile', 'description',
                        'requirements', 'benefits', 'telecommuting', 'has_company_logo', 'has_questions']
    df_numeric = new_df[['sus_country', 'sus_region', 'sus_city', 'exp_weight', 'edu_weight', 'edu_exp_mismatch',
                                'industry and function score', 'fraudulent']]
    numeric_ordinally = ['Sus words in title', 'Sus words in description', 'Sus words in department',
                         'Sus words in benefits', 'Sus words in requirements', 'n_words_in_company_profile',
                         'n_words_in_description', 'n_words_in_requirements']
    df_numeric_ordinally = new_df[numeric_ordinally]
    df_numeric_ordinally = ordinally_categorical(df_numeric_ordinally) # change the categorical to numbers
    df_numeric = df_numeric.astype(float)
    df_categorical = new_df.drop(textual_features, axis=1)
    df_categorical = df_categorical.drop(df_numeric, axis=1)
    df_categorical = df_categorical.drop(numeric_ordinally, axis=1)
    df_categorical = pd.get_dummies(df_categorical)
    label = df_numeric['fraudulent']
    df_numeric = df_numeric.drop(['fraudulent'], axis=1)
    df_united = pd.merge(df_categorical, df_numeric, left_index=True, right_index=True)
    df_united = pd.merge(df_united, df_numeric_ordinally, left_index=True, right_index=True)
    scaler = StandardScaler()
    feature_names = df_united.columns
    df_united_scaled = scaler.fit_transform(df_united)
    df_united_scaled = pd.DataFrame(df_united_scaled, columns=feature_names)
    df_combine = pd.merge(df_united_scaled, label, left_index=True, right_index=True)
    df_combine = df_combine.drop_duplicates()
    df_united_scaled1 = df_combine.drop(['fraudulent'], axis = 1)
    label1 = df_combine['fraudulent']
    x_train, x_test, y_train, y_test = train_test_split(df_united_scaled1, label1, test_size=0.15, random_state=123)
    # df_combined is the training data we use

    #---------TEST DATA:
    jobid = test_data['job_id']
    test_data.drop(training_data.columns[0], axis=1, inplace=True)
    textual_features = ['title', 'location', 'department', 'salary_range', 'company_profile', 'description',
                        'requirements', 'benefits', 'telecommuting', 'has_company_logo', 'has_questions']
    df_numeric = test_data[['sus_country', 'sus_region', 'sus_city', 'exp_weight', 'edu_weight', 'edu_exp_mismatch',
                                'industry and function score']]
    numeric_ordinally = ['Sus words in title', 'Sus words in description', 'Sus words in department',
                         'Sus words in benefits', 'Sus words in requirements', 'n_words_in_company_profile',
                         'n_words_in_description', 'n_words_in_requirements']
    rd_numeric_ordinally = test_data[numeric_ordinally]
    rd_numeric_ordinally = ordinally_categorical(rd_numeric_ordinally)  # change the categorical to numbers
    rd_numeric = df_numeric.astype(float)
    rd_categorical = test_data.drop(textual_features, axis=1)
    rd_categorical = rd_categorical.drop(df_numeric, axis=1)
    rd_categorical = rd_categorical.drop(numeric_ordinally, axis=1)
    rd_categorical = pd.get_dummies(rd_categorical)
    rd_united = pd.merge(rd_categorical, rd_numeric, left_index=True, right_index=True)
    rd_united = pd.merge(rd_united, rd_numeric_ordinally, left_index=True, right_index=True)
    scaler = StandardScaler()
    feature_names = rd_united.columns
    rd_united_scaled = scaler.fit_transform(rd_united)
    ready_rd = pd.DataFrame(rd_united_scaled, columns=feature_names)
    #df_combined is the training data we use
    #ready_rd is the test data we use

    y_pred = ann_final_testing(df_combine, ready_rd)
    res = pd.DataFrame({'job_id': test_data['job_id'], 'fraudulent': y_pred})
    #res.to_excel('G30_ytest.csv', index=True)

    # ----------- DECISION TREE ------------------------


    #optimal_depth =int( holdout(training_data, df_united, label)) #DONT USE IN THE PROJECT
    grid_search_k_fold(x_train, x_test, y_train, y_test)
    optimal_depth = int(kfold(x_train, y_train))
    jobid = jobid.reset_index(drop=True)
    y_pred = ann_final_testing(df_combine, ready_rd)
    y_pred = y_pred.squeeze()
    res = pd.DataFrame({'job_id': jobid, 'fraudulent': y_pred})
    res.to_excel('G30_ytest.xlsx', index=False)

    # ----------- ANN ------------------------

    ann(x_train, y_train, x_test, y_test)
    annMultylayers(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    ann(x_train, y_train, x_test, y_test, random_state=123, layers=(110,80,80,20), activation='logistic', learning_rate_init=0.001, max_iter=100)
    ann_hyperParameters(x_train, y_train, x_test, y_test)

    # --------SVM------------------
    SVM_hyperParameters(x_train, y_train, x_test, y_test)
    SVM(x_train,y_train,x_test,y_test,c=0.05)
    SVM_pca(x_train,y_train,x_test,y_test,c=0.05)

    # --------SVM------------------
    SVM_hyperParameters(x_train, y_train, x_test, y_test)
    SVM(x_train,y_train,x_test,y_test,c=0.05)
    SVM_pca(x_train,y_train,x_test,y_test,c=0.05)

    # ------------K-means----------
    df_united = pd.merge(df_united_scaled, label, left_index=True, right_index=True) #לא צריך אותו
    k_means_gower(x_train,n_cluster=3)
    k_mean_clusters_check_by_graph(x_train)


    # --------------------- OVERSAMPLING--------------------
    #x_train_smote, y_train_smote = smote_over_sampling(x_train, y_train)
    #ann_improved(x_train_smote, y_train_smote, x_test, y_test)
    #ann_hyperParameters_generate(x_train_smote, y_train_smote, x_test, y_test)


