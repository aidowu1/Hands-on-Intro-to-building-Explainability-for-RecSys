import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pathlib

# working_dir = pathlib.Path(os.getcwd()).parent.parent.parent
# os.chdir(working_dir)

import Code.DataAccessLayer.Src.Constants as c


class EDAHelpers(object):
    """
    Helper utils for doing EDA on the Movielens dataset
    Reference: Kaggle Notebook by Amardeep Chauhan
    URL: https://www.kaggle.com/code/amar09/eda-recommendation-model-on-movielens-100k
    """
    @staticmethod
    def generateCommonGenreMovies(movies_df: pd.DataFrame) -> None:
        """
        Generates plot of common genre movies
        :param movies_df: Movies dataset
        """
        genres = c.MOVIELENS_GENRE_COLUMNS
        genere_counts = movies_df.loc[:, genres].sum().sort_values(ascending=False)
        sns.barplot(x=genere_counts.index, y=genere_counts.values)
        plt.xticks(rotation=60)

    @staticmethod
    def generateMoviesReleaseProfile(movies_df: pd.DataFrame) -> None:
        """
        Generates movies release profile
        :param movies_df: Movies dataset
        """
        plt.figure(figsize=(12, 7))
        yearly_release_counts = movies_df.groupby(pd.to_datetime(movies_df.release_date).dt.year).size().sort_values(ascending=False)
        sns.lineplot(yearly_release_counts.index, yearly_release_counts.values);
        plt.xlabel('Release Year')

    @staticmethod
    def findAgeGroupsThatWatchMoreMovies(users_df: pd.DataFrame) -> None:
        """
        Find age group that watch more movies
        :param users_df: User dataset
        """
        EDAHelpers.addAgeGroupToUsers(users_df)
        plt.figure(figsize=(9, 6))
        sns.barplot(users_df.groupby('age_group').size().index, users_df.groupby('age_group').size().values)
        plt.title('movie watchers age_group wise')

    @staticmethod
    def addAgeGroupToUsers(users_df: pd.DataFrame):
        """
        Adds the age group to the Users dataset
        :param users_df: User dataset
        """
        users_df['age_group'] = users_df.age.apply(lambda age: 'Gradeschooler' if 5 <= age <= 12 else (
            'Teenager' if 13 <= age <= 19 else (
                'Young' if 20 <= age <= 35 else ('Midlife' if 35 <= age <= 55 else 'Old'))))

    @staticmethod
    def findWhatOccupationsWatchesMoreMovies(users_df: pd.DataFrame) -> None:
        """
        Find what occupation watches more movies
        :param users_df: Users dataset
        """
        plt.figure(figsize=(12, 7))
        movie_watcher_occupants = users_df.groupby('occupation').size().sort_values(ascending=False)
        sns.barplot(movie_watcher_occupants.index, movie_watcher_occupants.values)
        plt.title('movie watchers age_group wise')
        plt.xticks(rotation=50)

    @staticmethod
    def findDistributionOfGenderPerGenre(
            movies_df: pd.DataFrame,
            users_df: pd.DataFrame,
            rating_df: pd.DataFrame
    ) -> None:
        """
        Find the distribution of gender per genre
        :param movies_df: Movies dataset
        :param users_df: Users dataset
        :param rating_df: Rating dataset
        """
        genres = c.MOVIELENS_GENRE_COLUMNS
        rating_user_df = rating_df.join(other=users_df, how='inner', on=c.MOVIELENS_USER_ID_COLUMN_KEY, lsuffix='_R')
        rating_user_movie_df = rating_user_df.join(other=movies_df, how='inner',
                                                   on=c.MOVIELENS_ITEM_ID_COLUMN_KEY, rsuffix='_M')
        temp_df = rating_user_movie_df.groupby('gender').sum().loc[:, genres]
        temp_df = temp_df.transpose()
        plt.figure(figsize=(12, 6))
        temp_df.M.sort_values(ascending=False).plot(kind='bar', color='teal', label="Male")
        temp_df.F.sort_values(ascending=False).plot(kind='bar', color='black', label="Fe-Male")
        plt.legend()
        plt.xticks(rotation=60)
        plt.show()

    @staticmethod
    def findDistributionOfRatingPerGender(
            users_df: pd.DataFrame,
            rating_df: pd.DataFrame
    ) -> None:
        """
        Find the distribution of gender per genre
        :param users_df: Users dataset
        :param rating_df: Rating dataset
        """
        rating_user_df = rating_df.join(other=users_df, how='inner', on=c.MOVIELENS_USER_ID_COLUMN_KEY, lsuffix='_R')
        temp_df = rating_user_df.groupby(['gender', 'rating']).size()
        plt.figure(figsize=(10, 5))
        m_temp_df = temp_df.M.sort_values(ascending=False)
        f_temp_df = temp_df.F.sort_values(ascending=False)

        plt.bar(x=m_temp_df.index, height=m_temp_df.values, label="Male", align="edge", width=0.3, color='teal')
        plt.bar(x=f_temp_df.index, height=f_temp_df.values, label="Female", width=0.3, color='black')
        plt.title('Ratings given by Male/Female Viewers')
        plt.legend()
        plt.xlabel('Ratings')
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def plotHorizontalMovieBar(
            movie_titles: pd.Series,
            ratings_count: int,
            title: str =''):
        """
        Plot Horizontal movie bar
        :param movie_titles: Movie titles
        :param ratings_count: Ratings count
        :param title: Title
        """
        plt.figure(figsize=(12, 7))
        sns.barplot(y=movie_titles, x=ratings_count, orient='h')
        plt.title(title)
        plt.ylabel('Movies')
        plt.xlabel('Count')
        plt.show()

    @staticmethod
    def plotTopTenRatedMovies(
            rating_df: pd.DataFrame,
            movies_df: pd.DataFrame) -> None:
        """
        Plots the top 10 rated movies
        :param rating_df: Ratings dataset
        :param movies_df: Movies dataset
        """
        rating_movie_df = rating_df.join(other=movies_df, how='inner', on=c.MOVIELENS_ITEM_ID_COLUMN_KEY, rsuffix='_M')
        top_ten_rated_movies = rating_movie_df.groupby(c.MOVIELENS_ITEM_ID_COLUMN_KEY).size().sort_values(ascending=False)[:10]
        top_ten_movie_titles = movies_df.iloc[top_ten_rated_movies.index].movie_title
        EDAHelpers.plotHorizontalMovieBar(
            top_ten_movie_titles.values,
            top_ten_rated_movies.values,
            'Top 10 watched movies')

    @staticmethod
    def plotTopTenRatedMoviesPerGender(
            rating_df: pd.DataFrame,
            users_df: pd.DataFrame,
            movies_df: pd.DataFrame) -> None:
        """
        Plots the top 10 rated movies per gender
        :param rating_df: Ratings dataset
        :param users_df: Users dataset
        :param movies_df: Movies dataset
        """
        gender_map = {
            "F": "Female",
            "M": "Male"
        }
        rating_user_df = rating_df.join(other=users_df, how='inner', on=c.MOVIELENS_USER_ID_COLUMN_KEY, lsuffix='_R')
        rating_user_movie_df = rating_user_df.join(other=movies_df, how='inner',
                                                   on=c.MOVIELENS_ITEM_ID_COLUMN_KEY, rsuffix='_M')
        top_rated_movies_gender_wise = rating_user_movie_df.groupby(['gender', c.MOVIELENS_ITEM_ID_COLUMN_KEY]).size()

        for index_label in top_rated_movies_gender_wise.index.get_level_values(0).unique():
            top_10_userkind_rated_movies = top_rated_movies_gender_wise[index_label].sort_values(ascending=False)[:10]
            top_10_userkind_rated_movie_titles = movies_df.iloc[top_10_userkind_rated_movies.index].movie_title
            EDAHelpers.plotHorizontalMovieBar(top_10_userkind_rated_movie_titles.values, top_10_userkind_rated_movies.values,
                                      f'Top 10 {gender_map[index_label]} watched movies')

    @staticmethod
    def plotTopTenRatedMoviesPerAgeGroup(
            rating_df: pd.DataFrame,
            users_df: pd.DataFrame,
            movies_df: pd.DataFrame) -> None:
        """
        Plots the top 10 rated movies per age group
        :param rating_df: Ratings dataset
        :param users_df: Users dataset
        :param movies_df: Movies dataset
        """
        EDAHelpers.addAgeGroupToUsers(users_df)
        rating_user_df = rating_df.join(other=users_df, how='inner', on=c.MOVIELENS_USER_ID_COLUMN_KEY, lsuffix='_R')
        rating_user_movie_df = rating_user_df.join(other=movies_df, how='inner',
                                                   on=c.MOVIELENS_ITEM_ID_COLUMN_KEY, rsuffix='_M')
        top_rated_movies_age_group_wise = rating_user_movie_df.groupby(['age_group', c.MOVIELENS_ITEM_ID_COLUMN_KEY]).size()

        for index_label in top_rated_movies_age_group_wise.index.get_level_values(0).unique():
            top_10_userkind_rated_movies = top_rated_movies_age_group_wise[index_label].sort_values(ascending=False)[
                                           :10]
            top_10_userkind_rated_movie_titles = movies_df.iloc[top_10_userkind_rated_movies.index].movie_title
            EDAHelpers.plotHorizontalMovieBar(top_10_userkind_rated_movie_titles.values, top_10_userkind_rated_movies.values,
                                      f'Top 10 {index_label} watched movies')