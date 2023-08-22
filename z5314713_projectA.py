import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import seaborn as sn
import os
import scipy.interpolate as sci
from math import isnan
import pprint
np.random.seed(1)

assets = ['BHP', 'CSL', 'RIO', 'CBA', 'WOW', 'WES', 'TLS', 'AMC', 'BXB', 'FPH']

def Station1_loadData():
    """
    Process data load post extract and transfer phases
    :return: 3 DataFrames with the cleaned asset, client and economic data
    """

    # please modify the line below to change the file path
    fileDB = ""     # C:/Temp/Datalake/

    def asset_data_station1():
        """
        Import and modify the asset data
        :return: cleaned df with the asset data
        """

        # read in the Excel sheets
        df = pd.read_excel(fileDB + 'ASX200top10.xlsx', sheet_name=1, header=1)

        # set the date column as index
        df.set_index('Dates', inplace=True)

        # rename column headers of the Index columns
        df.rename(columns={df.columns[0]: "index: total return"}, inplace=True)
        df.rename(columns={df.columns[1]: "index: close px"}, inplace=True)

        # replace the column headers of the rest of the columns
        i = -2
        j = 0
        for column in df:
            if i < 0:
                i += 1
                continue
            elif j % 5 == 0:
                df.rename(columns={column: str(assets[i // 5]) + ": " + "total return"}, inplace=True)
            elif j % 5 == 1:
                df.rename(columns={column: str(assets[i // 5]) + ": " + "close px"}, inplace=True)
            elif j % 5 == 2:
                df.rename(columns={column: str(assets[i // 5]) + ": " + "eq weighted px"}, inplace=True)
            elif j % 5 == 3:
                df.rename(columns={column: str(assets[i // 5]) + ": " + "Volume px"}, inplace=True)
            elif j % 5 == 4:
                df.rename(columns={column: str(assets[i // 5]) + ": " + "cur market cap"}, inplace=True)
            i += 1
            j += 1

        # ensure that the dataframe is sorted
        df.sort_index(inplace=True)

        # replace all nan values with fitting values
        i = -2
        for column in df:
            if i == -2 or i % 5 == 0 or i % 5 == 3:
                df[column].fillna(0, inplace=True)
            else:
                df[column].fillna(method='ffill', inplace=True)

        # remove unnecessary columns
        columns_to_delete = []
        for x in range(52):
            if (x - 2) % 5 != 0 and x != 0:
                columns_to_delete.append(x)

        # sort the list reversely to delete the columns one by one
        columns_to_delete.sort(reverse=True)

        # delete all close price, volume, eqt weighted price and market cap columns
        for x in columns_to_delete:
            df.drop(df.columns[x], axis=1, inplace=True)

        return df

    def client_data_station1():
        """
        Import and modify the client data
        :return: cleaned df with the client data
        """

        # read in the Excel sheets
        df_client = pd.read_excel(fileDB + 'Client_Details.xlsx', sheet_name=1)

        # set the date column as index
        df_client.set_index('client_ID', inplace=True)

        # replace the column names of df_client
        for index, column in enumerate(df_client):
            if index == 0 or index == 1:
                continue
            else:
                df_client.rename(columns={column: column[:-10]}, inplace=True)

        # ensure that the dataframe is sorted
        df_client.sort_index(inplace=True)

        # replace all nan values with fitting values
        df_client.dropna(inplace=True)

        # check that all allocations in the client table add up to 100%
        rows_to_drop = []
        for i in range(len(df_client.index)):
            if df_client.iloc[i, 2:].sum() != 1:
                rows_to_drop.append(i)

        rows_to_drop.sort(reverse=True)

        # remove all rows in df where the allocations do not add up to 100%
        for i in rows_to_drop:
            df_client.drop(df_client.index[i])

        return df_client


    def economic_data_station1():
        """
        Import and modify the economic data
        :return: cleaned df with the economic data
        """

        # read in the Excel sheets
        df_eco_mth = pd.read_excel(fileDB + 'Economic_Indicators.xlsx', header=3, nrows=25)
        df_eco_qrt = pd.read_excel(fileDB + 'Economic_Indicators.xlsx', header=30)

        # transpose the economic data DataFrames so that the indicators are the columns of the df
        df_eco_mth = df_eco_mth.transpose()
        df_eco_mth.columns = df_eco_mth.iloc[0]
        df_eco_mth = df_eco_mth.iloc[1:]

        df_eco_qrt = df_eco_qrt.transpose()
        df_eco_qrt.columns = df_eco_qrt.iloc[0]
        df_eco_qrt = df_eco_qrt.iloc[1:]

        # set the date column as index
        df_eco_mth.index = pd.to_datetime(df_eco_mth.index)
        df_eco_mth.rename(index={0: 'Date'}, inplace=True)
        df_eco_qrt.index = pd.to_datetime(df_eco_qrt.index)
        df_eco_qrt.rename(index={0: 'Date'}, inplace=True)

        # concatenate monthly and quarterly economic data
        df_eco = pd.concat([df_eco_mth, df_eco_qrt])

        # sort according to index values
        df_eco.sort_index(inplace=True)

        # remove all rows that have insufficient input values
        df_eco = df_eco.replace('-', np.NaN)
        df_eco = df_eco.dropna(thresh=12)
        df_eco = df_eco.dropna(axis=1, thresh=9)

        # only keep all the columns that provide significant economic data
        df_eco = df_eco[['Retail Sales (%y/y)', 'Unemployment Rate (sa)', 'Average Weekly Earnings (%y/y)',
                         'CPI (%q/q)', 'Real GDP Growth (%q/q, sa)', 'Consumer Spending Growth (%q/q, sa)']]

        df_eco = df_eco.iloc[14:32]

        # take the average monthly data to convert monthly to quarterly data
        i = 0
        averages = [0, 0, 0, 0]

        for idx, row in df_eco.iterrows():
            if row['CPI (%q/q)'] is np.NaN:
                averages[0] += row['Retail Sales (%y/y)']
                averages[1] += 1

                averages[2] += row['Unemployment Rate (sa)']
                averages[3] += 1
            else:
                if isnan(row['Retail Sales (%y/y)']) and averages[1] > 0:
                    df_eco.loc[idx, 'Retail Sales (%y/y)'] = averages[0] / averages[1]
                    averages[0] = 0
                    averages[1] = 0
                if isnan(row['Unemployment Rate (sa)']) and averages[3] > 0:
                    df_eco.loc[idx, 'Unemployment Rate (sa)'] = averages[2] / averages[3]
                    averages[2] = 0
                    averages[3] = 0

        # remove the monthly data, so that only quaterly data persists
        df_eco = df_eco[df_eco['CPI (%q/q)'].notna()]

        # forward fill the missing values
        df_eco['Average Weekly Earnings (%y/y)'].fillna(method='ffill', inplace=True)


        return df_eco


    df = asset_data_station1()
    df_client = client_data_station1()
    df_eco = economic_data_station1()


    return df, df_client, df_eco




def Station2_featuresEngineeringBase(df, df_client, df_eco):
    """
    Receive cleaned data from Station #1 process all relevant features
    :param: 3 DataFrames from Station #1 with asset, client and economic data
    :return: return relevant features as diverse data types, such as dictionaries, Series and DataFrames
    """

    def asset_data_featuresEngineering(df):

        # build the covariance matrix of all investments
        cov_matrix = pd.DataFrame.cov(df)

        # calculate the mean return of all the investments
        mean_return_series = df.mean(axis=0)
        # convert the daily returns to annual returns
        mean_return_series = mean_return_series.apply(lambda x: x * 252)

        print("Mean annual return of assets:")
        print(mean_return_series)

        sn.heatmap(cov_matrix, annot=True, fmt='g')
        plt.show()

        return cov_matrix, mean_return_series


    def client_data_featuresEngineering(df_client):


        # remove age group to avoid multicollinearity beteween age group and risk profile
        df_client.drop('age_group', axis=1, inplace=True)

        # create a risk score for all assets based on the existing client data
        risk_score = {}
        for asset in assets:
            risk_score[asset] = 0

        for index, row in df_client.iterrows():
            i = 1
            for idx in range(len(df_client.columns)):
                if idx == 0: continue
                risk_score[df_client.columns[idx]] += (row[idx] * row[0])

        print("Risk score based on client data:")
        pprint.pprint(risk_score)

        return risk_score


    def eco_data_featuresEngineering(df_eco):


        recession = []
        for idx, row in df_eco.iterrows():
            recession_score = 0
            if row['Retail Sales (%y/y)'] < 0: recession_score += 1
            if row['Unemployment Rate (sa)'] > 7: recession_score += 2
            if row['Average Weekly Earnings (%y/y)'] < 0: recession_score += 1
            if row['CPI (%q/q)'] > 4 or row['CPI (%q/q)'] < -1 : recession_score += 1
            if row['Real GDP Growth (%q/q, sa)'] < 0: recession_score += 2
            if row['Consumer Spending Growth (%q/q, sa)'] < 0: recession_score += 1

            if recession_score >= 4: recession.append(True)
            else: recession.append(False)

        df_eco['Recession'] = recession

        # df_eco.to_excel('Station3.xlsx', sheet_name='new_sheet_name')
        print(df_eco)

        return df_eco



    cov_matrix, mean_return_series = asset_data_featuresEngineering(df)

    risk_score = client_data_featuresEngineering(df_client)

    df_eco = eco_data_featuresEngineering(df_eco)


    return df, cov_matrix, mean_return_series, risk_score, df_eco


def main():
    # Station #1
    df1, df1_client, df1_eco = Station1_loadData()

    # Station #2
    df, cov_matrix, mean_return_series, risk_score, df_eco = Station2_featuresEngineeringBase(df1, df1_client, df1_eco)


if __name__ == '__main__':
    main()


