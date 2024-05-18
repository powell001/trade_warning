import pandas as pd
import os
import sys

os.chdir(r'C:\Users\jpark\VSCode\trade_warning\\')

from config.defintions import ROOT_DIR
print(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data\\')
HELPER_DIR = os.path.join(ROOT_DIR, 'helper\\')

def country_mappings():
  baci_country = pd.read_csv(DATA_DIR + "country_codes_V202301.csv", usecols=['country_code', 'iso_3digit_alpha'])

  ### Make Hong part of China (sorry Hong Kong)
  #baci_country[baci_country['country_code'] == 344] = 156
  #baci_country[baci_country['country_code'] == 344] = 156

  iso_regions = pd.read_csv(DATA_DIR + "iso_countries_regions.csv", usecols=['name', 'alpha-3', 'region_eu'])

  print(iso_regions[iso_regions['region_eu'] == 'EU'])

  iso_regs = baci_country.merge(iso_regions, left_on='iso_3digit_alpha', right_on='alpha-3', how='left')
  iso_regs.drop(columns=['iso_3digit_alpha'], inplace=True)
  #iso_regs.to_csv("tmp3.csv")
  #iso_regs = iso_regs.dropna()

  iso_regs['OECD'] = 'RoW'

  #iso_regs['new'] = np.where(iso_regs['region_eu'] == 'Europe', 'Europe', np.NAN)
  iso_regs.loc[(iso_regs['alpha-3'] == 'JPN', 'OECD')] = 'Japan'
  iso_regs.loc[(iso_regs['alpha-3'] == 'USA', 'OECD')] = 'United States'
  iso_regs.loc[(iso_regs['alpha-3'] == 'CHN', 'OECD')] = 'China, incl. Hong-Kong'
  iso_regs.loc[(iso_regs['alpha-3'] == 'HKG', 'OECD')] = 'China, incl. Hong-Kong'
  iso_regs.loc[(iso_regs['alpha-3'] == 'KOR', 'OECD')] = 'Korea'
  iso_regs.loc[(iso_regs['alpha-3'] == 'TWN', 'OECD')] = 'Chinese Taipei'

  eu = [
    'AUT', 'BEL', 'BGR','HRV', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
    'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT',
    'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE', 'CYP'
  ]

  iso_regs.loc[(iso_regs['alpha-3'].isin(eu), 'OECD')] = 'European Union'


  iso_regs.rename(columns = {'alpha-3': 'ISO3'}, inplace = True)

  return iso_regs[['country_code', 'ISO3', 'region_eu', 'OECD']]


#country_mappings().to_csv("tmp.csv")


def readindata() -> pd.DataFrame:
  df1 = pd.read_csv(DATA_DIR + 'BACI_HS17_Y2021_V202301.csv')
  # rename columns to make them meaningful
  df1.rename(columns={'t': 'Year', 'i': 'Exporter', 'j': 'Importer', 'k': 'Product', 'v': 'Value', 'q': 'Tons'}, inplace=True)
  df1.drop(columns=('Year'),inplace=True)

  ### Make Hong Kong part of China (sorry Hong Kong)
  #df1[df1['Exporter'] == 344] = 156
  #df1[df1['Importer'] == 344] = 156

  # country names from BACI
  baci_countries = pd.read_csv(DATA_DIR + 'tmp.csv', usecols=['country_code', 'ISO3', 'region', 'OECD'])

  # add code for TWN, was 490,"Other Asia, nes","Other Asia, not elsewhere specified",N/A,N/A
  # Add Taiwan directly to Baci Country Codes

  # first merge to add the names of exporters
  df2 = df1.merge(baci_countries, how='left', left_on='Exporter', right_on='country_code')
  df2.rename(columns={"ISO3": "Exporter_ISO3"}, inplace=True)

  # then add the names of importers
  df3 = df2.merge(baci_countries, how='left', left_on='Importer', right_on='country_code')
  df3.rename(columns={"ISO3": "Importer_ISO3"}, inplace=True)

  # drop unnecessary columns
  df4 = df3.drop(columns=['country_code_x', 'country_code_y'])

  # add sub regions for exporters and importers
  df5 = df4.merge(baci_countries[['ISO3', 'region']], how='left', left_on='Exporter_ISO3', right_on='ISO3')
  df5.rename(columns={"region": "Exporter_Region", "OECD_x": "Exporter_OECD"}, inplace=True)

  df6 = df5.merge(baci_countries[['ISO3', 'region']], how='left', left_on='Importer_ISO3', right_on='ISO3')
  df6.rename(columns={"region": "Importer_Region", "OECD_y": "Importer_OECD"}, inplace=True)

  df6.drop(columns=['ISO3_x', 'ISO3_y', 'region_x', 'region_y'], inplace=True)

  df6.to_csv("tmp2.csv")

  return df6

#print(readindata().head())