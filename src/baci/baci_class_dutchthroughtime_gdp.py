import pandas as pd
import os
import sys 
import numpy as np
from functools import reduce
import itertools
os.chdir(r'C:\Users\jpark\VSCode\trade_warning\\')

from combine_country_regions import country_mappings
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import statsmodels.api as sm
from statsmodels.formula.api import ols

import warnings
warnings.filterwarnings("ignore")

# from config.defintions import ROOT_DIR
# DATA_DIR = os.path.join(ROOT_DIR, 'data\\')
# FIGURES_DIR = os.path.join(ROOT_DIR, 'data\\output\\forfigures\\')

desired_width=1000
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)
pd.options.display.max_rows = 50

# QUESTIONS

# Dutch vulnerablity through time
# Vulnerability is based on the exporting country of a country
    # From which countries does the Netherlands import strategic goods?
    # What is the total value of Dutch imports of strategic goods?, has that volumn changed through time?


class baci:
    def readindata(self, bacidata, verbose = False, tmp_save = True) -> pd.DataFrame:
        df1 = pd.read_csv(bacidata, usecols=['t','i','j','k','v','q'], 
                          dtype= {'t': 'int64',
                                  'i': 'int64', 
                                  'j': 'int64', 
                                  'k': 'object',
                                  'v': 'float64',
                                  'q': 'object'}
                          )

        # This is too complicated, but '   NA' should be converted to float
        df1['q'] = df1['q'].apply(lambda x: x.strip())
        df1['q'].replace('NA', np.NaN, inplace=True)
        df1['q'] = df1['q'].astype(float)

        #dimensions
        #print("shape", df1.shape)

        # rename columns to make them meaningful
        df1.rename(columns={'t': 'Year', 'i': 'Exporter', 'j': 'Importer', 'k': 'Product', 'v': 'Value', 'q': 'Quantity'}, inplace=True)

        # replace number with name of country exporter
        iso1 = pd.read_csv(r"src\baci\data\country_codes_V202401.csv", usecols=['country_code', 'country_iso3'])
        df1 = df1.merge(iso1, left_on="Exporter", right_on="country_code", how="left")
        df1.drop(columns=['country_code', 'Exporter'], inplace = True)
        df1.rename(columns={"country_iso3": "Exporter"}, inplace=True)
    
        # replace number with name of country importer
        df1 = df1.merge(iso1, left_on="Importer", right_on="country_code", how="left")
        df1.drop(columns=['country_code', 'Importer'], inplace = True)
        df1.rename(columns={"country_iso3": "Importer"}, inplace=True)

        if verbose:
            hcodes = [str(x)[0:2] for x in df1["Product"]]
            print(set(hcodes))
            print(len(set(hcodes)))

        return df1
    
    def addprodcode(self, data):
        # add product_codes
        prodcodes = pd.read_csv(r"src\baci\data\product_codes_HS92_V202401.csv", usecols=['code', 'description'])
        mask = prodcodes['code'] == '9999AA'
        prodcodes = prodcodes[~mask]
        #prodcodes['code'] = prodcodes['code'].astype('int64')
        data = data.merge(prodcodes, left_on = "Product", right_on = "code", how = "left")
        
        return data
       
    def subsetData(self, data_param: pd.DataFrame(), iso3_param: list[str], imp_exp_param: str, products_param: list[str], minvalue_param=0.0) -> pd.DataFrame():
        df1 = data_param.copy()
        if products_param:
            out1 = df1[(df1[imp_exp_param].isin(iso3_param)) & (df1["Product"].isin(products_param))]
            out1.sort_values(by=['Value', 'Importer'], inplace=True, ascending=False)
            out1 = out1[out1['Value'] >= minvalue_param]
            out1['Value'] = out1['Value'].astype(int)
        else: # return all products
            out1 = df1[df1[imp_exp_param].isin(iso3_param)]
            out1.sort_values(by=['Value', 'Importer'], inplace=True, ascending=False)
            out1 = out1[out1['Value'] >= minvalue_param]
            out1['Value'] = out1['Value'].astype(int)

        return out1
    
    def subsetStrategicGoods(self, data, strategicProducts: list):
        df1 = data.copy()
        df2 = df1[df1['Product'].isin(strategicProducts)]
        return df2
    
    def addregions(self, data):
        regions = pd.read_csv("data\\iso_countries_regions.csv")
       
        # Exporter
        exp = data.merge(regions, left_on="Exporter", right_on="alpha-3", how="left")
        exp.drop(columns=['alpha-2', 'alpha-3', 'country-code', 'name', 'iso_3166-2', 'intermediate-region', 'region-code', 'sub-region-code', 'intermediate-region-code', 'Unnamed: 12'], inplace = True)
        exp.rename(columns={"region": "Exporter_region", "region_eu": "Exporter_region_eu", "sub-region": "Exporter_sub_region"}, inplace = True)
        # Importer
        imp = exp.merge(regions, left_on="Importer", right_on="alpha-3", how="left")
        imp.drop(columns=['alpha-2', 'alpha-3', 'country-code', 'name', 'iso_3166-2', 'intermediate-region', 'region-code', 'sub-region-code', 'intermediate-region-code', 'Unnamed: 12'], inplace = True)
        imp.rename(columns={"region": "Importer_region", "region_eu": "Importer_region_eu", "sub-region": "Importer_sub_region"}, inplace = True)
        data = imp.copy()

        return data

    def addregion(self, data, exim):
        if exim == "Exporter":
            iso_regions = pd.read_csv(r"src\baci\data\iso_countries_regions.csv")
            iso_regions = iso_regions[['alpha-3', 'region']]
            data = data.merge(iso_regions, left_on="Exporter", right_on="alpha-3", how="left")
            data.rename(columns = {'region': 'Exporter_Region'}, inplace = True)
            data.drop(columns=["alpha-3"], inplace=True)
        elif exim == "Importer":
            iso_regions = pd.read_csv(r"src\baci\data\iso_countries_regions.csv")
            iso_regions = iso_regions[['alpha-3', 'region']]
            data = data.merge(iso_regions, left_on="Importer", right_on="alpha-3", how="left")
            data.rename(columns = {'region': 'Importer_Region'}, inplace = True)
            data.drop(columns=["alpha-3"], inplace=True)
        else: 
            print("Error")

        return data

    def addshortdescriptoProdname(self, data):
        localdata = data.copy()
        prod_h6 = pd.read_csv(r"src\baci\data\hs6twodigits.csv", dtype = str)

        # this is necessary because codes 1:9 should be 01:09
        localdata.loc[:, 'code'] = ["0" + x if len(x) == 5 else x for x in localdata['Product'].astype(str)]

        localdata['shrtDescription'] = localdata['code'].astype(str).str[0:2]
        proddesc = localdata.merge(prod_h6, left_on="shrtDescription", right_on="code")
        proddesc.drop(columns = {'code_x', 'shrtDescription', 'code_y'}, inplace = True)

        proddesc.rename(columns = {"product": "code"}, inplace = True)

        return proddesc
    
    def addlongdescription(self, data):
        localdata = data.copy()
        longdesc = pd.read_csv(r"src\baci\data\product_codes_HS92_V202401.csv", dtype = str)
        longdesc.rename(columns = {"code": "isocode"}, inplace=True)
        longproddesc = localdata.merge(longdesc, left_on="Product", right_on="isocode", how = 'left', suffixes = ['x', 'y'])
       
        r1 = localdata.shape[0]
        r2 = longproddesc.shape[0]
        assert r1 == r2

        return longproddesc
    
    def add_gdp(self, data, GDP, year):

        ### join GDP to data
        
        # Exporters
        gdp = GDP[GDP.index == year]
        gdp = gdp.T
        gdp['Exporter_gdp'] = gdp.index
        
        gdp.rename(columns={year: year + "_gdp_Exporter"}, inplace=True)

        dataj = data.merge(gdp, left_on = "Exporter", right_on = "Exporter_gdp")
        dataj[year + '_gdp_Exporter'] = dataj[year + '_gdp_Exporter']/1e+6
        
        # Importers
        gdp = GDP[GDP.index == year]
        gdp = gdp.T
        gdp['Importer_gdp'] = gdp.index
        gdp.rename(columns={year: year + '_gdp_Importer'}, inplace=True)

        data = dataj.merge(gdp, left_on = "Importer", right_on = "Importer_gdp")
       
        data.drop(columns = ["Exporter_gdp", "Importer_gdp"], inplace=True)

        return data
         
    def valueacrossallcountries(self, data_param: pd.DataFrame()):
        ### Relative size of Step1 inputs per product
        g = data_param[['Product', 'Value']].groupby(['Product']).sum()
        valueofStep1products = g.apply(lambda x: x.sort_values(ascending=False))
        valueofStep1products['Percentage'] = 100 * (valueofStep1products / valueofStep1products['Value'].sum())

        print(valueofStep1products)

        return valueofStep1products

    def valuepercountryacrossprods(self, data_param, imp_exp_param):
        ### Relative size of Step1 inputs per exporter
        g = data_param[[imp_exp_param, 'Value']].groupby([imp_exp_param]).sum()
        valueofStep1perExporter = g.apply(lambda x: x.sort_values(ascending=False))
        valueofStep1perExporter['Percentage'] = 100 * (valueofStep1perExporter / valueofStep1perExporter['Value'].sum())

        print(valueofStep1perExporter)
        return valueofStep1perExporter

    def valueperprod(self, data_param, imp_exp_param):

        exp1 = data_param[['Value', imp_exp_param, 'Product']]
        g = exp1.groupby([imp_exp_param, 'Product']).sum().reset_index()  #this is now a data frame

        allprods = []
        for p in g['Product'].unique():
            prod = g[g['Product'] == p]
            prod.sort_values(by = ['Value'], ascending=False, inplace=True)
            allprods.append(prod)

        print(pd.concat(allprods))

        return pd.concat(allprods)

    def OECD_agg(self, data_param, baci_countries_param, imp_exp_param):

        print(imp_exp_param)

        assert (imp_exp_param == 'Exporter_ISO3') or (imp_exp_param == 'Importer_ISO3'), "needs to be Exporter_ISO3 or Importer_ISO3"

        grp_perCountry = data_param[['Value', imp_exp_param]].groupby([imp_exp_param]).sum().reset_index()
        merged1 = grp_perCountry.merge(baci_countries_param[['ISO3', 'OECD']], left_on=imp_exp_param, right_on="ISO3",
                                   how="left")

        out = merged1[[imp_exp_param, 'Value', 'OECD']].groupby(['OECD']).sum().reset_index()
        out['Percentage'] = 100 * (out['Value'] / out['Value'].sum())
        out.sort_values(['Percentage'], ascending=False,inplace=True)
        print(out)

        return out

    def strategicgoodExportingImportingregions(self, data, impexp: str):
        
        if impexp == 'Importer':
            data = bc1.addregion(data, exim='Importer')
            regionValue = data[['Value', 'Importer_Region']].groupby('Importer_Region').sum()
            print("Major strategic importing regions: ", regionValue.sort_values(['Value'], ascending = False))

            stateValue = data[['Value', 'Importer']].groupby('Importer').sum()
            print("Major stratefic importing states: ", stateValue.sort_values(['Value'], ascending = False))
        
        if impexp == 'Exporter':
            data = bc1.addregion(data, exim='Exporter')
            regionValue = data[['Value', 'Exporter_Region']].groupby('Exporter_Region').sum()
            print("Major strategic exporting regions: ", regionValue.sort_values(['Value'], ascending = False))

            stateValue = data[['Value', 'Exporter']].groupby('Exporter').sum()
            print("Major strategic exporting states: ", stateValue.sort_values(['Value'], ascending = False))

    def typesofstrategicgoods(self, data):
        valuestrategicgood = data[['Value', 'code']].groupby("code").sum()
        strategicproducts = valuestrategicgood.sort_values(['Value'], ascending = False)
    
        HS6codestrategic = set([str(x)[0:2] for x in data["Product"]])
            
        return strategicproducts

def GDPData():
    # https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?end=2022&start=1960&view=chart
    # Taiwan comes from IMF data, added by hand. https://www.imf.org/external/datamapper/NGDPD@WEO/OEMDC/ADVEC/WEOWORLD
    data = pd.read_csv(r"data\\global_gdp.csv", index_col=[1], skiprows=4)
    data = data.drop(columns=['Country Name', 'Indicator Code', 'Indicator Name'])
    data = data.T
    return data
GDP = GDPData()

def getStrategicGoods():
    data = pd.read_csv(r"src\\pdf_extractor\\strategicProducts.csv", index_col=[0],
        dtype= {'0': 'object'}
        )
    data = data.iloc[:,0].tolist()
    return data
STRATEGOODS = getStrategicGoods()

# ININTIALIZE object
bc1 = baci()

# from which regions to the Dutch get, value of imports per region 
def dutchimportsregionthroughtime():
    yearsData = np.arange(1995, 2023, step=1)
    yearsdata = []
    for yr in yearsData:
        data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y"+ str(yr) + "_V202401.csv")
        data = bc1.addregions(data)
        data = bc1.subsetStrategicGoods(data, STRATEGOODS)
        data = data[data['Importer'] == "NLD"]
        data = data[['Value', 'Exporter_sub_region']]
        gp1 = data[['Value', 'Exporter_sub_region']].groupby(['Exporter_sub_region']).sum()
        gp1['Year'] = yr
        gp1['subregion'] = gp1.index

        gp1 = gp1.pivot(index = 'Year', columns = 'subregion', values = 'Value')
        yearsdata.append(gp1)

    return yearsdata

# dt1 = dutchimportsregionthroughtime()
# dt1 = pd.concat(dt1)
# print(dt1)
# dt1.to_csv("sumperregion.csv")

data = pd.read_csv("sumperregion.csv", index_col=[0])
data.drop(columns=["Micronesia", "Melanesia", "Polynesia"], inplace=True)
# np.log(data).plot()
# plt.show()

def compareDutchimportperregion(data):
    import random
    lastyear = []
    for i in range(0, data.shape[1]):
        end_value = np.log(data.iloc[:,i].values[-1]).round(2)
        begin_value = np.log(data.iloc[:,i].values[0]).round(2)
        colname = data.columns[i]
        lastyear.append(pd.DataFrame({'Value': [end_value], 'Col': [colname]}))

        if colname == 'Northern America':
            name_column = data.columns[i]
            plt.plot(data.index, np.log(data.iloc[:,i]), label = name_column)
            plt.text(2022, end_value + .005*end_value, name_column + " " + str(end_value), fontsize = 6)
            plt.title("Logged origin of Dutch imports of strategic goods, excluding Micronesia, Melanisia, Polynesia")
        elif colname == 'Eastern Europe': 
            name_column = data.columns[i]
            plt.plot(data.index, np.log(data.iloc[:,i]), label = name_column)
            plt.text(2022, end_value - .005*end_value, name_column + " " + str(end_value), fontsize = 6)
        else:
            name_column = data.columns[i]
            plt.plot(data.index, np.log(data.iloc[:,i]), label = name_column)
            plt.text(2022, end_value, name_column + " " + str(end_value), fontsize = 6)
        
    t1 = pd.concat(lastyear)
    print(t1.sort_values("Value"))
    plt.show()

# data = pd.read_csv("sumperregion.csv", index_col=[0])
# data.drop(columns=["Micronesia", "Melanesia", "Polynesia"], inplace=True)
# compareDutchimportperregion(data)

# relative to other countries, how much have imports of strategic goods increased through time.  To compare with other countrie, remove the effects of gdp
# first, get top trader importers in terms of total trade
# Or, just look at the trade of strategic goods as a proportion of total imports or gdp   

# for netherlads, group by subcategories, see though time
def dutchbreakdown():
    yearsData = np.arange(1995, 2023, step=1)
    yearsdata = []
    for yr in yearsData:
        print(yr)
        data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y"+ str(yr) + "_V202401.csv")
        data = bc1.addregions(data)
        data = bc1.subsetStrategicGoods(data, STRATEGOODS)
        data = bc1.addshortdescriptoProdname(data)
        data = data[data['Importer'] == "NLD"]
        
        data = data[['Value', 'code']]
        grp1 = data.groupby(['code']).sum()
        grp = grp1.T
        grp['Year'] = yr
        yearsdata.append(grp)
    
    return yearsdata        

# dutchbycode = dutchbreakdown()
# dutchbycode = pd.concat(dutchbycode)
# dutchbycode.set_index("Year", inplace = True)
# dutchbycode.to_csv("dutchbycode.csv")
# np.log(dutchbycode).plot()
# plt.show()

def plotdutchbycode():

    plotable = pd.read_csv("dutchbycode.csv", index_col=[0])
    plotable['AverageAllProducts']  = plotable.mean(axis=1)
    print(plotable)

    n = plotable.shape[1]
    color = iter(cm.twilight_shifted(np.linspace(0, 1, n)))
    for i in range(n):
        c = next(color)
        namecolumn = plotable.columns[i]
        if namecolumn == 'AverageAllProducts':
                c = "black"
                lw = 2.0
                ls = '--'
        else:
                lw = 1.5
                ls ='-'

        x = plotable.index.to_list()
        y = np.log(plotable.iloc[:, i])
        plt.plot(x,y,c=c, linewidth = lw, linestyle = ls, label = namecolumn)
        plt.text(2022+.5, y.iloc[-1], namecolumn + " " + str(y.iloc[-1])[0:4], fontsize = 6)

    plt.legend(loc="upper left")  
    plt.title('Log Value per Product Code per Year')
    plt.show()

def toptraders2022():
    dt1 = bc1.readindata("C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y2022_V202401.csv")
    dt1 = dt1[['Value', 'Importer']]
    dt2 = dt1.groupby(['Importer']).sum()
    dt3 = dt2.sort_values(['Value'], ascending=False)
    
    return dt3.index
toptraders = toptraders2022().tolist()

def slowmethodtogetregressiondata():
    yearsData = np.arange(1995, 2023, step=1)
    Country = toptraders[0:19]
    allCountries = []
    for st in Country:
        print(st)
        allyearsdata = []
        for yr in yearsData:
            data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y"+ str(yr) + "_V202401.csv")
            data = bc1.subsetStrategicGoods(data, STRATEGOODS)
            data = bc1.add_gdp(data, GDP, str(yr))
            data = data[data['Importer'] == st]    
            gdp = data[str(yr) + '_gdp_Importer'].to_list()[0]
            data = data[['Year', 'Value']]
            data.rename(columns = {'Value': 'SumStratGoods'}, inplace = True)
            stratgood = data[['SumStratGoods']].sum().tolist()[0]
        
            allyearsdata.append([st, yr, stratgood, gdp])

        df = pd.DataFrame(allyearsdata)
        df.to_csv('regressionData.csv', mode='a', index=False, header=False )

def regressiondata():
    yearsData = np.arange(1995, 2023, step=1)
    allyears = []
    Country = toptraders[0:110] 
    for yr in yearsData:
        print(yr)
        data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y"+ str(yr) + "_V202401.csv")
        
        ######################
        # total imports
        ######################
        totalimports = data[data['Importer'].isin(Country)]
        totalimports_sum = totalimports[['Value', 'Importer']].groupby(['Importer']).sum()
        totalimports_sum.rename(columns = {'Value': 'sumImports'}, inplace=True)

        ######################
        # strategic
        ######################
        strategic = bc1.subsetStrategicGoods(data, STRATEGOODS)
        strategic = bc1.add_gdp(strategic, GDP, str(yr))
        strategic = strategic[strategic['Importer'].isin(Country)]
        gdpsub = strategic[['Importer', str(yr) + '_gdp_Importer']]
        gdpsub.rename(columns = {str(yr) + '_gdp_Importer': 'GDP'}, inplace = True)
        gdpsub.drop_duplicates(inplace = True)
        sumddata = strategic[['Value', 'Importer']].groupby(['Importer']).sum()
        sumdata = sumddata.merge(gdpsub, left_on = "Importer", right_on = "Importer")
        sumdata.rename(columns = {"Value": "sumStratGoods"}, inplace = True)

        ######################
        # join
        ######################
        alldata = sumdata.merge(totalimports_sum, left_on = "Importer", right_on = "Importer")

        # add year
        alldata['Year'] = yr

        # orginal values
        alldata['sumImports'] = alldata['sumImports']*1e+3
        alldata['sumStratGoods'] = alldata['sumStratGoods']*1e+3 

        # divisions
        alldata['percentStratofImports'] = (alldata['sumStratGoods']/alldata['sumImports'])*100
        alldata['percentStratofGDP'] = (alldata['sumStratGoods']/alldata['GDP'])*100

        allyears.append(alldata)

    return allyears

# data = regressiondata()
# regressiondata = pd.concat(data)
# regressiondata.sort_values(['Importer', 'Year'], inplace=True)
# regressiondata.to_csv("regressionData.csv", index = False)

def augmentregressiondata():
    data = pd.read_csv("regressionData.csv", index_col=[0])
    data.sort_values(['Importer', 'Year'], inplace=True)
    data['state'] = data.index
    data['Yearidx'] = data['Year']
    data.set_index("Yearidx", inplace=True, drop = True)

    # first percent of total imports
    meanallcountries = data[['Year', 'percentStratofImports']].groupby("Year").mean()
    meanallcountries.rename(columns = {'percentStratofImports': 'allstateAverageStrat'}, inplace=True)

    meanallcountries = data[['Year', 'percentStratofGDP']].groupby("Year").mean()
    meanallcountries.rename(columns = {'percentStratofGDP': 'percentStratofGDP'}, inplace=True)

    # for each state, subtract average
    recollectalldata = []
    for st in data['state'].unique():

        # averages across all states
        meanallcountriesImports = data[['Year', 'percentStratofImports']].groupby("Year").mean().values
        meanallcountriesGDP = data[['Year', 'percentStratofGDP']].groupby("Year").mean().values
            
        stdata = data[data['state'] == st]
        if stdata.shape[0] < 28:
            print("Missing years of data: ", st)
            continue
        
        stdata['difffrommeanStratofImports'] = stdata[['percentStratofImports']].values - meanallcountriesImports
        stdata['difffrommeanStratofGDP'] = stdata[['percentStratofGDP']].values - meanallcountriesGDP
        recollectalldata.append(stdata)

    newdata = pd.concat(recollectalldata)
    newdata.to_csv("regressionDataAugmented.csv")

# Which countries have had higher than average 

newdata = pd.read_csv('regressionDataAugmented.csv', index_col=[0])
newdata['Year'] = newdata.index

def plotdifferences(whichcomparison, whichstates, newdata):

    subset1 = newdata[['Year', 'state', whichcomparison]]
    plotable = subset1.pivot(index = 'Year', columns = 'state', values = whichcomparison)
    plotable = plotable[whichstates]

    # Average should be for all countries
    plotable['AverageAllCountries'] =  newdata[['Year', whichcomparison]].groupby("Year").mean()
    
    n = plotable.shape[1]
    color = iter(cm.tab20b(np.linspace(0, 1, n)))
    for i in range(n):
        c = next(color)
        namecolumn = plotable.columns[i]
        if namecolumn == 'AverageAllCountries':
                c = "black"
                lw = 2.0
                ls = '--'
        elif namecolumn == 'NLD':
                c = "orange"
                lw = 2.0
                ls = 'dashdot'
        else:
                lw = 1.0
                ls ='-'

        x = plotable.index.to_list()
        y = np.log(plotable.iloc[:, i])
        plt.plot(x,y,c=c, linewidth = lw, linestyle = ls, label = namecolumn)
        plt.text(2022, y.iloc[-1], namecolumn + " " + str(y.iloc[-1])[0:4], fontsize = 6)

    plt.legend(loc="upper left")  
    plt.title(whichcomparison)
    plt.show()

# whichcomparison = 'percentStratofGDP'
# whichcomparison = 'percentStratofImports'
# whichstates = toptraders[0:20]
# plotdifferences(whichcomparison, whichstates, newdata)

###################
#countries with highest/lowest average percentgdp
###################
def percentgdptopbottom():
    highestpercentGDP = newdata[['percentStratofGDP', 'state']].groupby('state').mean()
    highestpercentGDP.sort_values(['percentStratofGDP'], ascending = False, inplace=True)
    highestpercentGDP.dropna(inplace=True)
    statehighestpercentGDP = ['NLD'] + highestpercentGDP.index.tolist()[-5:-1] +  highestpercentGDP.index.tolist()[0:5]
    whichcomparison = 'percentStratofGDP'
    whichstates = statehighestpercentGDP
    plotdifferences(whichcomparison, whichstates, newdata)
# percentgdptopbottom()

# get number of rows, assign labels highest, high, average, low, lowest

def addcategorical(data: pd.DataFrame):
    repeatthese = ['highest', 'second', 'third', 'fourth', 'average', 'sixth', 'seventh', 'eighth', 'nineth', 'lowest']
    numrows = range(perstateGDP.shape[0])
    splits = np.array_split(np.array(numrows), len(repeatthese))
    repeaters = [len(numrows) for numrows in splits]
    perstateGDP['rankStratperGDP'] = pd.Series(repeatthese).repeat(repeaters).tolist()
    return perstateGDP

# perstateGDP = newdata[['state', 'percentStratofGDP']].groupby(['state']).sum()
# b1 = perstateGDP.sort_values(['percentStratofGDP'], ascending=False, inplace=True)
# b2 = addcategorical(b1)
# print(b2.groupby('rankStratperGDP').mean())


def percenttradetopbottom():
    highestpercentpercentStrat = newdata[['percentStratofImports', 'state']].groupby('state').mean()
    highestpercentpercentStrat.sort_values(['percentStratofImports'], ascending = False, inplace=True)
    highestpercentpercentStrat.dropna(inplace=True)
    statehighestpercentImports = ['NLD'] + highestpercentpercentStrat.index.tolist()[-5:] +  highestpercentpercentStrat.index.tolist()[0:5]
    whichcomparison = 'percentStratofImports'
    whichstates = statehighestpercentImports
    plotdifferences(whichcomparison, whichstates, newdata)
#percenttradetopbottom()

newdata = pd.read_csv('regressionDataAugmented.csv', index_col=[0])
newdata['Year'] = newdata.index

# the idea, after having taken into account changes in gdp along with a time trend, how much has the purchase of strategic goods increased.
# the residuals show us the change in purchases of strategic goods after having removed or reduced the effects of changes in a state's GDP.
# Negative residuals don't imply that actually spending for strategic goods fell, only that purchases have fallen relative to GDP.  
# If GDP stays the same or falls, while ...

def regressiongdp(data):
    allstateresids = []
    allresidscoef = []
    for st in data.state.unique():
        print(st)
        onestate = data[data['state'] == st]
        # if a state, eg GIB, is missing data, continue
        if onestate.dropna().shape[0] < 28:
            print("Missing data: ", st)
            continue
    
        X = onestate['GDP'].diff()
        # X
        X = sm.add_constant(X)
        X = X.dropna()
        # y
        y = onestate['sumStratGoods'].diff()
        y = y.dropna()
        
        model = sm.OLS(y,X).fit()
        
        onestate['resids']  = model.resid

        allstateresids.append(onestate[['state', 'Year', 'sumStratGoods', 'GDP', 'resids']])
        allresidscoef.append(pd.DataFrame({"State": st, "GDP_coef": model.params['GDP'], "tvalue_GDP": model.tvalues['GDP']}, index = [0]))

    return allstateresids, allresidscoef

rg1, coef = regressiongdp(newdata)
allresids = pd.concat(rg1)
allcoef = pd.concat(coef)
print(allresids)
print(allcoef)
allcoef.to_csv("allcoef.csv")
allresids.to_csv("allresids.csv")

allresids = pd.read_csv("allresids.csv", index_col=[0])

# which states have shown positive increases in percent
print(allresids)

# for st in allresids['state'].unique():
#     country = allresids[allresids['state'] == st]

#     lastvalue = country.loc[2022, 'resids']
#     if lastvalue < 0:
#         print(st)
#         # calc decrease
#         lasttwo = country.loc[[2021,2022], "resids"]
#         print(((lasttwo.iloc[1] - lasttwo.iloc[0])/lasttwo.iloc[0])*100)



# no significant change

# which countries have recently shown an increase, greatest percentage change





def DutchThroughTime():
    yearsData = np.arange(1995, 2023, step=1)
    yearsdata = []
    for yr in yearsData:
        data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y"+ str(yr) + "_V202401.csv")
        data = bc1.subsetStrategicGoods(data, STRATEGOODS)
        data = bc1.addshortdescriptoProdname(data)
        data = data[data['Importer'] == "NLD"]
        data = data[['Value', 'code']]
        gp1 = data[['Value', 'code']].groupby(['code']).sum()
        gp1['Year'] = yr
        gp1['code1'] = gp1.index
        gp1 = gp1.pivot(index = 'Year', columns = 'code1', values = 'Value')
        yearsdata.append(gp1)

    return yearsdata

# dt1 = DutchThroughTime()
# dt1 = pd.concat(dt1)
# dt1.to_csv("sumpercode.csv")
# data = pd.read_csv("sumpercode.csv", index_col=[0])

# data = bc1.add_gdp(data, GDP, '2022')

# bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y2022_V202401.csv"
# data = bc1.readindata(bacidata, tmp_save=False)
# data = bc1.subsetStrategicGoods(data, STRATEGOODS)
# data = bc1.addshortdescriptoProdname(data)

#data = bc1.strategicgoodExportingImportingregions(data, impexp = "Importer")
# strategicproducts = bc1.typesofstrategicgoods(data)
# print(strategicproducts)

# datalongdesc = bc1.addlongdescription(data)

def barchartstrategicprod(strategicproducts):
    ax = strategicproducts[['Value']].plot.barh(stacked=False,  rot=0, cmap='tab20', figsize=(10, 7))
    ax.legend('best')
    plt.tight_layout()
    ax.set_title("Value of trade strategic goods by HS6 2-digit category)")
    plt.show()
    plt.savefig("output\ValueoftradestrategicgoodHS6",bbox_inches='tight')

def pearlsprecioussemi(data):
    
    allprecious = data[data['code'] == "pearls, precious, semi-precious"]
    allprecious.to_csv("tmp.csv")
    # remove gold or silver from isocode
    selectthese = [x for x in allprecious['description'] if "gold" in x or "silver" in x or "diamonds" in x or "Gold" in x or "Silver" in x or "Diamonds" in x or "platinum" in x or "Platinum" in x]
    goldsilver = allprecious[allprecious['description'].isin(selectthese)]

    print(goldsilver)
    allprecious = data['Value'][data['code'] == "pearls, precious, semi-precious"].sum()
    allgoldsilver = goldsilver['Value'].sum()
    print((allgoldsilver/allprecious)*100)

#pearlsprecioussemi(datalongdesc)

def pearlsprecioussemi_thoughtime():
    yearsData = np.arange(1995, 2023, step=1)
    yearly = []
    for i in yearsData:
        print(i)
        bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y" + str(i) + "_V202401.csv"

        data = bc1.readindata(bacidata, verbose = True, tmp_save = False)
        data = bc1.subsetStrategicGoods(data, STRATEGOODS)
        data = bc1.addshortdescriptoProdname(data)
        
        allprecious = data[data['code'] == "pearls, precious, semi-precious"]
        perstate = allprecious[['Value', 'Importer']].groupby(['Importer']).sum()
        perstate.rename(columns = {"Value": str(i)}, inplace = True)
        yearly.append(perstate)

    return yearly

def pearlpercentage():
    yearsprec = pearlsprecioussemi_thoughtime()
    out1 = reduce(lambda left, right: pd.merge(left, right, left_index=True,right_index=True, how='outer'), yearsprec)
    print(out1)
    out1.to_csv("preciousmetals.csv")

    prcmet = pd.read_csv("preciousmetals.csv", index_col=[0])
    print(prcmet)

def preciousthroughtime(data):
    prcmet.sort_values(by = ['2022'], inplace=True,  ascending = False)
    print(prcmet)
    prcmet_subset = prcmet.iloc[0:10,:]
    prcmet_subset.T.plot(title = "Imports of strategic precious metals")
    plt.savefig("output\Importsofstrategicgoods", bbox_inches='tight')
    plt.show()

    top3throughtime = []
    for column in prcmet:
        data = prcmet.nlargest(3, column)
        data = data.index.tolist()
        top3throughtime.append(data)

    top3 = list(itertools.chain.from_iterable(top3throughtime))
    print(set(top3))

# preciousthroughtime(prcmet)    

def standardizedrelative(data):
    prcmet = data.copy()
    prcmet.dropna(inplace=True)
    prcmet = prcmet.T
    
    means = prcmet.mean()
    means.to_csv("tmp.csv")


    prcmet =(prcmet-prcmet.mean())/prcmet.std()
    print(prcmet)

    volt = prcmet.abs().sum()
    topten = volt.sort_values(ascending=False)
    print(topten)
    top20_countries = topten.index.to_list()[0:10]

    standr = prcmet[top20_countries]
    print(standr)
    standr.plot(title = "Standardized data, top volatility through time")
    plt.show()
    plt.savefig("output\Standardizeddatatopvolatility", bbox_inches='tight')

# standardizedrelative(prcmet)

def netimportsCountry():
    bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y2022_V202401.csv"
    data = bc1.readindata(bacidata, tmp_save=False)
    
    # select country
    state = 'NLD'
    imprts = data[data['Importer'] == 'NLD']
    imprts = imprts[['Value', 'Product']]
    gimports = imprts.groupby("Product").sum()

    exprts = data[data['Exporter'] == 'NLD']
    exprts = exprts[['Value', 'Product']]
    gexports = exprts.groupby("Product").sum()

    m1 = gimports.merge(gexports, left_on='Product', right_on ='Product', how='outer', suffixes = ('imports', 'exports'))
    m1['NetImports'] = m1['Valueimports'] - m1['Valueexports']
   
    m1.to_csv('tmp.csv')
    return m1

# n1 = netimportsCountry()
# print(n1)

def allWorldImports():
    yearsData = np.arange(1995, 2023, step=1)
    yearly = []
    for i in yearsData:
        print(i)
        bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y" + str(i) + "_V202401.csv"
        df1 = bc1.readindata(bacidata, tmp_save=True)
        
        # doesn't make sense/impossible to distinguish between imports and exports
        trade_world_value = df1['Value'].sum()
        trade_world_quantity = df1['Quantity'].sum()
        year  = df1['Year'].tolist()[0]

        # subset on strategic goods
        st1 = bc1.subsetStrategicGoods(df1, STRATEGOODS)

        # doesn't make sense/impossible to distinguish between imports and exports
        strategic_world_value = st1['Value'].sum()
        strategic_world_quantity = st1['Quantity'].sum()
        year  = st1['Year'].tolist()[0]

        strategic_world_quantity = st1['Quantity'].sum()
        yearly.append([year, trade_world_value, trade_world_quantity, strategic_world_value, strategic_world_quantity])

    data = pd.DataFrame(yearly)
    data.columns = ['Year', 'trade_world_value', 'trade_world_quantity', 'strategic_world_value', 'strategic_world_quantity']
    data.set_index('Year', inplace=True)

    return data

#data = allWorldImports()
#print(data.head())
#data = data.merge(GDP, left_index=True, right_index=True)
#data['Percentage_Trade'] = ((data['trade_world_value'] * 2)/data['World']) * 100
#data['Percentage_Trade'].plot(title="World")
#data.to_csv("World.csv")

def plotofworlddata():
    world = pd.read_csv("World.csv")
    print(world)

    world[['trade_world_value', 'trade_world_quantity', 'strategic_world_value', 'strategic_world_quantity']].plot(title = "World trade and strategic strategic trade in values and quantities")
    plt.show()
    plt.savefig("output\worldtradestrategic.png")

#plotofworlddata()
# are there countries in which trade of strategic goods has grown relative to their trade
def relativeIncreasePerCountry():
    print("TAKES TIME, GET COFFEE OR TEA AND PASTRY!")
    yearsData = np.arange(1995, 2023, step=1)
    allyears = []
    for i in yearsData:
        print(i)
        bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y" + str(i) + "_V202401.csv"
        df1 = bc1.readindata(bacidata, tmp_save=True)
        countries_import = df1['Importer'].unique()

        allstates = []
        for j in countries_import:
            st = df1[df1['Importer'] == j]
            st_value = st['Value'].sum()
            st_quantity = st['Quantity'].sum()
            year  = st['Year'].tolist()[0]

            #############
            # subset on strategic goods
            #############
            
            st_strat = bc1.subsetStrategicGoods(st, STRATEGOODS)
            st_strat = bc1.addshortdescriptoProdname(st_strat)
    
            # doesn't make sense/impossible to distinguish between imports and exports
            st_strategic_value = st_strat['Value'].sum()
            st_strategic_quantity = st_strat['Quantity'].sum()

            #############
            # subset on precious
            #############
            allprecious = st_strat[st_strat['code'] == "pearls, precious, semi-precious"]
            st_strategic_value_precious = allprecious['Value'].sum()
            st_strategic_quantity_precious = allprecious['Quantity'].sum()

            allstates.append([year, j, st_value, st_quantity, st_strategic_value, st_strategic_quantity, st_strategic_value_precious, st_strategic_quantity_precious])     
                
        data = pd.DataFrame(allstates)
        data.columns = ['Year', 'state', 'state_value', 'state_quantity', 'strategic_state_value', 'strategic_state_quantity', 'st_strategic_value_precious', 'st_strategic_quantity_precious']
        data.set_index('Year', inplace=True)
    
        allyears.append(data)
    
    return allyears

# relativeCountry = relativeIncreasePerCountry()
# relativeCountry = pd.concat(relativeCountry)
# relativeCountry.to_csv("relativeCountry.csv")

############
# compare to world average percentages
############

##################################################################
# percent through time compared to world percent (we have df1 with percentages per country)
##################################################################

# world averages through time

def addprecentages(data):
    allstates = data['state'].unique()
    allstatevalues = []
    for i in allstates:
        dt1 = data[data['state'] == i]
        # all years should be present    
        allyears = dt1['Year'].tolist()
        if len(allyears) >= 28:
            dt1.sort_values(['Year'], ascending=False, inplace = True)
            dt1['percent_strategic'] = (dt1['strategic_state_value']/dt1['state_value'])* 100
            dt1['percent_strategic_precious'] = (dt1['st_strategic_value_precious']/dt1['state_value'])* 100
            allstatevalues.append(dt1)
    return allstatevalues

# data = pd.read_csv("relativeCountry.csv")
# print(data)
# newdf1 = addprecentages(data)
# newdf1 = pd.concat(newdf1)
# print(newdf1)
# newdf1.to_csv("allstatesprecious.csv")

def wrldstate(df1):

    wrldavg = df1[['Year', 'state_value', 'st_strategic_value_precious']].groupby(['Year']).sum()
    wrldavg['percent'] = (wrldavg['st_strategic_value_precious']/wrldavg['state_value'])*100
    wrldavg = wrldavg[['percent']]
    wrldavg.columns = ['World_Avg']

    df2 = df1[['Year', 'state', 'percent_strategic_precious']]
    df2.set_index('Year', inplace=True)
    print(df2)
    states1 = df2['state']
    allstates = []
    for i in states1.unique():
        print(i)
        df3 = df2[df2['state'] == i]
        df3.drop(columns = ['state'], inplace = True)
        df3.columns = [i]
        allstates.append(df3)

    percent_allstates = reduce(lambda left, right: pd.merge(left, right, left_index=True,right_index=True, how='outer'), allstates)
    print(percent_allstates)

    percent1_mean = percent_allstates.mean()
    topten = percent1_mean.sort_values(ascending=False)
    topten = topten.index.to_list()[0:10]

    allstatesWorld = wrldavg.merge(percent_allstates, left_index=True, right_index=True)
    columns = ['World_Avg'] + topten
    allstatesWorld = allstatesWorld[columns]
    allstatesWorld.to_csv('percent_allstates.csv')

    allstatesWorld.plot(title = "Highest percentage of precious metals across time")
    plt.show()
    #plt.savefig("output\Highestpercentageacrosstime")

    return allstatesWorld

#allstatesWorld1 = wrldstate(newdf1)

############
# trade in levels
############

def tradelevels(data):
    g1 = data[['state', 'state_value', 'strategic_state_value']].groupby(['state']).mean()

    ####################
    g1 = g1[g1['state_value'] >= 0]
    ####################

    g1['state_nonstrategic'] = g1['state_value'] - g1['strategic_state_value']
    g1['strategic_percentage'] = (g1['strategic_state_value']/g1['state_value']) * 100
    print("strategic percent: ", g1['strategic_percentage'].mean())
    print("strategic_percentage: ", g1.sort_values(['strategic_percentage']))
    print("strategic percent min max: ", g1['strategic_percentage'].min(), g1['strategic_percentage'].max())
    g1.sort_values(['state_value'], inplace=True)
    print(g1)

    g1 = g1.iloc[-24:,:]
    ax = g1[['strategic_state_value', 'state_nonstrategic']].plot.bar(stacked=True, rot=0, cmap='tab20', figsize=(10, 7))
    ax.legend(loc='best')
    plt.tight_layout()
    ax.set_title("Value of Imports (Strategic vs Non-Strategic)")
    plt.savefig(r"output\ValueofImportsStrategicNonStrategic",bbox_inches='tight')

# data = pd.read_csv("relativeCountry.csv")
# print(data)
# tradelevels(data)

def getpercountryimports(state):
    #############
    # What is happening with Swiz imports?
    #############
    bc1 = baci()    
    bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y2022_V202401.csv"
    df1 = bc1.readindata(bacidata, tmp_save=False)

    imports = df1[df1['Importer'] == state]
    imports = bc1.addprodcode(imports)
    print(imports)

    imports = imports[['Product', 'Value', 'description']].groupby(['Product']).sum()
    print(imports)
    swissimp = imports.sort_values(['Value'])
    swisstopten = swissimp.iloc[-9:,:]
    print(swisstopten)

    #strategic codes
    STRATEGOODS = getStrategicGoods()

    print(set(swisstopten.index).intersection(set(STRATEGOODS)))

# getpercountryimports('ISR')

def chinaimportofstrategicgoodthroughtime():

    df1 = pd.read_csv("xxxx.csv", index_col=[0])
    df1['Year1'] = df1.index

    print(df1)
    print(df1[['Year1', 'percent']].groupby(['Year1']).mean())

    # china = df1[df1['state'] == "NAM"]
    # print(china)
    # china[['percent']].plot()
    # plt.show()

# chinaimportofstrategicgoodthroughtime()

#############
# percentage
#############
def tradepercent(data):
    g2 = data[['state', 'state_value', 'strategic_state_value']].groupby(['state']).mean()
    g2['percentage_strategic'] = g2['strategic_state_value']/g2['state_value']
    g2['percentage_non_strategic'] = (g2['state_value'] - g2['strategic_state_value'])/g2['state_value']
    g2.sort_values(['percentage_strategic'], inplace=True)
    print(g2)

    ###################
    g2 = g2[g2['state_value'] >= 1e6]
    g2 = g2.iloc[-24:,:]
    ax = g2[['percentage_strategic', 'percentage_non_strategic']].plot.bar(stacked=True, rot=0, cmap='tab20', figsize=(10, 7))
    ax.legend(loc='best')
    plt.tight_layout()
    ax.set_title("Percentage of Strategic Imports (Strategic vs Non-Strategic), trade value >= 1e6")
    plt.savefig(r"output\PercentageofStrategic",bbox_inches='tight')

    highestpercent_importers = g2.index.tolist()[0:24]
    return highestpercent_importers

# data = pd.read_csv("relativeCountry.csv")
# highestpercent_importers =  tradepercent(data)
# print(highestpercent_importers)

#############
#closerlookattop25
#############

############################################################

def closerlookattop25():
    yearsData = np.arange(1995, 2023, step=1)
    allyears = []
    for j in yearsData:
        print(j)
        bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y" + str(j) + "_V202401.csv"
        data = bc1.readindata(bacidata, verbose = False, tmp_save = False)
        data = bc1.subsetStrategicGoods(data, STRATEGOODS)
        data = bc1.addshortdescriptoProdname(data)

        for i in highestpercent_importers:
            dt1 = data[data['Importer'] == i]
            if dt1.empty:
                allyears.append([j, i, np.NaN])
            else: 
                dt2 = dt1[['Value', 'code']].groupby(['code']).sum()
                topProd = dt2.sort_values(['Value'], ascending = False)
                allyears.append([j, i, topProd.index.tolist()[0]])

    return pd.DataFrame(allyears)

# top25 = closerlookattop25()
# top25.columns = ['Year', 'State', 'H66']
# top25.sort_values(['State', 'Year'], ascending=[True, True], inplace = True)
# top25.to_csv("top25.csv")

def strategicImports(country):
    yearsData = np.arange(1995, 2022, step=1)
    yearly = []
    for i in yearsData:
        print(i)
        bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y" + str(i) + "_V202401.csv"
        df1 = bc1.readindata(bacidata, tmp_save=True)
        
        # subset on strategic goods
        df1 = bc1.subsetStrategicGoods(df1, STRATEGOODS)

        df_imp = bc1.subsetData(df1, [country], 'Importer', [])
        imp_quant = df_imp['Quantity'].sum()
        imp_value = df_imp['Value'].sum()
        
        df_exp = bc1.subsetData(df1, [country], 'Exporter', [])
        exp_quant = df_exp['Quantity'].sum()
        exp_value = df_exp['Value'].sum()
        
        year  = df_imp['Year'].tolist()[0]
        yearly.append([year, imp_quant, imp_value, exp_quant, exp_value])

    data = pd.DataFrame(yearly)
    data.columns = ['Year', 'Imp_Quantity', 'Imp_Value',  'Exp_Quantity', 'Exp_Value']
    data.set_index('Year', inplace=True)

    return data

# data = strategicImports("PRK")
# print(data.head())

# data = data.merge(GDP, left_index=True, right_index=True)
# data['Percentage_Trade'] = ((data['Exp_Value'] + data['Imp_Value'])/data['Netherlands']) * 100
# data['Percentage_Trade'].plot(title="Nederland")
# data.to_csv("Netherlands.csv")
# plt.show()

# netherlands = pd.read_csv("Netherlands.csv")
# print(netherlands)

def DutchImportsValuePerProd():
    yearsData = np.arange(1995, 2022, step=1)
    yearly = []
    for i in yearsData:
        print(i)
        bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_V202401\BACI_HS92_Y" + str(i) + "_V202401.csv"
        df1 = bc1.readindata(bacidata, tmp_save=True)
        df2 = bc1.subsetData(df1, ["NLD"], 'Importer', [])
        
        # subset on strategic goods
        df2 = bc1.subsetStrategicGoods(df2, STRATEGOODS)

        #add short description
        df2 = bc1.addshortdescriptoProdname(df2)

        g1 = df2[['Value', 'Product', 'code']].groupby(['Product', 'code']).sum(numeric_only=True)
        
        # groupby values per year
        g2 = g1[['Value']].groupby(['code']).sum()
        g2.columns = [i]
        
        yearly.append(g2)
    
    return yearly

# data = DutchImportsValuePerProd()
# print(data)
# #print(data.groupby(['code']).sum())
# from functools import reduce
# xxx = reduce(lambda left, right: pd.merge(left, right, left_index=True,right_index=True, how='outer'), data)
# print(xxx)
# xxx.to_csv("tmp100.csv")

# data = pd.read_csv("tmp100.csv", index_col=[0])
# data1 = data.T
# print(data1)
# data1.to_csv("tmp3.csv")
# data1.plot()
# plt.show()

def allDutchGreenTransition():
    yearsData = np.arange(1995, 2022, step=1)
    yearly = []
    for i in yearsData:
        print(i)
        bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_V202401\BACI_HS92_Y" + str(i) + "_V202401.csv"
        df1 = bc1.readindata(bacidata, tmp_save=True)
        df2 = bc1.subsetData(df1, ["NLD"], 'Importer', [])

        #######
        # green transition
        #######
        strategicProducts = pd.read_csv(r"src\baci\data\strategicProducts.csv", index_col=[0])
        strategicProducts.columns = ['ProdID']
        strategicData = df2[df2['Product'].isin(strategicProducts.loc[:,'ProdID'].tolist())]    

        quant = strategicData['Quantity'].sum()
        value = strategicData['Value'].sum()
        year  = strategicData['Year'].tolist()[0]
        yearly.append([year, quant, value])

    data = pd.DataFrame(yearly)
    data.columns = ['Year', 'Quantity', 'Value']
    data.set_index('Year', inplace=True)

    return data
    
# data = allDutchGreenTransition()
# data.to_csv("DutchGreenTranistionImports.csv")
# data.plot()
# plt.show()

def getSpecificImports(product):

    assert isinstance(product, int)

    yearsData = np.arange(2021, 2022, step=1)
    yearly = []
    for i in yearsData:
        print(i)
        bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_V202401\BACI_HS92_Y" + str(i) + "_V202401.csv.csv"
        df1 = bc1.readindata(bacidata, tmp_save=True)
        df2 = bc1.subsetData(df1, ["NLD"], 'Importer', [])
        print("Number of unique products: ", len(df2['Product'].unique()))

        #######
        # specific products
        #######
        df3 = df2[df2['Product'].isin([product])]

        if df3.empty:
            print("EMPTY LIST!")
            return 
        else:
            quant = df3['Quantity'].sum()
            value = df3['Value'].sum()
            year  = df3['Year'].tolist()[0]
            yearly.append([year, quant, value])
            
        data = pd.DataFrame(yearly)
        data.columns = ['Year', 'Quantity', 'Value']
        data.set_index('Year', inplace=True)

    return data

# product = 811090
# getSpecificImports(product)
    
def getSpecificImportsforHS17_Y202Data(product):

    assert isinstance(product, int)
 
    bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS17_V202401\\BACI_HS17_Y2022_V202401.csv"
    df1 = bc1.readindata(bacidata, tmp_save=True)
    df2 = bc1.subsetData(df1, ["NLD"], 'Importer', [])
    print("Number of unique products: ", len(df2['Product'].unique()))
    
    #######
    # specific products
    #######
    data = df2[df2['Product'].isin([product])]
    
    return data

# product = 811090
# getSpecificImportsforHS17_Y202Data(product)


def allDutchGreenTransitionOneYear(year):
   
    bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y" + str(year) + "_V202401.csv"
    df1 = bc1.readindata(bacidata, tmp_save=True)
    df2 = bc1.subsetData(df1, ["NLD"], 'Importer', [])
    
    #######
    # green transition
    #######
    strategicProducts = pd.read_csv(r"src\baci\data\strategicProducts.csv", index_col=[0])
    strategicProducts.columns = ['ProdID']
    strategicData = df2[df2['Product'].isin(strategicProducts.loc[:,'ProdID'].tolist())]    

    print(len(strategicData['Product'].unique()))

    return strategicData

#data = allDutchGreenTransitionOneYear(2021)
#data.to_csv("tmp.csv")
#def allOECDGreenTransistion

