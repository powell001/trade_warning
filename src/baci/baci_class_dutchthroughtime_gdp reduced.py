import pandas as pd
import os
import sys 
import numpy as np
from functools import reduce
import itertools
from ast import literal_eval #converts object list to list of strings
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import statsmodels.api as sm
from statsmodels.formula.api import ols

# this points to a Python file with the function country_mappings (not used)
from combine_country_regions import country_mappings

# not great practice, but this removes warnings from the output
import warnings
warnings.filterwarnings("ignore")

# display settings so I can see more on the screen
desired_width=1000
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)
pd.options.display.max_rows = 50

#############################################################
# set this to point to your folder or create a new folder,
# (in my case my computer is called jpark and I called the folder trade_warning) 
#############################################################
os.chdir(r'C:\Users\jpark\VSCode\trade_warning\\')

# points to country codes as defined by BACI
COUNTRY_CODES = r"C:\Users\jpark\Downloads\BACI_HS22_V202401b\country_codes_V202401b.csv"
# point to product codes
PRODUCT_DESCRIPTION = r"C:\Users\jpark\Downloads\BACI_HS22_V202401b\product_codes_HS22_V202401b.csv"
# add region data, might be better sources
ADD_REGIONS = r"data\\iso_countries_regions.csv"
# add short HS2 description (could be better descriptions)
SHORT_CODES = r"src\baci\data\hs6twodigits.csv"
# add long product description
LONG_DESCRIPTION = r"C:\Users\jpark\Downloads\BACI_HS22_V202401b\product_codes_HS22_V202401b.csv"
# add gdp data
GDP_DATA = r"data\\global_gdp.csv"
# strategic goods
STRATEGIC_GOODS = r"src\\pdf_extractor\\strategicProducts.csv"
# wgi values
WGI_VALUES = r"src\baci\various_codes\WGI_iso3.csv"
# location of BACI unpacked data, note(!) last part of string, I keep the year 
# as a variable, so I can write, for instance:
j = 2022
BACI_DATA = r"C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401b\BACI_HS92_Y"
BACI_DATA + str(j) + "_V202401.csv"

# # QUESTIONS

# # Dutch vulnerablity through time
# # Dutch imports of strategic goods through time
# # Vulnerability is partially based on the conditions of an exporting country
#     # From which countries does the Netherlands import strategic goods?
#     # What is the total value of Dutch imports of strategic goods?, has that volume changed through time?
# # How has the vulnerability of a product that the Netherlands imports change through time
# # Look at value of imports through time.  
# # Histogram of WGI value

class baci:
    '''baci class contains the methods to load baci data and add characteristics such as geographic and strategic'''
    def readindata(self, bacidata, verbose = False, tmp_save = True) -> pd.DataFrame:
        '''main method to read in baci data'''
        df1 = pd.read_csv(bacidata, usecols=['t','i','j','k','v','q'], 
                          dtype= {'t': 'int64',
                                  'i': 'int64', 
                                  'j': 'int64', 
                                  'k': 'object',
                                  'v': 'float64',
                                  'q': 'object'}
                          )

        # This is too complicated, but '   NA' should be converted to float
        df1['q'] = df1['q'].apply(lambda x: x.strip()) # remove spaces in data
        df1['q'].replace('NA', np.NaN, inplace=True)   # np.NaN is different than string NaN
        df1['q'] = df1['q'].astype(float)

        # rename columns to make them meaningful to humans
        df1.rename(columns={'t': 'Year', 'i': 'Exporter', 'j': 'Importer', 'k': 'Product', 'v': 'Value', 'q': 'Quantity'}, inplace=True)

        # replace number with name of country *exporter* 
        iso1 = pd.read_csv(COUNTRY_CODES, usecols=['country_code', 'country_iso3'])
        df1 = df1.merge(iso1, left_on="Exporter", right_on="country_code", how="left")
        df1.drop(columns=['country_code', 'Exporter'], inplace = True)
        df1.rename(columns={"country_iso3": "Exporter"}, inplace=True)
    
        # replace number with name of country *importer*
        df1 = df1.merge(iso1, left_on="Importer", right_on="country_code", how="left")
        df1.drop(columns=['country_code', 'Importer'], inplace = True)
        df1.rename(columns={"country_iso3": "Importer"}, inplace=True)

        # 2015 has some strange data, take only Values greater than 10.00, otherwise number of exporting countries in 2015 is an outlier
        df1 = df1[df1['Value'] > 10.00]

        # if verbose is True, this will print out
        if verbose:
            hcodes = [str(x)[0:2] for x in df1["Product"]]
            print(set(hcodes))
            print(len(set(hcodes)))

        # make product code and int, otherwise its an object which can be confusing
        df1['Product'] = df1['Product'].astype(int)    

        return df1
    
    def addprodcode(self, data):
        '''add the product description if needed'''
        # add product_codes
        prodcodes = pd.read_csv(PRODUCT_DESCRIPTION, usecols=['code', 'description'])
        # product '9999AA' appears to be a filler--empty
        mask = prodcodes['code'] == '9999AA'
        prodcodes = prodcodes[~mask]
        # I love merges, note its a left merge, I want all baci data to have a code, but dont care for product codes without products.
        data = data.merge(prodcodes, left_on = "Product", right_on = "code", how = "left")
        
        return data
       
    def subsetData(self, data_param: pd.DataFrame, iso3_param: list[str], imp_exp_param: str, products_param: list[str], minvalue_param=0.0) -> pd.DataFrame():
        '''Select the importing (so not exporting at this point) country and the product, can be extended to select exporters'''
        
        df1 = data_param.copy()

        # select the importing (so, again, not exporting at this point) country and the product
        if products_param:
            out1 = df1[(df1[imp_exp_param].isin(iso3_param)) & (df1["Product"].isin(products_param))]
            out1.sort_values(by=['Value', 'Importer'], inplace=True, ascending=False)
            out1 = out1[out1['Value'] >= minvalue_param]
            out1['Value'] = out1['Value'].astype(int)
        else: # return all products in no product selected
            out1 = df1[df1[imp_exp_param].isin(iso3_param)]
            out1.sort_values(by=['Value', 'Importer'], inplace=True, ascending=False)
            out1 = out1[out1['Value'] >= minvalue_param]
            out1['Value'] = out1['Value'].astype(int)

        return out1
    
    def subsetStrategicGoods(self, data, STRATEGICGOODS: list):
        '''Selects products based on a list of strings'''
        df1 = data.copy()
        df2 = df1[df1['Product'].isin(STRATEGICGOODS)]
        return df2
    
    def addregions(self, data):
        '''Add regions to data, there are two of these almost identical functions, I need to combine them or get rid of one of them'''
        regions = pd.read_csv(ADD_REGIONS)
    
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
        '''Add regions to data, there are two of these almost identical functions, I need to combine them or get rid of one of them'''
        if exim == "Exporter":
            iso_regions = pd.read_csv(ADD_REGIONS)
            iso_regions = iso_regions[['alpha-3', 'region']]
            data = data.merge(iso_regions, left_on="Exporter", right_on="alpha-3", how="left")
            data.rename(columns = {'region': 'Exporter_Region'}, inplace = True)
            data.drop(columns=["alpha-3"], inplace=True)
        elif exim == "Importer":
            iso_regions = pd.read_csv(ADD_REGIONS)
            iso_regions = iso_regions[['alpha-3', 'region']]
            data = data.merge(iso_regions, left_on="Importer", right_on="alpha-3", how="left")
            data.rename(columns = {'region': 'Importer_Region'}, inplace = True)
            data.drop(columns=["alpha-3"], inplace=True)
        else: 
            print("Error")

        return data

    def addshortdescriptoProdname(self, data):
        '''Add short product description based on codes'''

        localdata = data.copy()
        prod_h6 = pd.read_csv(SHORT_CODES, dtype = str)

        # this is necessary because codes 1:9 should be 01:09
        localdata.loc[:, 'code'] = ["0" + x if len(x) == 5 else x for x in localdata['Product'].astype(str)]

        localdata['shrtDescription'] = localdata['code'].astype(str).str[0:2]
        proddesc = localdata.merge(prod_h6, left_on="shrtDescription", right_on="code")
        proddesc['product'] = proddesc['product'] + "_" + proddesc['shrtDescription']
        proddesc.drop(columns = {'code_x', 'shrtDescription', 'code_y'}, inplace = True)

        proddesc.rename(columns = {"product": "code"}, inplace = True)

        return proddesc
    
    def addlongdescription(self, data):
        '''Add product product description based on codes'''
        localdata = data.copy()
        longdesc = pd.read_csv(LONG_DESCRIPTION, dtype = str)

        # this is necessary because codes 1:9 should be 01:09
        localdata.loc[:, 'Product'] = ["0" + x if len(x) == 5 else x for x in localdata['Product'].astype(str)]

        longdesc.rename(columns = {"code": "isocode"}, inplace=True)
        longproddesc = localdata.merge(longdesc, left_on="Product", right_on="isocode", how = 'left', suffixes = ['x', 'y'])
       
        r1 = localdata.shape[0]
        r2 = longproddesc.shape[0]
        assert r1 == r2

        return longproddesc
    
    def add_gdp(self, data, GDP, year):
        '''Join GDP to data'''

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
        '''sums value across all product categories'''

        ### Relative size of Step1 inputs per product
        g = data_param[['Product', 'Value']].groupby(['Product']).sum()
        valueofStep1products = g.apply(lambda x: x.sort_values(ascending=False))
        valueofStep1products['Percentage'] = 100 * (valueofStep1products / valueofStep1products['Value'].sum())

        return valueofStep1products

    def valuepercountryacrossprods(self, data_param, imp_exp_param):
        '''not used, finds percentage of a product category compared to total'''
        ### Relative size of Step1 inputs per exporter
        g = data_param[[imp_exp_param, 'Value']].groupby([imp_exp_param]).sum()
        valueofStep1perExporter = g.apply(lambda x: x.sort_values(ascending=False))
        valueofStep1perExporter['Percentage'] = 100 * (valueofStep1perExporter / valueofStep1perExporter['Value'].sum())

        print(valueofStep1perExporter)
        return valueofStep1perExporter

    def valueperprod(self, data_param, imp_exp_param):
        '''not used, another means to get summed value of exported or imported product categories'''
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
        '''not used, if the country a member of the OECD, get percentage of value of OECD exports/imports'''
        
        assert (imp_exp_param == 'Exporter_ISO3') or (imp_exp_param == 'Importer_ISO3'), "needs to be Exporter_ISO3 or Importer_ISO3"

        grp_perCountry = data_param[['Value', imp_exp_param]].groupby([imp_exp_param]).sum().reset_index()
        merged1 = grp_perCountry.merge(baci_countries_param[['ISO3', 'OECD']], left_on=imp_exp_param, right_on="ISO3", how="left")

        out = merged1[[imp_exp_param, 'Value', 'OECD']].groupby(['OECD']).sum().reset_index()
        out['Percentage'] = 100 * (out['Value'] / out['Value'].sum())
        out.sort_values(['Percentage'], ascending=False,inplace=True)
        print(out)

        return out

    def strategicgoodExportingImportingregions(self, data, impexp: str):
        '''sum of value of strategic imports/exports grouped by region'''

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

        return data


    def typesofstrategicgoods(self, data):
        '''not used, just sums value by product code and sorts, done in other code'''
        valuestrategicgood = data[['Value', 'code']].groupby("code").sum()
        strategicproducts = valuestrategicgood.sort_values(['Value'], ascending = False)

            
        return strategicproducts

def GDPData():
    '''should alway be run, need to move to BACI class'''
    # https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?end=2022&start=1960&view=chart
    # Taiwan comes from IMF data, added by hand. https://www.imf.org/external/datamapper/NGDPD@WEO/OEMDC/ADVEC/WEOWORLD
    
    data = pd.read_csv(GDP_DATA, index_col=[1], skiprows=4)
    data = data.drop(columns=['Country Name', 'Indicator Code', 'Indicator Name'])
    data = data.T
    return data
GDP = GDPData()

def getStrategicGoods():
    '''should alway be run, need to move to BACI class'''
    data = pd.read_csv(STRATEGIC_GOODS, index_col=[0],dtype= {'0': 'object'})
    # objects to ints
    data.iloc[:,0] = data.iloc[:,0].astype(int)
    data = data.iloc[:,0].tolist()

    return data
STRATEGICGOODS = getStrategicGoods()

def getWGI():
    '''should alway be run, need to move to BACI class'''
    wgi = pd.read_csv(WGI_VALUES, index_col=[0])
    wgi.columns = ['ISO3', 'WGI_2022xxx']
    return wgi
WGI = getWGI()

# #############################################################
# # INITIALIZE object, needs to be run to create a BACI object instance
bc1 = baci()
# #############################################################

# test data
j = 2022
bacidata = BACI_DATA + str(j) + "_V202401b.csv"
test_data = bc1.readindata(bacidata, verbose = False, tmp_save = False)

def WorldMarketConcentration(data, product, verbose = False):
    '''
      Following CBS`: De indicator wordt als volgt berekend: neem voor ieder land dat dit product exporteert het aandeel van dit land in de wereldmarkt.
    '''
    # select the product
    oneProd = data[data['Product'] == product]

    # who exports this product
    ExporterOfProduct = oneProd['Exporter'].unique()

    Exports = oneProd[['Value', 'Exporter']]
    totalExportsOfProd = Exports[["Value"]].sum()

    # largest 3 non_EU exports in value, not needed but nice to check
    largestExporters = Exports[['Value', 'Exporter']].groupby(['Exporter']).sum()
    largestExporters = largestExporters.sort_values(by = ['Value'], ascending=False)
    
    # MUST BE RUN PER-COUNTRY
    running_total = 0
    for st in ExporterOfProduct:
        ACountryTotalExp = oneProd['Value'][(oneProd['Exporter'] == st)].sum()
        tt = np.power(ACountryTotalExp/totalExportsOfProd, 2)
        running_total = running_total + tt

    if verbose:
        print(" Product: ", product,
            "\n WorldMarketConcentration: ", running_total,
            "\n Top three exporters: ", largestExporters.index.tolist()[0:3])

    return running_total

# note that the product code is an int so product 071332 should be entered as 71332
out = WorldMarketConcentration(test_data, 261390, verbose = False)
print(out)

def WGI_calc(data, product,  wgi, country_name = 'NLD', verbose = False):
    '''
    Following CBS`: De indicator wordt als volgt berekend: neem voor ieder land dat dit product exporteert het aandeel van dit land in de wereldmarkt.
    NOTE:  I wasn't able to reproduce CBS's analysis, this is purely a application of the HHI index to NLD.
    '''

    # select the one product
    oneProd = data[data['Product'] == product]

    ExporterOfProducttocountry = oneProd['Exporter'][oneProd['Importer'] == country_name].unique()

    # Exports
    Exports = oneProd[['Value', 'Exporter', 'Importer']]
    largestExporterstocountry = Exports[['Exporter', 'Value']][Exports['Importer'] == country_name].groupby(['Exporter']).sum()
    largestExporterstocountry = largestExporterstocountry.sort_values(by = ['Value'], ascending=False)

    # confusing, but largestExporterstocountry actually includes all countries exporting to a country
    totalExportsOfProdtocountry = largestExporterstocountry[["Value"]].sum()

    # MUST BE PER-COUNTRY
    wgi_total = []
    thesestatesmissing = []
    for st in ExporterOfProducttocountry:
        # some states are missing, skip to next one
        if (st not in oneProd['Exporter'].values) or (st not in wgi["ISO3"].values):
            thesestatesmissing.append(st)
            continue

        ACountryTotalExp = oneProd['Value'][(oneProd['Exporter'] == st) & (oneProd['Importer'] == country_name)].sum()
        wgi_mult_cntry = wgi[["WGI_2022xxx"]][wgi["ISO3"] == st].values[0][0]
        percent_ofExports = ACountryTotalExp / totalExportsOfProdtocountry
        wgi_weight_cntry =  percent_ofExports * wgi_mult_cntry
        wgi_total.append(wgi_weight_cntry.tolist()[0])

    total_wgi = sum(wgi_total)

    if verbose:
        print(" Product: ", product,
            "\n WGI_forProduct: ", total_wgi,
            "\n Top three exporters: ", largestExporterstocountry.index.tolist()[0:3])

    return total_wgi, totalExportsOfProdtocountry.Value, largestExporterstocountry.index.tolist(), thesestatesmissing

# test
# out = WGI_calc(test_data, 710610, WGI, "NLD",  verbose = True)
# print(out)

def ImportDiversificationcountry(data, product, country_name = 'NLD', verbose = False):
    '''
    Following CBS`: De indicator wordt als volgt berekend: neem voor ieder land dat dit product exporteert het aandeel van dit land in de wereldmarkt.
    NOTE:  I wasn't able to reproduce CBS's analysis, this is purely a application of the HHI index to NLD.
    '''

    oneProd = data[data['Product'] == product]

    ExporterOfProducttocountry = oneProd['Exporter'][oneProd['Importer'] == country_name].unique()

    # fyi: largest exports in value to country
    Exports = oneProd[['Value', 'Exporter', 'Importer']]
    largestExporterstocountry = Exports[['Exporter', 'Value']][Exports['Importer'] == country_name].groupby(['Exporter']).sum()
    largestExporterstocountry = largestExporterstocountry.sort_values(by = ['Value'], ascending=False)
    totalExportsOfProdtocountry = largestExporterstocountry[["Value"]].sum()

    # wrong for 2015
    totalnumberofExportstocountry = largestExporterstocountry.shape[0]
    #print('totalnumberofExportstocountry:',  totalnumberofExportstocountry)

    topthreeexportersinorder = largestExporterstocountry.iloc[0:3, 0].index.tolist()

    # if one country supplies more than 50% of value
    percentonecountry = 100 * (largestExporterstocountry.iloc[0:1, 0].sum()/totalExportsOfProdtocountry).values[0]
    percenttwocountry = 100 * (largestExporterstocountry.iloc[0:2, 0].sum()/totalExportsOfProdtocountry).values[0]
    
    # get % top three
    if largestExporterstocountry.shape[0] >= 3:
        valuetopthree_ifthree = 100*(largestExporterstocountry.iloc[0:3, 0].sum()/totalExportsOfProdtocountry)
    else:
        valuetopthree_ifthree = 100*(largestExporterstocountry.iloc[0:1, 0].sum()/totalExportsOfProdtocountry)

    # MUST BE PER-COUNTRY
    # use float here
    running_total = 0
    for st in ExporterOfProducttocountry:
        ACountryTotalExp = oneProd['Value'][(oneProd['Exporter'] == st) & (oneProd['Importer'] == country_name)].sum()
        tt = np.power(ACountryTotalExp/totalExportsOfProdtocountry, 2)
        running_total = running_total + tt

    if verbose:
        print(" Product: ", product,
              "\n ImportDiversificationcountry: ", running_total,
              "\n Top three exporters: ", largestExporterstocountry.index.tolist()[0:3])

    # ugly code to account for diffeences in values and arrays, must be a better way
    if type(running_total) == pd.core.series.Series:
        running_total = running_total.array[0]

    return running_total, valuetopthree_ifthree['Value'], totalnumberofExportstocountry, percentonecountry, percenttwocountry, topthreeexportersinorder

# test
# out1 = ImportDiversificationcountry(test_data, 710610, country_name = 'NLD', verbose = True)
# print(out1)

def importratio(data, product, country_name):
    '''For all products, divide total imports by total exports'''
    oneProd = data[data['Product'] == product]

    # total imports to a country
    importsOfProductfromAllcountries = oneProd['Value'][oneProd['Importer'] == country_name]
    imports = importsOfProductfromAllcountries.sum()
    # total exports of same product to all countries
    exportsOfProducttocountry = oneProd['Value'][oneProd['Exporter'] == country_name]
    exports = exportsOfProducttocountry.sum()

    impRatio = (imports - exports)/(exports + imports)

    return impRatio

# out1 = importratio(test_data, 710610, 'NLD')
# print(out1)

def importscarcityEurope(product):
    bc2 = baci()
    data = BACI_DATA + str(2022) + "_V202401b.csv"
    data = bc2.readindata(data, verbose = False, tmp_save = False)
    data = bc2.addregions(data)
    
    print(data.head())
    print(data["Importer_region_eu"].unique())

    # select product
    oneProd = data[data['Product'] == product]

    # total imports to a country
    # total imports EU
    totalImpEU = oneProd[oneProd['Importer_region_eu'] == "EU"]
    totalImpEU = totalImpEU['Value'].sum()

    # total imports out EU
    totalimpNOT_EU = oneProd[oneProd['Importer_region_eu'] == "EU"] #just like above
    totalimpNOT_EU = totalimpNOT_EU[totalimpNOT_EU["Exporter_region_eu"] == 'Not_EU']
    totalImpNOT_EU = totalimpNOT_EU['Value'].sum()

    # total imports not from EU
    scarcityEU = totalImpNOT_EU/totalImpEU

    return scarcityEU
    
# out1 = importscarcityEurope(710610)    
# print(out1)


def substitutionEurope(product):
    bc2 = baci()
    data = BACI_DATA + str(2022) + "_V202401b.csv"
    data = bc2.readindata(data, verbose = False, tmp_save = False)
    data = bc2.addregions(data)
    data.to_csv("tmp.csv")

    print(data.head())
    print(data["Importer_region_eu"].unique())

    # select product
    oneProd = data[data['Product'] == product]

    # total exports EU
    totalExpEU = oneProd[oneProd['Exporter_region_eu'] == "EU"]
    totalExpEU = totalExpEU['Value'].sum()

    # total imports EU from nonEU countries
    totalimpNOT_EU = oneProd[oneProd['Importer_region_eu'] == "EU"] #just like above
    totalimpNOT_EU = totalimpNOT_EU[totalimpNOT_EU["Exporter_region_eu"] == 'Not_EU']
    totalImpNOT_EU = totalimpNOT_EU['Value'].sum()

    # total imports not from EU
    substitutionEU = totalImpNOT_EU/totalExpEU

    return substitutionEU
    
# out1 = substitutionEurope(710610)    
# print(out1)


def add_hhi_tostrategic(STRATEGICGOODS, country_name = "NLD"):
    '''Function loops over all baci years for a country (only tested with NLD for now)'''
    yearsData = np.arange(1995, 2023, step=1)
    yearsdata = []
    for yr in yearsData:
        print(yr)
        data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401b\BACI_HS92_Y"+ str(yr) + "_V202401b.csv")
        data = bc1.addregions(data)
        data = bc1.subsetStrategicGoods(data, STRATEGICGOODS)
        data = data[data['Importer'] == country_name]

        collecthhis = []
        for prd in data['Product'].unique():
            out, valuetopthree, totalnumberofExportstoNLD, percentonecountry, percenttwocountry, topthreeexportersinorder = ImportDiversificationcountry(data, prd, verbose = False)
            collecthhis.append([prd, out, valuetopthree, totalnumberofExportstoNLD, percentonecountry, percenttwocountry, topthreeexportersinorder])

        df1 = pd.DataFrame(collecthhis)
        df1.columns = ['Product', 'HHI', 'Percenttopthree_ifthree', 'totalnumberofExportstoNLD', 'percentonecountry', 'percenttwocountry', 'topthreeexportersinorder']
       
        data = data.merge(df1, left_on="Product", right_on="Product")

        yearsdata.append(data)

    out = pd.concat(yearsdata)
    out.to_csv("HHI_concentration.csv", index=False)
    
    return out

# test
# out1 = add_hhi_tostrategic(STRATEGICGOODS, country_name = "NLD")
# print(out1)

# # # above takes a long time to run, so I save it so I don't have to run the above code too often
# hhi = pd.read_csv("HHI_concentration.csv")
# hhi['LargestExporter'] = [literal_eval(x)[0] for x in hhi['topthreeexportersinorder']]
# # add WGI
# hhi_wgi = hhi.merge(WGI, left_on="LargestExporter", right_on="ISO3")
# hhi_wgi.drop(columns = ["ISO3"], inplace = True)
# hhi_wgi.to_csv("hhi_wgi.csv", index=False)

def avgoverregions_HHI():
    data = pd.read_csv("hhi_wgi.csv")
    data = bc1.addshortdescriptoProdname(data)
    mean_perregion_HHI = data[['HHI', 'code']].groupby(['code']).mean()
    mean_perregion_HHI.sort_values(['HHI'], ascending = False, inplace = True)
    mean_perregion_HHI.to_csv("mean_perregion_HHI.csv")
#avgoverregions_HHI()

def plothhithroughtime():
    data = pd.read_csv("hhi_wgi.csv")    

    ############
    data = data[data['Value'] >= 500]

    data = bc1.addlongdescription(data)

    # hhi through time avg all Dutch imports
    avgHHI = data[['HHI', 'Year']].groupby(['Year']).mean()

    theseitems = ['Cobalt', 'Lithium', 'Zinc', 'Aluminium', 'Nickel', 'Copper']

    allnumbersellers = []
    allcommodities = []
    for i in theseitems:
        print(str.lower(i))
        selectthese = [x for x in data['description'] if i in x or str.lower(i) in x]
        data1 = data[data['description'].isin(selectthese)]
        data2 = data1[['HHI', 'Year']].groupby(['Year']).mean()
        data2.columns = [i]

        allcommodities.append(data2)

    data = pd.concat(allcommodities, axis=1)
    data['Average'] = avgHHI
    print(data)

    plt.rcParams['figure.figsize'] = 10, 7
  
    lastyear = []
    for i in range(0, data.shape[1]):
        colname = data.columns[i]
        end_value = data.iloc[:,i].values[-1].round(2)
        lastyear.append(pd.DataFrame({'Value': [end_value], 'Col': [colname]}))

        if colname == 'Average':
            name_column = data.columns[i]
            plt.plot(data.index, data.iloc[:,i].round(2), label = name_column)
            plt.text(2022, end_value, name_column + " " + str(end_value), fontsize = 6)
            plt.title("HHI through time important commodity groups")
        elif colname == 'Nickel':
            name_column = data.columns[i]
            plt.plot(data.index, data.iloc[:,i].round(2), label = name_column)
            plt.text(2022, end_value + .001, name_column + " " + str(end_value), fontsize = 6)    

        else:
            name_column = data.columns[i]
            plt.plot(data.index, data.iloc[:,i].round(2), label = name_column)
            plt.text(2022, end_value, name_column + " " + str(end_value), fontsize = 6)
        
    t1 = pd.concat(lastyear)
    print(t1.sort_values("Value"))
    plt.legend(loc="upper left",  fontsize="6") 
    plt.savefig(r"output\hhicommodities",bbox_inches='tight')
    plt.show()
# plothhithroughtime()
 
def allsellersthroughtime():
    
    data = pd.read_csv("hhi_wgi.csv")    
    
    #######################
    #data = data[data['Value'] >= 500]
    
    data = bc1.addlongdescription(data)

    # hhi through time avg all Dutch imports
    avgNumSellers = data[['Exporter', 'Year']].groupby(['Year']).nunique()

    theseitems = ['Cobalt', 'Lithium', 'Zinc', 'Aluminium', 'Nickel', 'Copper']
    
    allnumbersellers = []
    for i in theseitems:
        print(str.lower(i))
        selectthese = [x for x in data['description'] if i in x or str.lower(i) in x]
        data1 = data[data['description'].isin(selectthese)]

        datanumsellers = data1[['Exporter', 'Year']].groupby(['Year']).nunique()
        datanumsellers.columns = [i]

        allnumbersellers.append(datanumsellers)

    data = pd.concat(allnumbersellers, axis=1)
    data['Average'] = avgNumSellers
    
    ##################
    # plot
    ##################

    plt.rcParams['figure.figsize'] = 10, 7
  
    lastyear = []
    for i in range(0, data.shape[1]):
        colname = data.columns[i]
        end_value = data.iloc[:,i].values[-1].round(0)
        lastyear.append(pd.DataFrame({'Value': [end_value], 'Col': [colname]}))

        if colname == 'Average':
            name_column = data.columns[i]
            plt.plot(data.index, data.iloc[:,i].round(0), label = name_column)
            plt.text(2022, end_value, name_column + " " + str(end_value), fontsize = 6)
            plt.title("Number exporters through time by important commodity groups")
        elif colname == 'Lithium':
            name_column = data.columns[i]
            plt.plot(data.index, data.iloc[:,i].round(0), label = name_column)
            plt.text(2022, end_value + .5, name_column + " " + str(end_value), fontsize = 6)    
        else:
            name_column = data.columns[i]
            plt.plot(data.index, data.iloc[:,i].round(0), label = name_column)
            plt.text(2022, end_value, name_column + " " + str(end_value), fontsize = 6)
        
    t1 = pd.concat(lastyear)
    print(t1.sort_values("Value"))
    plt.legend(loc="upper left",  fontsize="6") 
    plt.savefig(r"output\numberimporters",bbox_inches='tight')
    plt.show()

    return allnumbersellers
# allsellersthroughtime()

def wgimeasure(prodname):

    plt.rcParams['figure.figsize'] = 10, 7

    data = pd.read_csv("hhi_wgi.csv")        
    data = bc1.addlongdescription(data)
    theseitems = [prodname]
    
    WGI = getWGI()

    allitems = []
    for i in theseitems:
        print(str.lower(i))
        selectthese = [x for x in data['description'] if i in x or str.lower(i) in x]
        data1 = data[data['description'].isin(selectthese)]

    data2 = data1[['Year', 'Value', 'Exporter', 'Product']]
    data3 = data2.merge(WGI, left_on = "Exporter", right_on = "ISO3") 
    uuu = data3[data3['Product'] == '282200']
    uuu.to_csv("uuu.csv")

    allyrs = []
    for yr in data3['Year'].unique():
        datayr = data3[data3["Year"] == yr]
        allprod = []
        namescol = []
        for prd in data3['Product'].unique():
            datayrprod = datayr[datayr['Product'] == prd]
            valueprod = datayrprod['Value'].sum()
            datayrprod['product'] = datayrprod['WGI_2022xxx'] * (datayrprod['Value']/valueprod)
            wdisum = datayrprod['product'].sum()
            allprod.append(wdisum)
            namescol.append(prd)
        allyrs.append(allprod)
   
    data = pd.DataFrame(data = allyrs, index = data3['Year'].unique())
    data.columns = namescol

    ################################
    ################################
    lastyear = []
    for i in range(0, data.shape[1]):
        colname = data.columns[i]
        end_value = data.iloc[:,i].values[-1]
        lastyear.append(pd.DataFrame({'Value': ["{:.2f}".format(end_value)], 'Col': [colname]}))

        name_column = data.columns[i]
        plt.plot(data.index, data.iloc[:,i], label = name_column)
        plt.ylabel("WGI")
        plt.text(2022, end_value, name_column + " " + str("{:.2f}".format(end_value)), fontsize = 6)
        plt.title("Value weighted WGI per product per year for " + prodname)
        
    t1 = pd.concat(lastyear)
    print(t1.sort_values("Value"))
    plt.legend(loc="upper left",  fontsize="6") 
    plt.savefig(r"output\wgiweighted" + prodname,bbox_inches='tight')

    return data
#out1 = wgimeasure(prodname = 'Metals: silver powder')

def wgicountries():
    data = pd.read_csv("hhi_wgi.csv")    
    
    #######################
    #data = data[data['Value'] >= 500]
    
    data = bc1.addlongdescription(data)

    # hhi through time avg all Dutch WGI (not necessary, only one year of data)
    avgWGI = data[['WGI_2022xxx', 'Year']].groupby(['Year']).mean()

    theseitems = ['Lithium', 'Cobalt', 'Zinc', 'Aluminium', 'Nickel', 'Copper']
    
    allitems = []
    for i in theseitems:
        print(str.lower(i))
        selectthese = [x for x in data['description'] if i in x or str.lower(i) in x]
        data1 = data[data['description'].isin(selectthese)]
        meanx = data1[['WGI_2022xxx', 'Year']].groupby(['Year']).mean()
        stdx = data1[['WGI_2022xxx', 'Year']].groupby(['Year']).std()
        allitems.append(pd.concat([meanx, stdx], axis=1))
       
    data = pd.concat(allitems)
    data['items'] = [val for val in theseitems for _ in (0, 1)]
    data.columns = ['Item', 'Mean', 'Std']
    return data
# out1 = wgicountries()
# print(out1)

def lithium():
    data = pd.read_csv("hhi_wgi.csv")    
    
    data = bc1.addlongdescription(data)

    theseitems = ['Lithium', 'Cobalt']
    
    allitems = []
    for i in theseitems:
        print(str.lower(i))
        selectthese = [x for x in data['description'] if i in x or str.lower(i) in x]
        data1 = data[data['description'].isin(selectthese)]
        numberofproductsperyear = data1[['Year', 'Product']].groupby(['Year']).nunique()
        numberofexportperyear = data1[['Exporter', 'Year', 'Product']].groupby(['Year', 'Product']).nunique()
        namesexporters = data1[['Exporter', 'Year', 'Product']].groupby(['Year', 'Exporter']).nunique()
        valueperproduct = data1[['Year', 'Product', 'Exporter', 'Value']].groupby(['Year', 'Exporter', 'Product']).sum()
        valueperproduct['Exporter1'] = valueperproduct.index
        percountry = valueperproduct[['Exporter1', 'Value']].groupby(['Exporter']).sum()

        allitems.append(pd.concat([numberofproductsperyear]))
    data = pd.concat(allitems, axis=1)
    data.columns = theseitems
    return data
#out1 = lithium()
#print(out1)

def dutchimportsregionthroughtime():
    yearsData = np.arange(1995, 2023, step=1)
    yearsdata = []
    for yr in yearsData:
        print(yr)
        data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y"+ str(yr) + "_V202401.csv")
        data = bc1.addregions(data)
        data = bc1.subsetStrategicGoods(data, STRATEGICGOODS)
        data = data[data['Importer'] == "NLD"]
        data = data[['Value', 'Exporter_sub_region']]
        gp1 = data[['Value', 'Exporter_sub_region']].groupby(['Exporter_sub_region']).sum()
        gp1['Year'] = yr
        gp1['subregion'] = gp1.index

        gp1 = gp1.pivot(index = 'Year', columns = 'subregion', values = 'Value')
        yearsdata.append(gp1)

    dt1 = pd.concat(yearsdata)
    dt1.to_csv("sumperregion.csv")

    return dt1
# dt1 = dutchimportsregionthroughtime()
# data = pd.read_csv("sumperregion.csv", index_col=[0])
# data.drop(columns=["Micronesia", "Melanesia", "Polynesia"], inplace=True)
# np.log(data).plot()
# plt.show()

def compareDutchimportperregion(data):
    plt.rcParams['figure.figsize'] = 10, 7
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
            plt.title("Logged origin of Dutch imports of strategic goods")
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
    plt.legend(loc="upper left",  fontsize="6") 
    plt.savefig("output\exportingregionsDutch",bbox_inches='tight')
    plt.show()
# data = pd.read_csv("sumperregion.csv", index_col=[0])
# data.drop(columns=["Micronesia", "Melanesia", "Polynesia"], inplace=True)
# compareDutchimportperregion(data)

def regionalexportersdetail(STRATEGICGOODS):
    ## Northern Europe
    yr = 2022
    data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y"+ str(yr) + "_V202401.csv")
    data = bc1.addregions(data)
    data = bc1.subsetStrategicGoods(data, STRATEGICGOODS)
    data = data[data['Importer'] == "NLD"]
    data = bc1.addshortdescriptoProdname(data)
    data = data[['Value', 'Exporter_sub_region', 'code']]
    data = data[['Exporter_sub_region', 'Value', 'code']].groupby(['Exporter_sub_region', 'code']).sum()
    
    # strange code to fill first index with values
    data.to_csv("exprtperregionvalue.csv")
    data = pd.read_csv("exprtperregionvalue.csv")
    topthreeperregion = []
    for reg in data['Exporter_sub_region'].unique():
        data1 = data[data['Exporter_sub_region'] == reg]
        data1.sort_values('Value', ascending = False, inplace = True)
        topthreeperregion.append(data1.iloc[0:2,:])
        
    out1 = pd.concat(topthreeperregion).to_csv("tmp.csv")
    return None
#regionalexportersdetail(STRATEGICGOODS)
# relative to other countries, how much have imports of strategic goods increased through time.  To compare with other countrie, remove the effects of gdp
# first, get top trader importers in terms of total trade
# Or, just look at the trade of strategic goods as a proportion of total imports or gdp   

# for netherlands, group by subcategories, see though time
def dutchbreakdown(STRATEGICGOODS):
    yearsData = np.arange(1995, 2023, step=1)
    yearsdata = []
    for yr in yearsData:
        print(yr)
        data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y"+ str(yr) + "_V202401.csv")
        data = bc1.addregions(data)
        data = bc1.subsetStrategicGoods(data, STRATEGICGOODS)
        data = bc1.addshortdescriptoProdname(data)
        data = data[data['Importer'] == "NLD"]
        
        data = data[['Value', 'code']]
        grp1 = data.groupby(['code']).sum()
        grp = grp1.T
        grp['Year'] = yr
        yearsdata.append(grp)
    
    return yearsdata        
# dutchbycode = dutchbreakdown(STRATEGICGOODS)
# dutchbycode = pd.concat(dutchbycode)
# dutchbycode.set_index("Year", inplace = True)
# dutchbycode.to_csv("dutchbycode.csv")
# np.log(dutchbycode).plot()
# plt.show()

def plotdutchbycode():
    plt.rcParams['figure.figsize'] = 10, 6

    plotable = pd.read_csv("dutchbycode.csv", index_col=[0])
    #plotable['AverageAllProducts']  = plotable.mean(axis=1)
    print(plotable)

    n = plotable.shape[1]
    color = iter(cm.tab20(np.linspace(0, 1, n)))
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
        #plt.text(2022+.5, y.iloc[-1], namecolumn[-2:] + " " + str(y.iloc[-1])[0:4], fontsize = 6)
        if namecolumn != "zinc and articles thereof_79":
            plt.text(2022+.5, y.iloc[-1], namecolumn[-2:], fontsize = 6)
        else: plt.text(2022 + .5, y.iloc[-1] + .1, namecolumn[-2:], fontsize = 6)

    
    plt.tight_layout()
    plt.title("Log Value Dutch Imports per Strategic HS6 2-digit category)")
    plt.legend(loc="upper left",  fontsize="7") 
    plt.savefig("output\logvaluedutchimports",bbox_inches='tight')
#plotdutchbycode()

def toptraders2022():
    dt1 = bc1.readindata("C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y2022_V202401.csv")
    dt1 = dt1[['Value', 'Importer']]
    dt2 = dt1.groupby(['Importer']).sum()
    dt3 = dt2.sort_values(['Value'], ascending=False)
    
    return dt3.index

def regressiondata():
    toptraders = toptraders2022().tolist()
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
        strategic = bc1.subsetStrategicGoods(data, STRATEGICGOODS)
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

# augmentregressiondata()
# # Which countries have had higher than average 
# newdata = pd.read_csv('regressionDataAugmented.csv', index_col=[0])
# newdata['Year'] = newdata.index

def plotdifferences(whichcomparison, whichstates, newdata, logornot = False):
    plt.rcParams['figure.figsize'] = 10, 6

    subset1 = newdata[['Year', 'state', whichcomparison]]
    plotable = subset1.pivot(index = 'Year', columns = 'state', values = whichcomparison)
    plotable = plotable[whichstates]

    # Average should be for all countries
    plotable['AverageAllCountries'] =  newdata[['Year', whichcomparison]].groupby("Year").mean()
    
    n = plotable.shape[1]
    color = iter(cm.tab20(np.linspace(0, 1, n)))
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
        if logornot:
            y = np.log(plotable.iloc[:, i])
        else:
            y = plotable.iloc[:, i]
        plt.plot(x,y,c=c, linewidth = lw, linestyle = ls, label = namecolumn)
        plt.text(2022, y.iloc[-1], namecolumn + " " + str(y.iloc[-1])[0:4], fontsize = 6)

    plt.legend(loc="upper left")  
    plt.title(whichcomparison)
    plt.legend(loc="upper left",  fontsize="7") 
    plt.savefig("output\percentageofImports",bbox_inches='tight')
# whichcomparison = 'percentStratofGDP'
# whichcomparison = 'percentStratofImports'
# toptraders = toptraders2022().tolist()
# whichstates = toptraders[0:10]
# newdata = pd.read_csv('regressionDataAugmented.csv', index_col=[0])
# newdata['Year'] = newdata.index
# plotdifferences(whichcomparison, whichstates, newdata, logornot = False)

###################
#countries with highest/lowest average percentgdp
###################
def percentgdptopbottom():
    newdata = pd.read_csv('regressionDataAugmented.csv', index_col=[0])

    highestpercentGDP = newdata[['percentStratofGDP', 'state']].groupby('state').mean()
    highestpercentGDP.sort_values(['percentStratofGDP'], ascending = False, inplace=True)
    highestpercentGDP.dropna(inplace=True)
    statehighestpercentGDP = ['NLD'] + highestpercentGDP.index.tolist()[-5:-1] +  highestpercentGDP.index.tolist()[0:5]
    whichcomparison = 'percentStratofGDP'
    whichstates = statehighestpercentGDP
    plotdifferences(whichcomparison, whichstates, newdata)
#percentgdptopbottom()

def addcategorical(data: pd.DataFrame):
    repeatthese = ['highest', 'second', 'third', 'fourth', 'average', 'sixth', 'seventh', 'eighth', 'nineth', 'lowest']
    numrows = range(perstateGDP.shape[0])
    splits = np.array_split(np.array(numrows), len(repeatthese))
    repeaters = [len(numrows) for numrows in splits]
    perstateGDP['rankStratperGDP'] = pd.Series(repeatthese).repeat(repeaters).tolist()
    return perstateGDP
# newdata = pd.read_csv('regressionDataAugmented.csv', index_col=[0])
# perstateGDP = newdata[['state', 'percentStratofGDP']].groupby(['state']).sum()
# b1 = perstateGDP.sort_values(['percentStratofGDP'], ascending=False, inplace=True)
# b2 = addcategorical(b1)
# print(b2.groupby('rankStratperGDP').mean())

def percenttradetopbottom():
    newdata = pd.read_csv('regressionDataAugmented.csv', index_col=[0])
    highestpercentpercentStrat = newdata[['percentStratofImports', 'state']].groupby('state').mean()
    highestpercentpercentStrat.sort_values(['percentStratofImports'], ascending = False, inplace=True)
    highestpercentpercentStrat.dropna(inplace=True)
    statehighestpercentImports = ['NLD'] + highestpercentpercentStrat.index.tolist()[-5:] +  highestpercentpercentStrat.index.tolist()[0:5]
    whichcomparison = 'percentStratofImports'
    whichstates = statehighestpercentImports
    plotdifferences(whichcomparison, whichstates, newdata)
# percenttradetopbottom()
# newdata = pd.read_csv('regressionDataAugmented.csv', index_col=[0])
# newdata['Year'] = newdata.index

# # the idea, after having taken into account changes in gdp along with a time trend, how much has the purchase of strategic goods increased.
# # the residuals show us the change in purchases of strategic goods after having removed or reduced the effects of changes in a state's GDP.
# # Negative residuals don't imply that actually spending for strategic goods fell, only that purchases have fallen relative to GDP.  
# # If GDP stays the same or falls, while ...

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
# newdata = pd.read_csv('regressionDataAugmented.csv', index_col=[0])
# rg1, coef = regressiongdp(newdata)
# allresids = pd.concat(rg1)
# allcoef = pd.concat(coef)
# print(allresids)
# print(allcoef)
# allcoef.to_csv("allcoef.csv")
# allresids.to_csv("allresids.csv")

# allresids = pd.read_csv("allresids.csv", index_col=[0])

# which states have shown positive increases in percent
# for st in allresids['state'].unique():
    country = allresids[allresids['state'] == st]

    lastvalue = country.loc[2022, 'resids']
    if lastvalue < 0:
        print(st)
        # calc decrease
        lasttwo = country.loc[[2021,2022], "resids"]
        print(((lasttwo.iloc[1] - lasttwo.iloc[0])/lasttwo.iloc[0])*100)


def DutchThroughTime():
    print("Running Dutch through time")
    yearsData = np.arange(1995, 2023, step=1)
    yearsdata = []
    for yr in yearsData:
        print(yr)
        data = bc1.readindata(bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y"+ str(yr) + "_V202401.csv")
        data = bc1.subsetStrategicGoods(data, STRATEGICGOODS)
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
# data.plot()
# plt.show()

# data = bc1.add_gdp(data, GDP, '2022')

# bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_Y2022_V202401.csv"
# data = bc1.readindata(bacidata, tmp_save=False)
# data = bc1.subsetStrategicGoods(data, STRATEGICGOODS)
# data = bc1.addshortdescriptoProdname(data)
# data = bc1.strategicgoodExportingImportingregions(data, impexp = "Importer")
# strategicproducts = bc1.typesofstrategicgoods(data)

def barchartstrategicprod(strategicproducts):
    ax = strategicproducts[['Value']].plot.barh(stacked=False,  rot=0, cmap='tab20', figsize=(10, 7))
    ax.legend('best')
    plt.tight_layout()
    ax.set_title("Value of trade strategic goods by HS6 2-digit category)")
    plt.show()
    plt.savefig("output\ValueoftradestrategicgoodHS6",bbox_inches='tight')
#barchartstrategicprod(strategicproducts)