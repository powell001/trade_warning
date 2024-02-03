import pandas as pd
import os
import sys 
import numpy as np
os.chdir(r'C:\Users\jpark\VSCode\trade_warning\\')

from combine_country_regions import country_mappings
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# from config.defintions import ROOT_DIR
# DATA_DIR = os.path.join(ROOT_DIR, 'data\\')
# FIGURES_DIR = os.path.join(ROOT_DIR, 'data\\output\\forfigures\\')

desired_width=1000
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',10)
pd.options.display.max_rows = 50

class baci:
    def readindata(self, bacidata, tmp_save = True) -> pd.DataFrame:
        df1 = pd.read_csv(bacidata, usecols=['t','i','j','k','v','q'], 
                          dtype= {'t': 'int64',
                                  'i': 'int64', 
                                  'j': 'int64', 
                                  'k': 'int64',
                                  'v': 'float64',
                                  'q': 'object'}
                          )

        # This is too complicated, but '   NA' should be converted to float
        df1['q'] = df1['q'].apply(lambda x: x.strip())
        df1['q'].replace('NA', np.NaN, inplace=True)
        df1['q'] = df1['q'].astype(float)

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

        return df1

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

# ININTIALIZE object
bc1 = baci()

def GDPData():
    data = pd.read_csv(r"src\\baci\\data\\GDP_CurrentUSDollars.csv", index_col=[0])
    return data
gdp1 = GDPData()

def getStrategicGoods():
    data = pd.read_csv(r"src\\pdf_extractor\\strategicProducts.csv", index_col=[0])
    data = data.iloc[:,0].tolist()
    return data

STRATEGOODS = getStrategicGoods()

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

# data = allWorldImports()
# print(data.head())
# data = data.merge(gdp1, left_index=True, right_index=True)
# data['Percentage_Trade'] = ((data['trade_world_value'] * 2)/data['World']) * 100
# data['Percentage_Trade'].plot(title="World")
# data.to_csv("World.csv")
# plt.show()

# world = pd.read_csv("World.csv")
# print(world)

# world[['trade_world_value', 'trade_world_quantity', 'strategic_world_value', 'strategic_world_quantity']].plot()
# plt.show()

# are there countries in which trade of strategic goods has grown relative to their trade


def relativeIncreasePerCountry():
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
        
            # doesn't make sense/impossible to distinguish between imports and exports
            st_value = st['Value'].sum()
            st_quantity = st['Quantity'].sum()
            year  = st['Year'].tolist()[0]

            # subset on strategic goods
            st_strat = bc1.subsetStrategicGoods(st, STRATEGOODS)

            # doesn't make sense/impossible to distinguish between imports and exports
            st_strategic_value = st_strat['Value'].sum()
            st_strategic_quantity = st_strat['Quantity'].sum()
        
            allstates.append([year, j, st_value, st_quantity, st_strategic_value, st_strategic_quantity])     
                
        data = pd.DataFrame(allstates)
        data.columns = ['Year', 'state', 'state_value', 'state_quantity', 'strategic_state_value', 'strategic_state_quantity']
        data.set_index('Year', inplace=True)
    
        allyears.append(data)
    
    return allyears

# relativeCountry = relativeIncreasePerCountry()
# relativeCountry = pd.concat(relativeCountry)
# relativeCountry.to_csv("relativeCountry.csv")

df1 = pd.read_csv("xxxx.csv")
# print(df1)

# g1 = df1[['state', 'state_value', 'strategic_state_value']].groupby(['state']).mean()
# g1['state_nonstrategic'] = g1['state_value'] - g1['strategic_state_value']
# g1.sort_values(['state_value'], inplace=True)
# print(g1)

# g1 = g1.iloc[-24:,:]
# ax = g1[['strategic_state_value', 'state_nonstrategic']].plot.bar(stacked=True, rot=0, cmap='tab20', figsize=(10, 7))
# ax.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
# plt.tight_layout()
# ax.set_title("Value of Imports (Strategic vs Non-Strategic)")
# plt.show()

#############
# percentage
#############
g2 = df1[['state', 'state_value', 'strategic_state_value']].groupby(['state']).mean()
g2['percentage_strategic'] = g2['strategic_state_value']/g2['state_value']
g2['percentage_non_strategic'] = (g2['state_value'] - g2['strategic_state_value'])/g2['state_value']
g2.sort_values(['percentage_strategic'], inplace=True)
print(g2)

g2 = g2[g2['state_value'] >= 1e6]
g2 = g2.iloc[-24:,:]
ax = g2[['percentage_strategic', 'percentage_non_strategic']].plot.bar(stacked=True, rot=0, cmap='tab20', figsize=(10, 7))
ax.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
plt.tight_layout()
ax.set_title("Percentage of Strategic Imports (Strategic vs Non-Strategic), trade value >= 1e6")
plt.show()


def allDutchImports():
    yearsData = np.arange(1995, 2022, step=1)
    yearly = []
    for i in yearsData:
        print(i)
        bacidata = "C:\\Users\\jpark\\Downloads\\BACI_HS92_V202401\BACI_HS92_V202401\BACI_HS92_Y" + str(i) + "_V202401.csv"
        df1 = bc1.readindata(bacidata, tmp_save=True)
        
        # subset on strategic goods
        df1 = bc1.subsetStrategicGoods(df1, STRATEGOODS)

        df_imp = bc1.subsetData(df1, ["NLD"], 'Importer', [])
        imp_quant = df_imp['Quantity'].sum()
        imp_value = df_imp['Value'].sum()
        
        df_exp = bc1.subsetData(df1, ["NLD"], 'Exporter', [])
        exp_quant = df_exp['Quantity'].sum()
        exp_value = df_exp['Value'].sum()
        
        year  = df_imp['Year'].tolist()[0]
        yearly.append([year, imp_quant, imp_value, exp_quant, exp_value])

    data = pd.DataFrame(yearly)
    data.columns = ['Year', 'Imp_Quantity', 'Imp_Value',  'Exp_Quantity', 'Exp_Value']
    data.set_index('Year', inplace=True)

    return data

# data = allDutchImports()
# print(data.head())
# data = data.merge(gdp1, left_index=True, right_index=True)
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
    




#######################
# add regions
# iso_regions = pd.read_csv(r"data\iso_countries_regions.csv")
# iso_regions = iso_regions[['alpha-3', 'region']]
# data = data.merge(iso_regions, left_on="largestExporter", right_on="alpha-3", how="left")
# data.drop(columns=["alpha-3"], inplace=True)
########################
    

########################
# these are the strategic product codes from the OECD
# strategicProducts = pd.read_csv(r"src/pdf_extract/strategicProducts.csv", index_col=[0])
# strategicProducts.columns = ['ProdID']
# strategicData = data[data['ProdID'].isin(strategicProducts.loc[:,'ProdID'].tolist())]    
#########################



# df2 = bc1.subsetData(df1, ["NLD"], 'Importer', [121110])  
# print(df2)
#df1.to_csv("dataxxx.csv")
#df1 = pd.read_csv("tmpdata_20xx.csv")


    # def test_code():


    #
    # EU_Imports_total = oneProd[oneProd['Importer_ISO3'] == 'NLD']
    # EU_Imports_total.to_csv("NL_Imports_total.csv")

    # NLD_imports_Not_EU = EU_Imports_both_EU_notEU
    #
    #
    # # EU_imports sum(Not_EU) > sum(EU)
    #
    # # NLD_imports > NLD_exports
    #
    # # EU imports from EU and Not_EU
    # EU_Imports_both_EU_notEU = oneProd[oneProd['Importer_Region_EU'] == 'EU']
    # EU_Imports_both_EU_notEU.to_csv("EU_Imports_both_EU_notEU.csv")
    #
    # # From EU_Imports_both_EU_notEU, sum value coming from European countries
    # Imports_from_EU = EU_Imports_both_EU_notEU[EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'EU']
    # value_from_EU = Imports_from_EU['Value'].sum()
    # print("Import_Value_from_EU: ", value_from_EU)
    # Imports_from_EU.to_csv("Imports_from_EU.csv")
    #
    # # From EU_Imports_both_EU_notEU, sum value coming from NOT European countries
    # Imports_from_Not_EU = EU_Imports_both_EU_notEU[EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'Not_EU']
    # value_from_not_EU = Imports_from_Not_EU['Value'].sum()
    # print("Import_Value_from_Not_EU: ", value_from_not_EU)
    # Imports_from_Not_EU.to_csv("Imports_from_Not_EU.csv")
    #
    # # If value from Not_EU exporters greater than EU exporters, determine if the
    # # top 3 Not_EU exporters of that good consist of 50% or more of the value.
    # if value_from_not_EU > value_from_EU:
    #     print("Value of imports coming to the EU from outside the EU: ", value_from_not_EU)
    #     # Only need EU import fron Not_EU countries from above
    #     Imports_from_Not_EU.sort_values(by = ['Value'], ascending=False, inplace = True)
    #     print("Top exporters to EU: ", Imports_from_Not_EU)
    #
    #     # Sum values per Not_EU exporters
    #     topExportersToEU = Imports_from_Not_EU[['Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3']).sum()
    #     topExportersToEU.sort_values(by = ['Value'], ascending=False, inplace = True)
    #     print("totalvalue: ", topExportersToEU)
    #
    #     # Netherlands
    #     # What is the total from all Exporters to the Netherlands
    #     value_from_not_EU_to_NLD = Imports_from_Not_EU[Imports_from_Not_EU['Importer_ISO3'] == 'NLD']
    #     allExporter_value_from_Not_EU_to_NLD = value_from_not_EU_to_NLD['Value'].sum()
    #     print("allExporter_value_from_Not_EU_to_NLD:",  allExporter_value_from_Not_EU_to_NLD)
    #
    #     # What is total from top-three Not_EU exporters to the Netherlands:
    #     topExporters_value_from_Not_EU_to_NLD = value_from_not_EU_to_NLD['Value'].head(3).sum()
    #     print("topExporters_from_Not_EU_to_NLD:", topExporters_value_from_Not_EU_to_NLD)
    #
    #     # Check for the Netherlands
    #     if (topExporters_value_from_Not_EU_to_NLD/allExporter_value_from_Not_EU_to_NLD >= 0.5):
    #         print("Hey!, Netherlands you're vulnerable for this product: ", i)
    #
    #     # If imports greater than exports, vulnerable
    #     Dutch_Imports > NLD_Exports

    #             # Dutch Exports and Imports
    #
    #                 count = count + 1
    #
    # print("Number vulnerable products: ", count)


#################
#################

def step1_test(data: pd.DataFrame):

    print(data.head())

    prod = data['Product'].unique()
    count = 0
    #for each product
    for i in prod:
        if i == 811213:
            oneProd = data[data['Product'] == i]
            oneProd.to_csv("oneProd.csv")
            ##################################################
            # NLD_imports Not_EU.head(3).sum()/Not_EU.sum() > 0.50
            ##################################################
            NL_Imports_total = oneProd[oneProd['Importer_ISO3'] == 'NLD']
            NL_Imports_total.to_csv("NL_Imports_total.csv")

            NL_sum_top3_Not_EU  = NL_Imports_total['Value'][NL_Imports_total['Exporter_Region_EU'] == 'Not_EU'].head(3).sum()
            print(NL_sum_top3_Not_EU)
            NL_sum_total_Not_EU = NL_Imports_total['Value'][NL_Imports_total['Exporter_Region_EU'] == 'Not_EU'].sum()
            print(NL_sum_total_Not_EU)

            NL_More_than_50_percent = (NL_sum_top3_Not_EU/NL_sum_total_Not_EU) > 0.50
            print(NL_More_than_50_percent)

            ##################################################
            # EU_imports Not_EU.sum() > EU_imports EU.sum()
            ##################################################
            EU_Imports_both_EU_notEU = oneProd[oneProd['Importer_Region_EU'] == 'EU']
            EU_Imports_both_EU_notEU.to_csv("EU_Imports_both_EU_notEU.csv")

            EU_imports_from_Not_EU = EU_Imports_both_EU_notEU['Value'][EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'Not_EU'].sum()
            EU_imports_from_EU = EU_Imports_both_EU_notEU['Value'][EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'EU'].sum()

            EU_Imports_NotEU_gt_EU_Imports_EU = EU_imports_from_Not_EU > EU_imports_from_EU
            print(EU_Imports_NotEU_gt_EU_Imports_EU)

            ##################################################
            # NLD_Imports > NLD_Exports
            ##################################################
            NL_Imports_total = oneProd['Value'][oneProd['Importer_ISO3'] == 'NLD'].sum()
            NL_Exports_total = oneProd['Value'][oneProd['Exporter_ISO3'] == 'NLD'].sum()
            NL_Imports_gt_NL_Exports = NL_Imports_total > NL_Exports_total

            print(NL_Imports_gt_NL_Exports)

            ######################
            if (NL_More_than_50_percent & EU_Imports_NotEU_gt_EU_Imports_EU & NL_Imports_gt_NL_Exports):
                print("Hey!, Netherlands you're vulnerable for this product: ", i)


#step1_test(df1)

def step1(data: pd.DataFrame):
    prod = data['Product'].unique()
    count = 0
    # for each product
    for i in prod:

        oneProd = data[data['Product'] == i]
        ##################################################
        # NLD_imports Not_EU.head(3).sum()/Not_EU.sum() > 0.50
        ##################################################
        NL_Imports_total = oneProd[oneProd['Importer_ISO3'] == 'NLD']
        NL_Imports_total.to_csv("NL_Imports_total.csv")

        NL_sum_top3_Not_EU = NL_Imports_total['Value'][NL_Imports_total['Exporter_Region_EU'] == 'Not_EU'].head(
            3).sum()
        NL_sum_total_Not_EU = NL_Imports_total['Value'][NL_Imports_total['Exporter_Region_EU'] == 'Not_EU'].sum()

        NL_More_than_50_percent = (NL_sum_top3_Not_EU / NL_sum_total_Not_EU) > 0.50

        ##################################################
        # EU_imports Not_EU.sum() > EU_imports EU.sum()
        ##################################################
        EU_Imports_both_EU_notEU = oneProd[oneProd['Importer_Region_EU'] == 'EU']
        EU_Imports_both_EU_notEU.to_csv("EU_Imports_both_EU_notEU.csv")

        EU_imports_from_Not_EU = EU_Imports_both_EU_notEU['Value'][
            EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'Not_EU'].sum()
        EU_imports_from_EU = EU_Imports_both_EU_notEU['Value'][
            EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'EU'].sum()

        EU_Imports_NotEU_gt_EU_Imports_EU = EU_imports_from_Not_EU > EU_imports_from_EU

        ##################################################
        # NLD_Imports > NLD_Exports
        ##################################################
        NL_Imports_total = oneProd['Value'][oneProd['Importer_ISO3'] == 'NLD'].sum()
        NL_Exports_total = oneProd['Value'][oneProd['Exporter_ISO3'] == 'NLD'].sum()
        NL_Imports_gt_NL_Exports = NL_Imports_total > NL_Exports_total

        ######################
        if (NL_More_than_50_percent & EU_Imports_NotEU_gt_EU_Imports_EU & NL_Imports_gt_NL_Exports):
            print("Hey Netherlands!, you're vulnerable for this product: ", i, " totalEUImports: ", oneProd['Value'][oneProd['Importer_Region_EU'] == 'EU'].sum())
            count = count + 1

    print("Total count: ", count, "Percent: ", count/len(prod))

#step1(df1)



def step1_original(data: pd.DataFrame):

    print(data.head())

    prod = data['Product'].unique()
    count = 0
    #for each product
    for i in prod:

        oneProd = data[data['Product'] == i]

        # does the EU import more from the EU or Non-EU?

        # EU imports from EU and Not_EU
        EU_Imports_both_EU_notEU = oneProd[oneProd['Importer_Region_EU'] == 'EU']
        EU_Imports_both_EU_notEU.to_csv("EU_Imports_both_EU_notEU.csv")

        # From EU_Imports_both_EU_notEU, sum value coming from European countries
        Imports_from_EU = EU_Imports_both_EU_notEU[EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'EU']
        value_from_EU = Imports_from_EU['Value'].sum()
        #print("Import_Value_from_EU: ", value_from_EU)
        Imports_from_EU.to_csv("Imports_from_EU.csv")

        # From EU_Imports_both_EU_notEU, sum value coming from NOT European countries
        Imports_from_Not_EU = EU_Imports_both_EU_notEU[EU_Imports_both_EU_notEU['Exporter_Region_EU'] == 'Not_EU']
        value_from_not_EU = Imports_from_Not_EU['Value'].sum()
        #print("Import_Value_from_Not_EU: ", value_from_not_EU)
        Imports_from_Not_EU.to_csv("Imports_from_Not_EU.csv")

        # If valuue from Not_EU exporters greater than EU exporters, determine if the
        # top 3 Not_EU exporters of that good consist of 50% or more of the value.
        if value_from_not_EU > value_from_EU:
            #print("Value of imports coming to the EU from outside the EU: ", value_from_not_EU)
            # Only need EU import fron Not_EU countries from above
            Imports_from_Not_EU.sort_values(by = ['Value'], ascending=False, inplace = True)
            #print("Top exporters to EU: ", Imports_from_Not_EU)

            # Sum values per Not_EU exporters
            topExportersToEU = Imports_from_Not_EU[['Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3']).sum()
            topExportersToEU.sort_values(by = ['Value'], ascending=False, inplace = True)
            #print("totalvalue: ", topExportersToEU)

            # What is the total from all Exporters
            allExportersValue = topExportersToEU['Value'].sum()
            #print("allExportersValue:",  allExportersValue)

            # What is total from top-three exporters:
            topExporters = topExportersToEU['Value'].head(3).sum()
            #print("topExportersValue:", topExporters)

            if topExporters/allExportersValue >= 0.5:
                print("Hey!, EU you're vulnerable for this product: ", i)
                print("Total values of imports: ", allExportersValue)

                count = count + 1

    print("Number vulnerable products: ", count)


#step1(df1)


##### test BACI data
import numpy as np

def test_baci():
    df1 = pd.read_csv(DATA_DIR + 'BACI_HS17_Y2021_V202401.csv', usecols=['t', 'i', 'j', 'k', 'v', 'q'])
    numberrows = df1.shape[0]; print(numberrows)

    # df1 = pd.read_csv(DATA_DIR + 'BACI_HS17_Y2017_V202401.csv', usecols=['i', 'j', 'k', 'v'])

    # rename columns to make them meaningful
    df1.rename(columns={'t': 'Year', 'i': 'Exporter', 'j': 'Importer', 'k': 'Product', 'v': 'Value', 'q': 'Quantity'}, inplace=True)

    # NAs recorded in a strange manner for Quantities
    df1['Quantity'] = df1['Quantity'].apply(lambda x: x.strip())
    df1 = df1.replace('NA', 0.0)
    df1['Quantity'] = df1['Quantity'].astype(float)

    return(df1)

def how_many_eu_counries(data):
    g1 = data[data['Exporter_Region_EU'] == 'Not_EU']
    print(g1['Exporter_ISO3'].unique())

    g2 = data[data['Exporter_Region_EU'] == 'EU']
    print(g2['Exporter_ISO3'].unique())

# how_many_eu_counries(df1)

def dutch_imports_from_allover(data):
    g1 = data[data['Importer_ISO3'] == 'NLD']

    #selected columns
    subset = ["Product", "Value", "Exporter_ISO3", "Importer_ISO3", "Importer_Region_EU"]
    g2 = g1[subset]
    print(g2)

    #import value per country
    sum_all_countries = g2[['Value', 'Exporter_ISO3']].groupby(['Exporter_ISO3']).sum()
    print(sum_all_countries)

    dutch_imports_from = sum_all_countries.sort_values(by = ['Value'], ascending=False)
    dutch_imports_from['Percentage'] = (dutch_imports_from['Value']/dutch_imports_from['Value'].sum()) * 100
    dutch_imports_from.rename(columns={'Value': "Dutch_Imports_from:"}, inplace = True)
    print(dutch_imports_from)

    return dutch_imports_from

#dutch_imports_from_allover(df1)

def dutch_export_to_allover(data):
    g1 = data[data['Exporter_ISO3'] == 'NLD']

    #selected columns
    subset = ["Product", "Value", "Exporter_ISO3", "Importer_ISO3", "Importer_Region_EU"]
    g2 = g1[subset]
    print(g2)

    #exports to each country
    sum_all_countries = g2[['Value', 'Importer_ISO3']].groupby(['Importer_ISO3']).sum()
    print(sum_all_countries)

    dutch_exports_to = sum_all_countries.sort_values(by = ['Value'], ascending=False)
    dutch_exports_to['Percentage'] = (dutch_exports_to['Value']/dutch_exports_to['Value'].sum()) * 100
    dutch_exports_to.rename(columns={'Value': "Dutch_Exports_to:"}, inplace=True)

    print(dutch_exports_to)

    return dutch_exports_to

#dutch_export_to_allover(df1)

def dutch_imports_per_product(data):
    # from all countries
    g1 = data[data['Importer_ISO3'] == 'NLD']

    # selected columns
    subset = ["Product", "Value", "Exporter_ISO3", "Importer_ISO3", "Exporter_Region_EU"]
    g2 = g1[subset]
    print(g2)

    # EU vs Non-EU
    EU_exporters_toNLD = g2[g2['Exporter_Region_EU'] == 'EU']
    EU_exporters_toNLD.drop(columns = ['Importer_ISO3', 'Exporter_Region_EU'], inplace=True)
    From_EU_Exps = EU_exporters_toNLD[['Product', 'Value']].groupby("Product").sum()
    From_EU_Exps.rename(columns = {"Value": "Exp_From_EU_Countries"}, inplace = True)
    print(From_EU_Exps)

    Not_EU_exporters_toNLD = g2[g2['Exporter_Region_EU'] == 'Not_EU']
    Not_EU_exporters_toNLD.drop(columns=['Importer_ISO3', 'Exporter_Region_EU'], inplace=True)
    Not_From_EU_Exps = Not_EU_exporters_toNLD[['Product', 'Value']].groupby("Product").sum()
    Not_From_EU_Exps.rename(columns={"Value": "Exp_From_Non_EU_Countries"}, inplace = True)
    print(Not_From_EU_Exps)

    # Join
    result = pd.merge(From_EU_Exps, Not_From_EU_Exps, on="Product", how = 'outer')

    result['Non_EU_div_EU_3x'] =  result["Exp_From_Non_EU_Countries"] / result["Exp_From_EU_Countries"]


    print(result)

    return result

def attach_product_codes(data):
    dt1 = dutch_imports_per_product(data)

    # attach name of product
    codes = pd.read_csv(DATA_DIR + 'product_codes_HS17_V202401.csv')

    codes["code"] = codes[["code"]].astype(int)

    codes.rename(columns = {"code": "Product"}, inplace = True)
    result = pd.merge(dt1, codes, on="Product", how='outer')
    print(result)

    result.to_csv("perEu_NonEU.csv")

    ###################
    print('#########################')
    totalImports = result[['Product', 'Exp_From_EU_Countries', 'Exp_From_Non_EU_Countries', 'description']]
    totalImports['Imports'] = totalImports['Exp_From_EU_Countries'] + totalImports['Exp_From_Non_EU_Countries']
    totalImports.drop(columns = ['Exp_From_EU_Countries', 'Exp_From_Non_EU_Countries'], inplace=True)
    totalImports.to_csv('totalImportsEUandNonEU.csv')


#attach_product_codes(df1)

#dutch_imports_per_product(df1)


#df1 = test_baci()

# print(df1)
# print(df1.describe())
# print(df1.sum(axis = 0))
# print(len(df1['Product'].unique()))

################
def unique_country_product_combinations(data):
    data = data[data['Value'] > 0]
    data['combinations']  = data['Exporter'].apply(str) + data['Importer'].apply(str)
    print(len(data['combinations'].unique()))

#unique_country_product_combinations(df1)


#products = df1.loc[(df1['Product'] >= 1e5) & (df1['Product'] <= 1e6)]

#
# #products = df1.loc[(df1['Product'] <= 1e5) | (df1['Product'] >= 1e6)]
# print(len(products['Product'].unique()))
# print(products)


# df1_products = df1[len(df1.loc[:, 'Product'])== 6 ]
# print(df1_products.groupby(df1_products["Product"]).count())


def runthroughsteps_rawoutpercountry(imp_exp_param, steps_param):

    allsteps_expimp = []
    for enum, i in enumerate(steps_param):
        print(enum)
        if enum < 1000:
            print('**********', i , '*********')
            #print(step_strings[enum])

            prods = i
            b = baci()
            data, baci_countries = b.readindata()
            baci_countries.to_csv("tmp1010.csv")
            allExporters = data['Exporter_ISO3'].unique()
            subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)

            #### Now we just need to sum Exports per country per product
            exportValue = subset_data[['Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3']).sum()
            importerValue = subset_data[['Importer_ISO3', 'Value']].groupby(['Importer_ISO3']).sum()

            df2 = pd.merge(exportValue, importerValue, left_index=True, right_index=True, how = 'outer', suffixes=('_exports', '_imports'))
            df2 = df2.fillna(0)
            print("df2: ", df2)
            df2['Avg'] = df2.mean(axis = 1)
            df3 = df2['Avg']
            print("df3: ", df3)

            allsteps_expimp.append(df3)

    return allsteps_expimp

# # 'Exporter_ISO3', 'Importer_ISO3'
# allstepsExpImp = runthroughsteps_rawoutpercountry(imp_exp_param = "Exporter_ISO3", steps_param = allSteps)
# print(allstepsExpImp)
# from functools import reduce
# df_merged = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True, how = 'outer'), allstepsExpImp)
# df_merged = df_merged.fillna(0)
# print(df_merged)
#
# #
# print(df_merged.sum(axis=1))
# exp_plus_imp = df_merged.sum(axis=1)
# print(exp_plus_imp)
# exp_plus_imp = exp_plus_imp * 1000
# exp_plus_imp.to_csv("exp_plus_imp.csv")
# exp_plus_imp = pd.read_csv("exp_plus_imp.csv", index_col=[0])
# print(exp_plus_imp)
#
# ##################
# #Link in gdp data
# ##################
# gdp = pd.read_csv(DATA_DIR + "GDP_2021_Nominal.csv", index_col=[0], header=None)
# gdp.columns = ['gdp']
# gdp = gdp.fillna(0)
# gdp = gdp * 1000000
# gdp.to_csv("gdp.csv")
#
# df2 = pd.merge(exp_plus_imp, gdp, left_index=True, right_index=True, how='outer')
# print(df2)
# df2.columns = ['exp_plus_imp', 'gdp']
# df2['percent'] = (df2['exp_plus_imp']/df2['gdp']) * 100
# df2.to_csv(DATA_DIR + "Paper_Percent_elec_gdp.csv")

################
# Merge country name back in
################

def runthroughsteps(imp_exp_param, steps_param):

    kan_csv = []
    for enum, i in enumerate(steps_param):

        print('**********', i , '*********')
        #print(step_strings[enum])

        prods = i
        b = baci()
        data, baci_countries = b.readindata()
        baci_countries.to_csv("tmp1010.csv")
        allExporters = data['Exporter_ISO3'].unique()
        subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
        out = b.OECD_agg(subset_data, baci_countries, imp_exp_param = imp_exp_param)

        forKan = out[['OECD', 'Percentage']]
        forKan['step'] = step_strings[enum]

        kan_csv.append(forKan)

        out.to_csv(FIGURES_DIR + step_strings[enum] + "_" + imp_exp_param + "_"  + ".csv", index = False)

    return pd.concat(kan_csv).to_csv(("kan.csv"))

#'Exporter_ISO3', 'Importer_ISO3'
#allstepsExp = runthroughsteps(imp_exp_param = "Exporter_ISO3", steps_param = allSteps)

##########################################################################################################################################

#######################
# Where to final consumer and industry exports go???
#######################

def destination_finalConsumerGoods(imp_exp_param, steps_param):

    for enum, i in enumerate(steps_param):
        print('**********', i, '*********')
        print(step_strings[enum])

        prods = i
        b = baci()
        data, baci_countries = b.readindata()

        allExporters = data['Exporter_ISO3'].unique()
        subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param,
                                   products_param=prods, minvalue_param=0)

        subset_data = subset_data[['Exporter_ISO3', 'Importer_ISO3', 'Value']]
        china = subset_data[subset_data['Exporter_ISO3'] == 'CHN']

        grp = china.groupby(['Exporter_ISO3', 'Importer_ISO3']).sum()
        grp.sort_values(by = 'Value', ascending=False, inplace=True)

        print(grp)

    grp.to_excel("test_taiwanExports_Final_Consumer_Products.xlsx")


    return subset_data

# Exporter_ISO3', 'Importer_ISO3'
#allstepsExp = destination_finalConsumerGoods(imp_exp_param = "Exporter_ISO3", steps_param = [step4_output_final_consumer])

def imports_UAE_china(imp_exp_param, steps_param):
    b = baci()
    data, baci_countries = b.readindata()

    allExporters = data['Exporter_ISO3'].unique()
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param,
                               products_param=steps_param, minvalue_param=0)

    subset_data = subset_data[['Exporter_ISO3', 'Importer_ISO3', 'Value']]
    china = subset_data[subset_data['Exporter_ISO3'] == 'CHN']

    grp = china.groupby(['Exporter_ISO3', 'Importer_ISO3']).sum()
    grp.sort_values(by = 'Value', ascending=False, inplace=True)

    print(grp)

    grp.to_excel("UAE_Imports_finalConsumer.xlsx")


    return subset_data

# Exporter_ISO3', 'Importer_ISO3'
# imports_UAE_china(imp_exp_param = "Importer_ISO3", steps_param = step4_output_final_consumer)



def destination_rawmaterial_exports(imp_exp_param, steps_param):

    for enum, i in enumerate(steps_param):
        print('**********', i, '*********')
        print(step_strings[enum])

        prods = i
        b = baci()
        data, baci_countries = b.readindata()

        allExporters = data['Exporter_ISO3'].unique()
        subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param,
                                   products_param=prods, minvalue_param=0)

        subset_data = subset_data[['Exporter_ISO3', 'Importer_ISO3', 'Product', 'Value']]

        #grp = subset_data.groupby(['Product', 'Exporter_ISO3', 'Importer_ISO3']).sum()

    for prd in subset_data['Product'].unique():
        prdx = subset_data[subset_data['Product'] == prd]
        prdx.drop(columns=['Product'], inplace=True)
        grp = prdx.groupby(['Exporter_ISO3']).sum()
        grp.sort_values(by = 'Value', ascending=False, inplace=True)
        grp['Product'] = prd
        grp = grp[grp['Value'] > 0]
        print(grp)

        grp.to_excel(f"Exports_Raw_Materials_{prd}.xlsx")


    return subset_data

# Exporter_ISO3', 'Importer_ISO3'
# allstepsExp = destination_rawmaterial_exports(imp_exp_param = "Exporter_ISO3", steps_param = [step1_rawmaterials])



######################
# where to step 1 raw materials come from
######################



def Lizetable_allimporters(imp_exp_param, products_param):

    prods = products_param

    b = baci()
    data, baci_countries = b.readindata()

    allExporters = data['Exporter_ISO3'].unique()

    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    subset_data = subset_data[['Value', 'Exporter_OECD', 'Importer_OECD']]

    print(subset_data)
    exporters = subset_data["Exporter_OECD"].unique()
    forCols  = exporters.tolist()
    #forCols.append('Total')

    allExporters = []
    for exp in exporters:
        print("Exporter: ", exp)
        exp_region = subset_data[subset_data['Exporter_OECD'] == exp]
        exp_region.drop(columns = ['Exporter_OECD'], inplace = True)
        grp = exp_region.groupby(['Importer_OECD']).sum().T
        grp[exp] = 0
        #grp['Total'] = grp.sum(axis = 1)
        grp = grp[forCols]
        allExporters.append(grp)

    allExporters = pd.concat(allExporters)

    return allExporters, forCols


# a1, forCols = Lizetable_allimporters(imp_exp_param = "Importer_ISO3", products_param = AllProducts)
# a1.index = forCols
# out1 = a1 * 1000
# out1.to_csv("lizekan_allproducts.csv")


def Lizetable_allimporters_row(imp_exp_param, products_param):

    prods = products_param

    b = baci()
    data, baci_countries = b.readindata()

    allExporters = data['Exporter_ISO3'].unique()

    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    subset_data = subset_data[['Value', 'Exporter_ISO3', 'Importer_ISO3']]

    print(subset_data)
    exporters = subset_data["Exporter_ISO3"].unique()
    forCols  = exporters.tolist()
    #forCols.append('Total')

    allExporters = []
    for exp in exporters:
        print("Exporter: ", exp)
        exp_region = subset_data[subset_data['Exporter_ISO3'] == exp]
        exp_region.drop(columns = ['Exporter_ISO3'], inplace = True)
        grp = exp_region.groupby(['Importer_ISO3']).sum().T
        print(grp)
        grp[exp] = 0
        #grp['Total'] = grp.sum(axis = 1)
        #grp = grp[forCols]
        allExporters.append(grp)

    allExporters = pd.concat(allExporters)

    return allExporters, forCols


def forpaper():

    a1, forCols = Lizetable_allimporters_row(imp_exp_param = "Importer_ISO3", products_param = AllProducts)
    a1.index = forCols

    print(a1.shape)

    df2 = pd.DataFrame(a1.sum())
    df2.columns = ["Total"]
    print(df2)

    df2.T.columns = a1.columns

    print(df2.shape)
    a3 = a1.append(df2.T)

    a3_total = a3.T
    a3_total.sort_values(by=['Total'], ascending = False,  inplace = True)
    a4 = a3_total.T

    # # out1 = a1 * 1000
    a4.to_csv("test2_lizekan_allproducts.csv")
    a4.to_excel("test2_lizekan_allproducts.xlsx")


def runthroughsteps_Netherlands(imp_exp_param, steps_param):

    for enum, i in enumerate(steps_param):
        print('**********', i , '*********')

        prods = i
        b = baci()
        data, baci_countries = b.readindata()
        baci_countries.to_csv("tmp1010.csv")
        #allExporters = data['Exporter_ISO3'].unique()
        allExporters = ["NLD"]
        subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)

        print(subset_data)

        if imp_exp_param == 'Importer_ISO3':
            nlData = subset_data[['Value', 'Exporter_ISO3']]
            out1 = nlData.groupby(['Exporter_ISO3']).sum().reset_index()
            out1 = out1[out1['Value'] > 0]

        if imp_exp_param == 'Exporter_ISO3':
            nlData = subset_data[['Value', 'Importer_ISO3']]
            out1 = nlData.groupby(['Importer_ISO3']).sum().reset_index()
            out1 = out1[out1['Value'] > 0]

        out1.to_csv(FIGURES_DIR + step_strings[enum] + "_"  +  "NLD_as_" + imp_exp_param + "_" + ".csv", index=False)


#'Exporter_ISO3', 'Importer_ISO3'
#runthroughsteps_Netherlands(imp_exp_param = "Exporter_ISO3", steps_param = allSteps)




def kanlize_export_8540():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = list(range(854100, 854300))
    imp_exp_param = "Exporter_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("Prod_854100_854400_exports.csv", index = False)
    g = subset_data[['Product', 'Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    export = pd.concat(allprods)
    exports = export[export['Value'] > 0]
    exports['Country_Prod'] = exports['Exporter_ISO3'] + exports['Product'].apply(str)

    exports.to_csv("854100_854400_exporters.csv", index = False)

    print(exports)

    g2 = subset_data[['Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3']).sum().reset_index()
    g2.to_csv("854100_854400_exporters_summed.csv", index = False)

    return exports

#kanlize_export_8540 = kanlize_export_8540()

def kanlize_imports_8540():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = list(range(854100, 854300))
    imp_exp_param = "Importer_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("Prod_854100_854400_imports.csv", index = False)
    g = subset_data[['Product', 'Importer_ISO3', 'Value']].groupby(['Importer_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    imports = pd.concat(allprods)
    imports = imports[imports['Value'] >= 0]
    imports['Country_Prod'] = imports['Importer_ISO3'] + imports['Product'].apply(str)
    imports.to_csv("854100_854400_importers.csv", index = False)

    g2 = subset_data[['Importer_ISO3', 'Value']].groupby(['Importer_ISO3']).sum().reset_index()
    g2.to_csv("854100_854400_importers_summed.csv", index = False)


    return imports

#kanlize_imports_8540 = kanlize_imports_8540()

def mergeExpImp(exports_param, imports_param):
    merged1 = exports_param.merge(imports_param, left_on = 'Country_Prod', right_on = 'Country_Prod', how = 'inner')
    merge1 = merged1.rename(columns = {"Product_x": "Product", "Value_x": "Value_exports", "Value_y": "Value_imports"})
    merge1.drop(columns = ['Product_y'], inplace = True)
    merge1.to_csv("merged_8540.csv")

#mergeExpImp(kanlize_export_8540, kanlize_imports_8540)


def kanlize_rawmaterials_exports():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = step1_rawmaterials
    imp_exp_param = "Exporter_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("Rawmaterials_exports_all.csv", index = False)
    g = subset_data[['Product', 'Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    export = pd.concat(allprods)
    exports = export[export['Value'] > 0]
    exports.to_csv("RawMaterials_exporters.csv", index = False)
    print(exports)

#kanlize_rawmaterials_exports()

def kanlize_rawmaterials_imports():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = step1_rawmaterials
    imp_exp_param = "Importer_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("Rawmaterials_imports_all.csv", index = False)
    g = subset_data[['Product', 'Importer_ISO3', 'Value']].groupby(['Importer_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    imports = pd.concat(allprods)
    imports = imports[imports['Value'] > 0]
    imports.to_csv("RawMaterials_imports.csv", index = False)
    print(imports)

#kanlize_rawmaterials_imports()


def kanlize_381800_exports():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = list(range(381800, 381900))
    imp_exp_param = "Exporter_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("381800_exports_all.csv", index = False)
    g = subset_data[['Product', 'Exporter_ISO3', 'Value']].groupby(['Exporter_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    export = pd.concat(allprods)
    exports = export[export['Value'] > 0]
    exports.to_csv("381800_exporters.csv", index = False)
    print(exports)

#kanlize_381800_exports()

def kanlize_381800_imports():

    b = baci()
    data, baci_countries = b.readindata()
    allExporters = data['Exporter_ISO3'].unique()
    prods = list(range(381800, 381900))
    imp_exp_param = "Importer_ISO3"
    subset_data = b.subsetData(data_param=data, iso3_param=allExporters, imp_exp_param=imp_exp_param, products_param=prods, minvalue_param=0)
    print(subset_data)
    subset_data.to_csv("381800_imports_all.csv", index = False)
    g = subset_data[['Product', 'Importer_ISO3', 'Value']].groupby(['Importer_ISO3', 'Product']).sum().reset_index()

    allprods = []
    for p in g['Product'].unique():
        prod = g[g['Product'] == p]
        prod.sort_values(by=['Value'], ascending=False, inplace=True)
        allprods.append(prod)

    imports = pd.concat(allprods)
    imports = imports[imports['Value'] > 0]
    imports.to_csv("381800_imports.csv", index = False)
    print(imports)

#kanlize_381800_imports()


def plotfigures(imp_exp_param):


    forprint = []
    for i in step_strings:

        if imp_exp_param == "Importer_ISO3":
            i = i + "_Importer_ISO3_"
        else:
            i = i + "_Exporter_ISO3_"

        print(i)
        data1 = pd.read_csv(FIGURES_DIR + i + '.csv', usecols=['OECD', 'Percentage'])
        data1.sort_values(['OECD'], inplace=True)

        data10 = pd.DataFrame(columns=data1['OECD'])
        data10.loc[0] = data1['Percentage'].values

        forprint.append(data10)


    figdata = pd.concat(forprint)
    figdata["Stages"] = step_strings

    plt.rc('ytick', labelsize=6)

    figdata.plot(kind = 'barh', stacked = True, x = "Stages")
    plt.yticks(rotation=0)
    plt.title(imp_exp_param)
    plt.legend(prop={'size': 6}, loc='upper center', bbox_to_anchor=(1, 0.5))
    plt.show()

#'Exporter_ISO3', 'Importer_ISO3'
#plotfigures("Exporter_ISO3")