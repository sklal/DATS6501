# Libraries
import pandas as pd
partd_prescriber = pd.read_csv("D:\Capstone_py\Medicare_Provider_Utilization_and_Payment_Data__2017_Part_D_Prescriber.csv", usecols = ['npi','nppes_provider_city','nppes_provider_state', 'nppes_provider_last_org_name', 'nppes_provider_first_name', 'specialty_description','drug_name','generic_name','total_drug_cost','total_claim_count','total_day_supply'])
print(partd_prescriber.head())


#Drug Information

partD_drugs = partd_prescriber.loc[:,['npi','drug_name','total_drug_cost','total_claim_count','total_day_supply','specialty_description']]

partD_drugs['npi'] = partD_drugs.npi.astype(object)

# Provider Information

partD_providerinfo= partd_prescriber.loc[:,['npi','nppes_provider_city','nppes_provider_state', \
                                               'nppes_provider_last_org_name', \
                                               'nppes_provider_first_name','specialty_description']]

partD_providerinfo = partD_providerinfo.drop_duplicates()


print(partD_providerinfo.head())

rename_dict = {'nppes_provider_first_name':'first_name', 'nppes_provider_last_org_name':'last_name','nppes_provider_city':'city','nppes_provider_state':'state','specialty_description':'Specialty'}
partD_providerinfo = partD_providerinfo.rename(columns=rename_dict)

agg_dict = {'total_drug_cost':['sum','mean','max'], \
           'total_claim_count':['sum','mean','max'],\
           'total_day_supply':['sum','mean','max']}


#Specialty Specific Drug Details

partD_drug_spec = partD_drugs.groupby('specialty_description').agg(agg_dict).astype(float)
print(partD_drug_spec.head())

# NPI Drug details
partD_drug_npi = partD_drugs.groupby('npi').agg(agg_dict).astype(float)
print(partD_drug_npi.head())

#Leveled
level0 = partD_drug_npi.columns.get_level_values(0)
level1 = partD_drug_npi.columns.get_level_values(1)
partD_drug_npi.columns = level0 + '_' + level1
partD_drug_npi = partD_drug_npi.reset_index()

print(partD_drug_npi.head())

#Merging Drug npi details & providerinfo

partD_provider_npi = pd.merge(partD_drug_npi,partD_providerinfo, how ='left',on = 'npi')

print(partD_provider_npi.head())
