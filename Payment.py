from Prescriber import partD_provider_npi
import pandas as pd
payment_pd = pd.read_csv("D:\Capstone_py\OP_DTL_GNRL_PGYR2017_P01222021.csv",usecols = ['Physician_First_Name',\
                                             'Physician_Last_Name', \
                                             'Recipient_City', \
                                             'Recipient_State', \
                                             'Total_Amount_of_Payment_USDollars'])
rename_dict = {'Physician_First_Name':'first_name', 'Physician_Last_Name':'last_name','Recipient_City':'city','Recipient_State':'state','Total_Amount_of_Payment_USDollars':'Total_Payment'}
payment_pd = payment_pd.rename(columns=rename_dict)


print(payment_pd.head())

#Grouping Payment Provider Details

payment_pd = payment_pd.groupby(['first_name','last_name','city','state'])\
                                   .agg({'Total_Payment':['sum']}).astype(float)

level0 = payment_pd.columns.get_level_values(0)
level1 = payment_pd.columns.get_level_values(1)

payment_pd.columns = level0 + '_' + level1

payment_pd.reset_index()

print(payment_pd.head())

payment_pd = payment_pd.apply(lambda x: x.astype(str).str.upper())

payment_pd = payment_pd.sort_values('Total_Payment_sum',ascending=False)

partD_payment = pd.merge(partD_provider_npi,payment_pd, how ='left', on = ['last_name','first_name','city','state'])

print(partD_payment.head())

print(partD_payment.dtypes)
partD_payment['Total_Payment_sum'] = partD_payment['Total_Payment_sum'].astype(float)