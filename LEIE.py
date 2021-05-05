from Payment import partD_payment
import pandas as pd
LEIE = pd.read_csv("D:\Capstone_py\LEIE.csv",usecols = ['NPI','EXCLTYPE'])

npi_fraud = LEIE

npi_fraud = npi_fraud.query('NPI !=0')

rename_dict = {'NPI':'npi', 'EXCLTYPE':'is_fraud'}
npi_fraud = npi_fraud.rename(columns=rename_dict)

npi_fraud['is_fraud'] = 1

print(npi_fraud.head())

print(npi_fraud.dtypes)

Final = pd.merge(partD_payment,npi_fraud, how ='left',on = 'npi')

print(Final.head())