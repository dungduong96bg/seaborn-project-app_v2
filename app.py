app = pd.read_csv('/kaggle/input/credit-risk/data_final/application.csv')

app['NAME_CONTRACT_TYPE']=np.where(app['NAME_CONTRACT_TYPE']=='Cash loans',1,0)

app['CODE_GENDER']=np.where(app['CODE_GENDER']=='F',2,np.where(app['CODE_GENDER']=='M',1,0))
app['AMT_ANNUITY']=np.where(app['AMT_ANNUITY'].isna()==True,(app['AMT_CREDIT']/21).round(1),app['AMT_ANNUITY']) #most common term for cash loan is 21
app['TERM']=app['AMT_CREDIT']/app['AMT_ANNUITY'] #New var: term
app['AMT_GOODS_PRICE']=np.where(app['AMT_GOODS_PRICE'].isna()==True,app['AMT_CREDIT'],app['AMT_ANNUITY'])

app['FLAG_OWN_CAR']=np.where(app['FLAG_OWN_CAR']=='Y',1,0)
app['FLAG_OWN_REALTY']=np.where(app['FLAG_OWN_REALTY']=='Y',1,0)

WEEKDAY_APPR_PROCESS_START_condition=[app['WEEKDAY_APPR_PROCESS_START']=='MONDAY'
                           ,app['WEEKDAY_APPR_PROCESS_START']=='TUESDAY'
                           ,app['WEEKDAY_APPR_PROCESS_START']=='WEDNESDAY'
                           ,app['WEEKDAY_APPR_PROCESS_START']=='THURSDAY'
                           ,app['WEEKDAY_APPR_PROCESS_START']=='FRIDAY'
                           ,app['WEEKDAY_APPR_PROCESS_START']=='SATURDAY'
                           ,app['WEEKDAY_APPR_PROCESS_START']=='SUNDAY']
WEEKDAY_APPR_PROCESS_START_new=['2','3','4','5','6','7','8']
app['WEEKDAY_APPR_PROCESS_START']=np.select(WEEKDAY_APPR_PROCESS_START_condition,WEEKDAY_APPR_PROCESS_START_new,default=np.nan).astype(float)

NAME_TYPE_SUITE_condition=[app['NAME_TYPE_SUITE']=='Unaccompanied'
                           ,app['NAME_TYPE_SUITE']=='Other_A'
                           ,app['NAME_TYPE_SUITE']=='Other_B'
                           ,app['NAME_TYPE_SUITE']=='Spouse, partner'
                           ,app['NAME_TYPE_SUITE']=='Family'
                           ,app['NAME_TYPE_SUITE']=='Children'
                           ,app['NAME_TYPE_SUITE']=='Group of people']
NAME_TYPE_SUITE_new=['0','1','1','2','3','3','4']
app['NAME_TYPE_SUITE']=np.select(NAME_TYPE_SUITE_condition,NAME_TYPE_SUITE_new,default=np.nan).astype(float)
app['NAME_TYPE_SUITE'] = app['NAME_TYPE_SUITE'].fillna(0)

NAME_INCOME_TYPE_condition=[app['NAME_INCOME_TYPE']=='Working'
                           ,app['NAME_INCOME_TYPE']=='Unemployed'
                           ,app['NAME_INCOME_TYPE']=='Student'
                           ,app['NAME_INCOME_TYPE']=='State servant'
                           ,app['NAME_INCOME_TYPE']=='Pensioner'
                           ,app['NAME_INCOME_TYPE']=='Maternity leave'
                           ,app['NAME_INCOME_TYPE']=='Commercial associate'
                           ,app['NAME_INCOME_TYPE']=='Businessman'] #income type sort by average amt income of each types
NAME_INCOME_TYPE_new=['0','1','2','3','4','5','6','7']
app['NAME_INCOME_TYPE']=np.select(NAME_INCOME_TYPE_condition,NAME_INCOME_TYPE_new,default=np.nan).astype(float)

app['DAYS_EMPLOYED']=np.where(app['DAYS_EMPLOYED']==365243,1,app['DAYS_EMPLOYED']) #Unemployed & pensioner

NAME_EDUCATION_TYPE_condition=[app['NAME_EDUCATION_TYPE']=='Lower secondary'
                           ,app['NAME_EDUCATION_TYPE']=='Secondary / secondary special'
                           ,app['NAME_EDUCATION_TYPE']=='Academic degree'
                           ,app['NAME_EDUCATION_TYPE']=='Incomplete higher'
                           ,app['NAME_EDUCATION_TYPE']=='Higher education']
NAME_EDUCATION_TYPE_new=[0,1,2,3,4]
app['NAME_EDUCATION_TYPE']=np.select(NAME_EDUCATION_TYPE_condition,NAME_EDUCATION_TYPE_new,default=np.nan).astype(float)

NAME_FAMILY_STATUS_condition=[app['NAME_FAMILY_STATUS']=='Unknown'
                           ,app['NAME_FAMILY_STATUS']=='Single / not married'
                           ,app['NAME_FAMILY_STATUS']=='Married'
                           ,app['NAME_FAMILY_STATUS']=='Civil marriage'
                           ,app['NAME_FAMILY_STATUS']=='Separated'
                           ,app['NAME_FAMILY_STATUS']=='Widow'
                             ]
NAME_FAMILY_STATUS_new=[0,1,2,2,3,4]
app['NAME_FAMILY_STATUS']=np.select(NAME_FAMILY_STATUS_condition,NAME_FAMILY_STATUS_new,default=np.nan).astype(float)

NAME_HOUSING_TYPE_condition=[app['NAME_HOUSING_TYPE']=='With parents'
                           ,app['NAME_HOUSING_TYPE']=='Rented apartment'
                           ,app['NAME_HOUSING_TYPE']=='Municipal apartment'
                           ,app['NAME_HOUSING_TYPE']=='Office apartment'
                           ,app['NAME_HOUSING_TYPE']=='Co-op apartment'
                           ,app['NAME_HOUSING_TYPE']=='House / apartment'
                             ]
NAME_HOUSING_TYPE_new=[0,1,2,3,4,5,]

app['NAME_HOUSING_TYPE']=np.select(NAME_HOUSING_TYPE_condition,NAME_HOUSING_TYPE_new,default=np.nan).astype(float)
app['OWN_CAR_AGE'] = app['OWN_CAR_AGE'].fillna(-1)

app['OCCUPATION_TYPE'] = app['OCCUPATION_TYPE'].fillna('Unknown')
OCCUPATION_TYPE_condition=[app['OCCUPATION_TYPE']=='Cleaning staff'
                           ,app['OCCUPATION_TYPE']=='Low-skill Laborers'
                           ,app['OCCUPATION_TYPE']=='Cooking staff'
                           ,app['OCCUPATION_TYPE']=='Waiters/barmen staff'
                           ,app['OCCUPATION_TYPE']=='Security staff'
                           ,app['OCCUPATION_TYPE']=='Medicine staff'
                           ,app['OCCUPATION_TYPE']=='Sales staff'
                           ,app['OCCUPATION_TYPE']=='Unknown'
                           ,app['OCCUPATION_TYPE']=='Secretaries'
                           ,app['OCCUPATION_TYPE']=='Laborers'
                           ,app['OCCUPATION_TYPE']=='Core staff'
                           ,app['OCCUPATION_TYPE']=='Private service staff'
                           ,app['OCCUPATION_TYPE']=='High skill tech staff'
                           ,app['OCCUPATION_TYPE']=='Drivers'
                           ,app['OCCUPATION_TYPE']=='HR staff'
                           ,app['OCCUPATION_TYPE']=='Accountants'
                           ,app['OCCUPATION_TYPE']=='Realty agents'
                           ,app['OCCUPATION_TYPE']=='IT staff'
                           ,app['OCCUPATION_TYPE']=='Managers'] #ocupation type sort by average amt income of each types
OCCUPATION_TYPE_new=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
app['OCCUPATION_TYPE']=np.select(OCCUPATION_TYPE_condition,OCCUPATION_TYPE_new,default=np.nan).astype(float)

app['CNT_FAM_MEMBERS']=app['CNT_FAM_MEMBERS'].fillna(1)

app['EXT_SOURCE_1']=app['EXT_SOURCE_1'].fillna(0)
app['EXT_SOURCE_2']=app['EXT_SOURCE_2'].fillna(0)
app['EXT_SOURCE_3']=app['EXT_SOURCE_3'].fillna(0)
app['AVG_EXT_SOURCE']=(app['EXT_SOURCE_1']+app['EXT_SOURCE_2']+app['EXT_SOURCE_3'])/3 #New var: Average external score

FONDKAPREMONT_MODE_condition=[app['FONDKAPREMONT_MODE']=='org spec account'
                           ,app['FONDKAPREMONT_MODE']=='reg oper spec account'
                           ,app['FONDKAPREMONT_MODE']=='reg oper account'
                           ,app['FONDKAPREMONT_MODE']=='not specified'] #sort by bad rate
FONDKAPREMONT_MODE_new=['0','1','2','3']
app['FONDKAPREMONT_MODE']=np.select(FONDKAPREMONT_MODE_condition,FONDKAPREMONT_MODE_new,default=np.nan).astype(float)

HOUSETYPE_MODE_condition=[app['HOUSETYPE_MODE']=='block of flats'
                           ,app['HOUSETYPE_MODE']=='terraced house'
                           ,app['HOUSETYPE_MODE']=='specific housing'] #sort house area?
HOUSETYPE_MODE_new=['0','1','2']
app['HOUSETYPE_MODE']=np.select(HOUSETYPE_MODE_condition,HOUSETYPE_MODE_new,default=np.nan).astype(float)

WALLSMATERIAL_MODE_condition=[app['WALLSMATERIAL_MODE']=='Monolithic'
                           ,app['WALLSMATERIAL_MODE']=='Panel'
                           ,app['WALLSMATERIAL_MODE']=='Block'
                           ,app['WALLSMATERIAL_MODE']=='Stone, brick'
                           ,app['WALLSMATERIAL_MODE']=='Mixed'
                           ,app['WALLSMATERIAL_MODE']=='Others'
                           ,app['WALLSMATERIAL_MODE']=='Wooden'] #sort by bad rate
WALLSMATERIAL_MODE_new=['0','1','2','3','4','5','6']
app['WALLSMATERIAL_MODE']=np.select(WALLSMATERIAL_MODE_condition,WALLSMATERIAL_MODE_new,default=np.nan).astype(float)

app['EMERGENCYSTATE_MODE']=np.where(app['EMERGENCYSTATE_MODE']=='Yes',2,np.where(app['EMERGENCYSTATE_MODE']=='No',1,app['EMERGENCYSTATE_MODE']))
app['EMERGENCYSTATE_MODE']=app['EMERGENCYSTATE_MODE'].astype(float)

apartment_list = ['APARTMENTS_AVG',	'BASEMENTAREA_AVG',	'YEARS_BEGINEXPLUATATION_AVG',	'YEARS_BUILD_AVG',	'COMMONAREA_AVG',	'ELEVATORS_AVG',	'ENTRANCES_AVG',	'FLOORSMAX_AVG',	'FLOORSMIN_AVG',	'LANDAREA_AVG',	'LIVINGAPARTMENTS_AVG',	'LIVINGAREA_AVG',	'NONLIVINGAPARTMENTS_AVG',	'NONLIVINGAREA_AVG',	'APARTMENTS_MODE',	'BASEMENTAREA_MODE',	'YEARS_BEGINEXPLUATATION_MODE',	'YEARS_BUILD_MODE',	'COMMONAREA_MODE',	'ELEVATORS_MODE',	'ENTRANCES_MODE',	'FLOORSMAX_MODE',	'FLOORSMIN_MODE',	'LANDAREA_MODE',	'LIVINGAPARTMENTS_MODE',	'LIVINGAREA_MODE',	'NONLIVINGAPARTMENTS_MODE',	'NONLIVINGAREA_MODE',	'APARTMENTS_MEDI',	'BASEMENTAREA_MEDI',	'YEARS_BEGINEXPLUATATION_MEDI',	'YEARS_BUILD_MEDI',	'COMMONAREA_MEDI',	'ELEVATORS_MEDI',	'ENTRANCES_MEDI',	'FLOORSMAX_MEDI',	'FLOORSMIN_MEDI',	'LANDAREA_MEDI',	'LIVINGAPARTMENTS_MEDI',	'LIVINGAREA_MEDI',	'NONLIVINGAPARTMENTS_MEDI',	'NONLIVINGAREA_MEDI',	'FONDKAPREMONT_MODE',	'HOUSETYPE_MODE',	'TOTALAREA_MODE',	'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']

masked_appartment = app['NAME_HOUSING_TYPE'].isin([0, 1])
app.loc[masked_appartment,apartment_list] = app.loc[masked_appartment,apartment_list].fillna(-2)
app.loc[~masked_appartment,apartment_list] = app.loc[~masked_appartment,apartment_list].fillna(-1)

social_circle_list = ['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']
app.loc[:,social_circle_list] = app.loc[:,social_circle_list].fillna(-1)

app['DAYS_LAST_PHONE_CHANGE'] = app['DAYS_LAST_PHONE_CHANGE'].fillna(0)

AMT_REQ_list = ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']
app.loc[:,AMT_REQ_list] = app.loc[:,AMT_REQ_list].fillna(-1)

c=0
for val in sorted(app['ORGANIZATION_TYPE'].unique()):
    mask_ORGANIZATION_TYPE = app['ORGANIZATION_TYPE'] == val
    app.loc[mask_ORGANIZATION_TYPE,'ORGANIZATION_TYPE']=c
    c+=1
app['ORGANIZATION_TYPE']=app['ORGANIZATION_TYPE'].astype(int)

app['FULL_DOC'] = app[['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
         'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
         'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
         'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
         'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']].sum(axis=1)
