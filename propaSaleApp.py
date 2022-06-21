import pandas as pd  # pip install pandas openpyxl
import numpy as np
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import requests
import io
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import matplotlib.style as style
from scipy.stats import skew
# machine learning libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
#---------------------------------#
st.set_page_config(page_title="property Sales Machine Learning App", page_icon=":bar_chart:", layout="wide")

st.write("""
# The Machine Learning App
In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
Try adjusting the hyperparameters!
""")


# ---- READ EXCEL ----
#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your training data (CSV)'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"], key="1")

with st.sidebar.header('2. Upload your testing data (CSV)'):
    uploaded_test = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"], key="2")

@st.cache
def get_data_from_excel():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    return df

# @st.cache
def get_data_for_test():
    if uploaded_test is not None:
        df = pd.read_csv(uploaded_test)
    return df


df = get_data_from_excel()
X_test = get_data_for_test()

st.markdown('**Glimpse of training dataset**')
st.write(df)

X_train = df.iloc[:,:-1] # Using all column except for the last column as X
Y_train = df.iloc[:,-1]

st.write('Training set')
st.info(X_train.shape)

st.write('Test set')
st.info(X_test.shape)

st.markdown('**1.3. Variable details**:')
st.write('X variable')
st.info(list(X_train.columns))
st.write('Y variable')
st.info(Y_train.name)


# ---- statistics ----
st.header('Statistical info about the numerical variables')
st.write(X_train.describe().T)

# Missing sort_values

def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    ## the two following line may seem complicated but its actually very simple.
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

st.header('Look at the missingness')
st.write(missing_percentage(X_train))


# Normal distribution of sort_values
def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules.
    style.use('fivethirtyeight')

    left_column, middle_column, right_column = st.columns(3)

    ## Creating a customized chart. and giving in figsize and everything.
    fig = plt.figure(constrained_layout=True, figsize=(12,8))

    with left_column:
        st.subheader("Histogram:")
        sns.distplot(df.loc[:,feature], norm_hist=True)
        st.pyplot(fig)

    with middle_column:
        ## Plotting the QQ_Plot.
        st.subheader("Probability:")
        fig = plt.figure()
        ax1 =plt.subplot(121)
        # fig.set_size_inches(10, 5)
        stats.probplot(df.loc[:,feature],dist=stats.norm, plot=ax1)
        st.pyplot(fig)

    with right_column:
        st.subheader("Boxplot:")
        fig = plt.figure()
        sns.boxplot(df.loc[:,feature], orient='v')
        st.pyplot(fig)


# Normal distribution
## trainsforming target variable using numpy.log1p,
# df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

# df["SalePrice"] = np.log1p(df["SalePrice"])

## Plotting the newly transformed response variable
plotting_3_chart(df, 'SalePrice')

# ---- CORRELATION ----
st.markdown("""---""")

st.header('Which variables are correlated with the sale price')


def customized_scatterplot(y, x):
    ## Sizing the plot.
    style.use('fivethirtyeight')
    fig = plt.figure(figsize=(12,8))
    #Plotting target variable with predictor variable(OverallQual)
    sns.scatterplot(y = y, x = x);
    st.pyplot(fig)


def scatterplot_(x):
    fig = plt.figure()
    sns.regplot(x=df.loc[:,x], y=df.SalePrice)
    st.pyplot(fig)


expand_correlations = st.expander("Correlations", expanded=False)
with expand_correlations:
    st.write((df.corr()**2)["SalePrice"].sort_values(ascending = False)[1:])


expand_cats = st.expander("Correlations of categorical variables with sale price", expanded=False)
with expand_cats:
    left_column, right_column = st.columns(2)
    with left_column:
        option = st.selectbox(
         'Select a categorical variable?',
         (df.select_dtypes(include=['object']).columns.tolist()))

    with right_column:
        customized_scatterplot(df.SalePrice, df.loc[:,option])


# plotting numerical values
expand_nums = st.expander("Correlations of numerical variables with sale price", expanded=False)
with expand_nums:
    left_column, right_column = st.columns(2)

    with left_column:
        numerical = st.selectbox(
         'Select a numerical variable?',
         (df.select_dtypes(include=np.number).columns.tolist()))

    # customized_scatterplot(df.SalePrice, df.OverallQual)
    with right_column:
        scatterplot_(numerical)


expand_correlation_matrix = st.expander("Correlations Matrix", expanded=False)

with expand_correlation_matrix:
    style.use('ggplot')
    sns.set_style('whitegrid')
    fig = plt.figure(figsize = (30,20))
    # plt.subplots(figsize = (30,20))

    # Generate a mask for the upper triangle (taken from seaborn example gallery)
    mask = np.zeros_like(df.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(df.corr(),
                cmap=sns.diverging_palette(20, 220, n=200),
                mask = mask,
                annot=True,
                center = 0,
               );
    st.pyplot(fig)


# ----------------------------------Feature Engineering ----------------------------------#
st.header("Feature Engineering");

Y_train = df.iloc[:,-1].reset_index(drop=True) # Selecting the last column as Y

def feature_eng(dff):
    dff.drop(columns=['Id'],axis=1, inplace=True)
    st.write(missing_percentage(dff))

    missing_val_col = ["Alley",
                       "PoolQC",
                       "MiscFeature",
                       "Fence",
                       "FireplaceQu",
                       "GarageType",
                       "GarageFinish",
                       "GarageQual",
                       "GarageCond",
                       'BsmtQual',
                       'BsmtCond',
                       'BsmtExposure',
                       'BsmtFinType1',
                       'BsmtFinType2',
                       'MasVnrType']

    for i in missing_val_col:
        dff[i] = dff[i].fillna('None')

    missing_val_col2 = ['BsmtFinSF1',
                        'BsmtFinSF2',
                        'BsmtUnfSF',
                        'TotalBsmtSF',
                        'BsmtFullBath',
                        'BsmtHalfBath',
                        'GarageYrBlt',
                        'GarageArea',
                        'GarageCars',
                        'MasVnrArea']

    for i in missing_val_col2:
        dff[i] = dff[i].fillna(0)

    ## Replaced all missing values in LotFrontage by imputing the median value of each neighborhood.
    dff['LotFrontage'] = dff.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))

    ## Zoning class are given in numerical; therefore converted to categorical variables.
    dff['MSSubClass'] = dff['MSSubClass'].astype(str)
    dff['MSZoning'] = dff.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    ## Important years and months that should be categorical variables not numerical.
    dff['YrSold'] = dff['YrSold'].astype(str)
    dff['MoSold'] = dff['MoSold'].astype(str)

    dff['Functional'] = dff['Functional'].fillna('Typ')
    dff['Utilities'] = dff['Utilities'].fillna('AllPub')
    dff['Exterior1st'] = dff['Exterior1st'].fillna(dff['Exterior1st'].mode()[0])
    dff['Exterior2nd'] = dff['Exterior2nd'].fillna(dff['Exterior2nd'].mode()[0])
    dff['KitchenQual'] = dff['KitchenQual'].fillna("TA")
    dff['SaleType'] = dff['SaleType'].fillna(dff['SaleType'].mode()[0])
    dff['Electrical'] = dff['Electrical'].fillna("SBrkr")

    st.write("Any Missing Values?")
    st.write(missing_percentage(dff))

    # Fixing Skewness
    # numeric_feats = dff.dtypes[dff.dtypes != "object"].index
    # skewed_feats = dff[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    # st.write(skewed_feats)


    # feture engineering a new feature "TotalFS"
    dff['TotalSF'] = (dff['TotalBsmtSF']
                           + dff['1stFlrSF']
                           + dff['2ndFlrSF'])

    dff['YrBltAndRemod'] = dff['YearBuilt'] + dff['YearRemodAdd']

    dff['Total_sqr_footage'] = (dff['BsmtFinSF1']
                                     + dff['BsmtFinSF2']
                                     + dff['1stFlrSF']
                                     + dff['2ndFlrSF']
                                    )


    dff['Total_Bathrooms'] = (dff['FullBath']
                                   + (0.5 * dff['HalfBath'])
                                   + dff['BsmtFullBath']
                                   + (0.5 * dff['BsmtHalfBath'])
                                  )


    dff['Total_porch_sf'] = (dff['OpenPorchSF']
                                  + dff['3SsnPorch']
                                  + dff['EnclosedPorch']
                                  + dff['ScreenPorch']
                                  + dff['WoodDeckSF']
                                 )


    dff['haspool'] = dff['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    dff['has2ndfloor'] = dff['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    dff['hasgarage'] = dff['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    dff['hasbsmt'] = dff['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    dff['hasfireplace'] = dff['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    dff = dff.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

    ## Creating dummy variable
    cols = dff.select_dtypes(include=['object']).columns
    from sklearn.preprocessing import LabelEncoder
    for c in cols:
        dff[c] = LabelEncoder().fit_transform(dff[c])
    return dff

X_train = feature_eng(X_train)
X_test = feature_eng(X_test)

st.markdown('training is ')
st.info(X_train.shape)
st.markdown('testing is ')
st.info(X_test.shape)

# def overfit_reducer(df):
#     """
#     This function takes in a dataframe and returns a list of features that are overfitted.
#     """
#     overfit = []
#     for i in df.columns:
#         counts = df[i].value_counts()
#         zeros = counts.iloc[0]
#         if zeros / len(df) * 100 > 99.94:
#             overfit.append(i)
#     overfit = list(overfit)
#     return overfit
#
#
# overfitted_features = overfit_reducer(X_train)
#
# X_train = X_train.drop(overfitted_features, axis=1)
# X_sub = X_sub.drop(overfitted_features, axis=1)


#---------------------------------#
# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('2.2. General Parameters'):
    parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

st.write(X_train)
st.write(Y_train)

# Machine Learning model
# Model building
def build_model(X_train,Y_train):

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=(100-split_size)/100)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = rf.predict(X_train)

    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_train, Y_pred_train) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_train, Y_pred_train) )

    st.markdown('**2.2. Test set**')
    Y_pred_test = rf.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info( r2_score(Y_test, Y_pred_test) )

    st.write('Error (MSE or MAE):')
    st.info( mean_squared_error(Y_test, Y_pred_test))
    st.subheader('3. Model Parameters')
    st.write(rf.get_params())

build_model(X_train,Y_train)











# ---- SIDEBAR ----
# st.sidebar.header("Please Filter Here:")
#
# neighborhood = st.sidebar.multiselect(
#     "Select the City:",
#     options=df["Neighborhood"].unique(),
#     default=df["Neighborhood"].unique()
# )
#
# roofStyle = st.sidebar.multiselect(
#     "Select the Roof Style:",
#     options=df["RoofStyle"].unique(),
#     default=df["RoofStyle"].unique()
# )
#
# df_selection = df.query(
#     "Neighborhood == @neighborhood & RoofStyle == @roofStyle"
# )
#
# st.dataframe(df_selection)
#
# # ---- MAINPAGE ----
# st.title(":bar_chart: Sales Dashboard")
# st.markdown("##")
#
# # Top KPIs
# total_sales = int(df_selection["SalePrice"].sum())
#
# left_column, middle_column, right_column = st.columns(3)
#
# with left_column:
#     st.subheader("Total Sales:")
#     st.subheader(f"US $ {total_sales:,}")
#
# with middle_column:
#     st.subheader("Total Sales:")
#     st.subheader(f"US $ {total_sales:,}")
#
# with right_column:
#     st.subheader("Total Sales:")
#     st.subheader(f"US $ {total_sales:,}")
#
# st.markdown("...")
#
# # SALES BY ZONING [BAR CHART]
# sales_by_zonning = (
#     df_selection.groupby(by=["MSZoning"]).sum()[["SalePrice"]].sort_values(by="SalePrice")
# )
#
# fig_product_sales = px.bar(
#     sales_by_zonning,
#     x="SalePrice",
#     y=sales_by_zonning.index,
#     orientation="h",
#     title="<b>Sales by Zonning</b>",
#     color_discrete_sequence=["#0083B8"] * len(sales_by_zonning),
#     template="plotly_white",
# )
#
# fig_product_sales.update_layout(
#     plot_bgcolor="rgba(0,0,0,0)",
#     xaxis=(dict(showgrid=False))
# )
#
# st.plotly_chart(fig_product_sales)
#

# ---- HIDE STREAMLIT STYLE ----
# hide_st_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)
