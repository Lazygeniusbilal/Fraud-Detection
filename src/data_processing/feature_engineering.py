# Feature engineering on the identity data (device, OS, browser, screen).
# Run this on the identity dataframe BEFORE merging it with the transactions,
# so `had_id` ends up as 1 for matched rows and NaN for unmatched ones.

import gc
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def Devices(df: pd.DataFrame) -> pd.DataFrame:
    """
        Splits DeviceInfo, id_30, id_31 and id_33 into separate parts, groups device
        model codes into brand names, puts rare brands into "Others" and adds a
        `had_id` flag.
    """
    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
    df['device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]
    df = df.drop(['DeviceInfo'], axis = 1)

    df['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
    df['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]
    df = df.drop(['id_30'], axis = 1)

    df['browser_id_31'] = df['id_31'].str.split(' ', expand=True)[0]
    df['version_id_31'] = df['id_31'].str.split(' ', expand=True)[1]
    df = df.drop(['id_31'], axis = 1)

    df['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
    df['screen_height'] = df['id_33'].str.split('x', expand=True)[1]
    df = df.drop(['id_33'], axis = 1)

    df['id_34'] = df['id_34'].str.split(':', expand=True)[1]
    df['id_23'] = df['id_23'].str.split(':', expand=True)[1]
    df = df.drop(['id_34', 'id_23'], axis = 1)


    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    df.loc[df.device_name.isin(df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    df['had_id'] = 1
    gc.collect()

    return df


def split_num_cat(train: pd.DataFrame):
    """
        Separates the target (isFraud), drops columns that are not features
        (isFraud, TransactionID, TransactionDT) and splits the rest into
        numerical and categorical dataframes.
        Note: pandas 'category' columns are not np.number, so they land in cat_df.
    """
    # unlabelled data (e.g. new transactions to score) has no isFraud -> y is None
    y = train['isFraud'] if 'isFraud' in train.columns else None
    train = train.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis = 1, errors='ignore')

    # Filtering numerical data
    num_df = train.select_dtypes(include=np.number)
    print(num_df.shape)

    # Filtering categorical data
    cat_df = train.select_dtypes(exclude=np.number)
    print(cat_df.shape)

    return train, y, num_df, cat_df


def impute_missing(num_df: pd.DataFrame, cat_df: pd.DataFrame):
    """
        Fills missing values: median for numerical columns, most frequent value for
        categorical columns. Fit these on TRAIN only and reuse the returned imputers
        with .transform() on test, so test is filled with train's medians/modes.
    """
    # Filling missing values by median for numerical columns
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=True)
    num_df = pd.DataFrame(imp_median.fit_transform(num_df), columns=num_df.columns, index=num_df.index).astype(np.float32)
    print(num_df.shape)

    # Filling missing values by most frequent value for categorical columns
    # ('category' dtype is converted to object first so SimpleImputer can handle it)
    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    cat_df = pd.DataFrame(imp_mode.fit_transform(cat_df.astype(object)), columns=cat_df.columns, index=cat_df.index)
    print(cat_df.shape)

    return num_df, cat_df, imp_median, imp_mode


def scale_numerical(num_df: pd.DataFrame):
    """
        Scales every numerical column with RobustScaler (median and IQR), so outliers
        do not squash the other values. Boosting models do not need scaling, but it does no harm.
        Fit on TRAIN only and reuse the returned scaler with .transform() on test.
    """
    scaler = preprocessing.RobustScaler()
    scaled_num = pd.DataFrame(scaler.fit_transform(num_df), columns = num_df.columns, index = num_df.index).astype(np.float32)
    print(scaled_num.shape)
    return scaled_num, scaler


def plot_pca_variance(scaled_num: pd.DataFrame):
    """
        Fits a full PCA and plots the cumulative explained variance per number of
        components, to see how many components are needed to keep most of the information.
    """
    pca = PCA().fit(scaled_num)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("number of components")
    plt.ylabel("Cumulative Rate of Variance")
    plt.show()


def apply_pca(scaled_num: pd.DataFrame, variance: float = 0.95):
    """
        Reduces the scaled numerical columns to the fewest principal components that
        keep `variance` (default 95%) of the variance.
        Fit on TRAIN only and reuse the returned pca with .transform() on test.
    """
    pca = PCA(n_components = variance)
    pca_fit = pca.fit_transform(scaled_num)

    # string column names so sklearn does not raise the feature-name warning later
    columns = [f"PC_{i + 1}" for i in range(pca.n_components_)]
    num_pca = pd.DataFrame(data = pca_fit, columns = columns, index = scaled_num.index).astype(np.float32)
    print(f"PCA components: {pca.n_components_}, explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    return num_pca, pca


# Which encoding each categorical column gets, based on its nature:
#   - two values (T/F)               -> label (T=1, F=0)
#   - few values, no order           -> one-hot
#   - many values (59 email domains) -> frequency
BINARY_COLUMNS = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
ONEHOT_COLUMNS = ['ProductCD', 'card4', 'card6', 'M4']
FREQUENCY_COLUMNS = ['P_emaildomain']


def encode_categorical(cat_df: pd.DataFrame, encoders: dict | None = None):
    """
        Encodes the (already imputed) categorical columns:
          - label for the T/F columns, one-hot for the low-cardinality ones and
            frequency (share of rows) for the email domain.
        On TRAIN call it without `encoders`: it fits them and returns them.
        On TEST pass the returned `encoders` so test gets the same columns and
        train's frequencies. Unseen categories in test become all-zero / frequency 0.
    """
    fit = encoders is None
    if fit:
        encoders = {}

    parts = []

    # label: T/F -> 1/0
    binary_cols = [c for c in BINARY_COLUMNS if c in cat_df.columns]
    # float32 so a missing value can stay NaN when imputation is off
    parts.append(cat_df[binary_cols].apply(lambda s: s.map({'T': 1, 'F': 0})).astype(np.float32))

    # one-hot
    onehot_cols = [c for c in ONEHOT_COLUMNS if c in cat_df.columns]
    if fit:
        encoders['onehot'] = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int8)
        encoders['onehot'].fit(cat_df[onehot_cols])
    onehot = encoders['onehot'].transform(cat_df[onehot_cols])
    parts.append(pd.DataFrame(onehot, columns=encoders['onehot'].get_feature_names_out(onehot_cols), index=cat_df.index))

    # frequency: each category -> share of train rows that have it
    for c in [c for c in FREQUENCY_COLUMNS if c in cat_df.columns]:
        if fit:
            encoders[c] = cat_df[c].value_counts(normalize=True)
        parts.append(cat_df[c].map(encoders[c]).fillna(0).astype(np.float32).rename(f"{c}_freq"))

    encoded = pd.concat(parts, axis=1)
    print(encoded.shape)
    return encoded, encoders


def combine_features(num_pca: pd.DataFrame, cat_encoded: pd.DataFrame) -> pd.DataFrame:
    """
        Joins the PCA components and the encoded categorical columns side by side
        into the final feature matrix. Both come from the same rows, so they share
        the index and line up row by row (and with `y` from split_num_cat).
    """
    if not num_pca.index.equals(cat_encoded.index):
        raise ValueError("num_pca and cat_encoded have different row indexes")

    X = pd.concat([num_pca, cat_encoded], axis=1)
    print(X.shape)
    return X


def fit_features(df: pd.DataFrame, impute: bool = True):
    """
        Runs the whole feature chain on TRAIN and fits every step on it:
        split -> impute -> scale -> encode -> combine (PCA step is commented out).
        impute=False skips imputing and scaling and keeps NaN in the numerical
        columns: tree models (XGBoost, LightGBM, HistGradientBoosting, Random Forest)
        handle NaN natively and learn which side of a split missing rows belong to,
        so the "this value was missing" signal is kept. Missing categoricals become
        their own one-hot category.
        Returns X, y and the fitted objects to reuse with transform_features().
    """
    _, y, num_df, cat_df = split_num_cat(train=df)
    num_cols, cat_cols = list(num_df.columns), list(cat_df.columns)
    if impute:
        num_df, cat_df, imp_median, imp_mode = impute_missing(num_df=num_df, cat_df=cat_df)
        num_df, scaler = scale_numerical(num_df=num_df)
    else:
        imp_median = imp_mode = scaler = None
        num_df = num_df.astype(np.float32)
        cat_df = cat_df.astype(object)
    # PCA disabled: trees don't need it and it threw away fraud signal (211 -> 8 columns)
    # num_pca, pca = apply_pca(scaled_num=num_df)
    cat_encoded, encoders = encode_categorical(cat_df=cat_df)
    # X = combine_features(num_pca=num_pca, cat_encoded=cat_encoded)
    X = combine_features(num_pca=num_df, cat_encoded=cat_encoded)

    fitted = {
        'impute': impute, 'num_cols': num_cols, 'cat_cols': cat_cols,
        'imp_median': imp_median, 'imp_mode': imp_mode, 'scaler': scaler,
        # 'pca': pca,
        'encoders': encoders, 'feature_columns': list(X.columns),
    }
    return X, y, fitted


def transform_features(df: pd.DataFrame, fitted: dict):
    """
        Applies the objects fitted by fit_features() to new data (validation, test
        or live transactions) without refitting anything. Columns are picked by the
        names seen during fitting, so the output always matches the training features.
        Returns X and y (y is None when df has no isFraud column).
    """
    y = df['isFraud'] if 'isFraud' in df.columns else None
    num_cols, cat_cols = fitted['num_cols'], fitted['cat_cols']
    num_df, cat_df = df[num_cols], df[cat_cols]

    if fitted['impute']:
        num_df = pd.DataFrame(fitted['imp_median'].transform(num_df), columns=num_cols, index=df.index).astype(np.float32)
        cat_df = pd.DataFrame(fitted['imp_mode'].transform(cat_df.astype(object)), columns=cat_cols, index=df.index)
        num_df = pd.DataFrame(fitted['scaler'].transform(num_df), columns=num_cols, index=df.index).astype(np.float32)
    else:
        num_df = num_df.astype(np.float32)
        cat_df = cat_df.astype(object)
    # PCA disabled (see fit_features)
    # pca = fitted['pca']
    # num_pca = pd.DataFrame(pca.transform(num_df), columns=[f"PC_{i + 1}" for i in range(pca.n_components_)], index=df.index).astype(np.float32)
    cat_encoded, _ = encode_categorical(cat_df=cat_df, encoders=fitted['encoders'])

    # X = combine_features(num_pca=num_pca, cat_encoded=cat_encoded)[fitted['feature_columns']]
    X = combine_features(num_pca=num_df, cat_encoded=cat_encoded)[fitted['feature_columns']]
    return X, y
