import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

atlas = pd.read_csv("/users/zetzioni/sharedscratch/atlas/atlas/atlas_dmr_by_read.blood+gi+tum.U100.l4.bed", sep="\t")
atlas = atlas[atlas.target=="OAC"]
atlas.reset_index(inplace=True)
samples = glob.glob("/mnt/lustre/users/bschuster/OAC_Trial_WGS_Tissue_CNA-Hatchet/Results/*")

sample_data_list = []
for sample in samples:
    sample_name = os.path.basename(sample.rstrip('/'))
    bbc_file = os.path.join(sample, "best.bbc.ucn")
    if not os.path.exists(bbc_file):
        print(f"Warning: File {bbc_file} not found, skipping sample {sample}")
        continue
    df = pd.read_csv(bbc_file, sep="\t")
    print(f"processing sample for {sample_name}")
    def map_region(row):
        chr = row['chr']
        start = row['start']
        end = row['end']
        reg1 = df[(df['#CHR']==chr) & (df['START']<start) & (df['END']>start)]
        reg2 = df[(df['#CHR']==chr) & (df['START']<end) & (df['END']>end)]
        if reg1.empty or reg2.empty:
            return 1.0
        if reg1.index.equals(reg2.index):
            return reg1.RD.values[0]
        reg1_length = (reg1['END'].values[0] - start)
        reg2_length = (end - reg2['START'].values[0])
        length = reg1_length + reg2_length
        weighted_rd = reg1_length/length * reg1.RD.values[0] + reg2_length/length * reg2.RD.values[0]
        return weighted_rd
    rd_values = atlas.apply(lambda row: map_region(row), axis=1)
    if not df.empty and 'SAMPLE' in df.columns:
        full_sample_name = df['SAMPLE'].iloc[0]
        parts = full_sample_name.split(':')
        if len(parts) >= 2:
            sample_name = f"{parts[0]}_{parts[1]}"
        else:
            sample_name = os.path.basename(sample.rstrip('/'))
    else:
        sample_name = os.path.basename(sample.rstrip('/'))
    sample_data = {'sample': sample_name}
    for i, rd_value in enumerate(rd_values):
        region_name = atlas.loc[i, 'name'] 
        sample_data[region_name] = rd_value
    sample_data_list.append(sample_data)

# Create the final dataframe from the list of sample data
result_df = pd.DataFrame(sample_data_list)
result_df.to_csv('/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/oac_detector/hatchet_rd_values.csv', index=False)

predictions = pd.read_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/oac_detector/predictions.csv")
predictions['sample']=predictions['sample_id'].map(lambda x: x.replace("_plasma_md",""))
predictions = predictions.drop(columns=['sample_id'])

merged_df = pd.merge(
    predictions,     
    result_df,       
    on='sample',  
    how='inner'      
)
ichorcna = pd.read_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/ab_ichorcna_cfdna.csv", sep="\t")
merged_df = pd.merge(merged_df, ichorcna, on='sample',how='inner')

rd_columns = [col for col in merged_df.columns if col not in ['sample', 'sample_name', 'estimated', 
                                                              'lower_ci', 'upper_ci', 'uncertainty', 
                                                              'tf', 'ploidy']]


# Check the correlation between original estimates and ichorCNA TF
original_corr = np.corrcoef(merged_df['estimated'], merged_df['tf'])[0,1]
print(f"Correlation between original DL estimate and ichorCNA TF: {original_corr:.4f}")

# Prepare features and target
X = merged_df[['estimated'] + rd_columns]
y = merged_df['tf']

# Check for data quality issues
print("\nChecking for problematic values in the data:")
print(f"X shape: {X.shape}")
print(f"NaN values in features: {np.isnan(X).sum().sum()}")
print(f"Infinite values in features: {np.isinf(X).sum().sum()}")
print(f"NaN values in target: {np.isnan(y).sum()}")
print(f"Infinite values in target: {np.isinf(y).sum()}")

X_simple = merged_df[['estimated']].values.reshape(-1, 1)
X_train_simple, X_test_simple, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)
y_pred_simple = model_simple.predict(X_test_simple)

mse_simple = mean_squared_error(y_test, y_pred_simple)
r2_simple = r2_score(y_test, y_pred_simple)
print(f"\nSimple correction model (estimated only):")
print(f"Mean Squared Error: {mse_simple:.4f}")
print(f"R-squared: {r2_simple:.4f}")
print(f"Formula: Corrected TF = {model_simple.intercept_:.4f} + ({model_simple.coef_[0]:.4f} × estimated)")

merged_df['mean_RD'] = merged_df[rd_columns].mean(axis=1)
merged_df['median_RD'] = merged_df[rd_columns].median(axis=1)
X_with_mean = merged_df[['estimated', 'mean_RD', 'median_RD']]
X_train_mean, X_test_mean, y_train, y_test = train_test_split(
    X_with_mean, y, test_size=0.2, random_state=42)

model_with_mean = LinearRegression()
model_with_mean.fit(X_train_mean, y_train)
y_pred_mean = model_with_mean.predict(X_test_mean)

mse_mean = mean_squared_error(y_test, y_pred_mean)
r2_mean = r2_score(y_test, y_pred_mean)
print(f"\nModel with estimated + mean/median RD:")
print(f"Mean Squared Error: {mse_mean:.4f}")
print(f"R-squared: {r2_mean:.4f}")
print(f"Coefficients: {dict(zip(['intercept', 'estimated', 'mean_RD', 'median_RD'], 
                              [model_with_mean.intercept_] + model_with_mean.coef_.tolist()))}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2')
ridge_grid.fit(X_train_scaled, y_train)

best_ridge = ridge_grid.best_estimator_
y_pred_ridge = best_ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"\nRidge regression model (alpha={ridge_grid.best_params_['alpha']}):")
print(f"Mean Squared Error: {mse_ridge:.4f}")
print(f"R-squared: {r2_ridge:.4f}")

merged_df['estimated_squared'] = merged_df['estimated'] ** 2
merged_df['estimated_log'] = np.log1p(merged_df['estimated'])  # log(1+x) to handle zeros
X_nonlinear = merged_df[['estimated', 'estimated_squared', 'estimated_log', 'mean_RD', 'median_RD']]
X_train_nl, X_test_nl, y_train, y_test = train_test_split(
    X_nonlinear, y, test_size=0.2, random_state=42)

model_nonlinear = LinearRegression()
model_nonlinear.fit(X_train_nl, y_train)
y_pred_nl = model_nonlinear.predict(X_test_nl)

mse_nl = mean_squared_error(y_test, y_pred_nl)
r2_nl = r2_score(y_test, y_pred_nl)
print(f"\nNon-linear transformation model:")
print(f"Mean Squared Error: {mse_nl:.4f}")
print(f"R-squared: {r2_nl:.4f}")

ridge_all_preds = best_ridge.predict(X_scaled)
ridge_full_mse = mean_squared_error(y, ridge_all_preds)
ridge_full_r2 = r2_score(y, ridge_all_preds)

original_mse = mean_squared_error(y, merged_df['estimated'])
original_r2 = r2_score(y, merged_df['estimated'])

print("\nPerformance on full dataset:")
print(f"Original DL model - MSE: {original_mse:.4f}, R²: {original_r2:.4f}")
print(f"Ridge regression model - MSE: {ridge_full_mse:.4f}, R²: {ridge_full_r2:.4f}")
print(f"Improvement: {(original_mse - ridge_full_mse) / original_mse * 100:.2f}% reduction in MSE")

# Visualize the comparison between original and corrected estimates
comparison_df = pd.DataFrame({
    'Sample': merged_df['sample'],
    'Original_Estimate': merged_df['estimated'],
    'Corrected_Estimate': ridge_all_preds,
    'ichorCNA_TF': merged_df['tf']
})

comparison_df.to_csv("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/oac_detector/corrected_estimates.csv", index=False)

fig = go.Figure()
fig.add_trace(go.Scatter(x=comparison_df['ichorCNA_TF'], 
                         y=comparison_df['Original_Estimate'],
                         mode='markers', name='Original Estimates',
                         marker=dict(color='blue')))
fig.add_trace(go.Scatter(x=comparison_df['ichorCNA_TF'], 
                         y=comparison_df['Corrected_Estimate'],
                         mode='markers', name='Corrected Estimates',
                         marker=dict(color='green')))
fig.add_trace(go.Scatter(x=[0, comparison_df['ichorCNA_TF'].max()], 
                         y=[0, comparison_df['ichorCNA_TF'].max()],
                         mode='lines', name='Perfect Prediction',
                         line=dict(color='red', dash='dash')))
fig.update_layout(title='Original vs Corrected Estimates (Ridge Regression Model)',
                  xaxis_title='ichorCNA Tumor Fraction',
                  yaxis_title='Estimated Tumor Fraction',
                  width=900, height=700)

fig.write_html("/users/zetzioni/sharedscratch/loyfer_atlas/OAC/analysis/AB/cfDNA/oac_detector/linear_regression.html")                   
