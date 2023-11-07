import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import seaborn as sns



# Load the data
education_districtwise = pd.read_csv('education_districtwise.csv')


# check the data
print(education_districtwise.head(10))

# compute descriptive stats
# .describe() excludes missing values
print(education_districtwise.describe())

# computing the stats for just one feature: OVERALL_LI
print(education_districtwise['OVERALL_LI'].describe())

# If the data in a column is categorical not numerical describe() gives the following output
# count: Number of non-NA/null observations
# unique: Number of unique values
# top: The most common value (the mode)
# freq: The frequency of the most common value

print(education_districtwise['STATNAME'].describe())

# Load EPA dataset, index_col = 0 --> first column will be read as an index
epa_data = pd.read_csv("c4_epa_air_quality.csv", index_col = 0)

# include = 'all', will have all the columns even the ones it is not possible to calculate statistics, categoroical, for and for those ones it will give the unique, top and freq
#print(epa_data.head(10), epa_data.describe(include='all'))
# Note on percentiles. "AQI" value for the percentile 25% is 2, meaning 25% of data has an "AQI" value below 2. and 75% have it below 9

print(epa_data['state_name'].describe())
print(epa_data['aqi'].mean(), epa_data['aqi'].median(), epa_data['aqi'].min(), epa_data['aqi'].max())


# By default, the numpy library uses 0 as the Delta Degrees of Freedom, while pandas library uses 1.
# To get the same value for standard deviation using either library, specify the ddof parameter to 1 when calculating standard deviation.
#print(epa_data['aqi'].std(ddof = 1))


###############################################################
######## Week 2 ###############################################
# use the eductaion_districtwise dataset

# Remove missing values (all lines with NAs will be removed

education_districtwise = education_districtwise.dropna()


education_districtwise['OVERALL_LI'].hist()
plt.show()

mean_overall_li = education_districtwise['OVERALL_LI'].mean()
print('Population mean literacy rate is ', mean_overall_li)


std_overall_li = education_districtwise['OVERALL_LI'].std()
print(std_overall_li)

# calculate the actual % of district literacy rates that is withing +/- 1, 2 and 3 STD from the mean
lower_limit = mean_overall_li - 1 * std_overall_li
upper_limit = mean_overall_li + 1 * std_overall_li
within_1STD = ((education_districtwise['OVERALL_LI'] >= lower_limit) & (education_districtwise['OVERALL_LI'] <= upper_limit)).mean()
lower_limit = mean_overall_li - 2 * std_overall_li
upper_limit = mean_overall_li + 2 * std_overall_li
within_2STD = ((education_districtwise['OVERALL_LI'] >= lower_limit) & (education_districtwise['OVERALL_LI'] <= upper_limit)).mean()
lower_limit = mean_overall_li - 3 * std_overall_li
upper_limit = mean_overall_li + 3 * std_overall_li
within_3STD =((education_districtwise['OVERALL_LI'] >= lower_limit) & (education_districtwise['OVERALL_LI'] <= upper_limit)).mean()

print(within_1STD, within_2STD, within_3STD, 'values arre close to the empirical rule which is 68%, 95% and 99.7% respectively')

compute the z - scores using the function scipy.stats.zscore().
education_districtwise['Z_SCORE'] = stats.zscore(education_districtwise['OVERALL_LI'])
print(education_districtwise.head())

# select rows based on the Z-score value previously calculated. Look at the samples that are outside the +/-3SD range
unusual_districts = education_districtwise[(education_districtwise['Z_SCORE'] > 3) | (education_districtwise['Z_SCORE'] < -3)]
print(unusual_districts)


###############################################################
######## Week 2 ###############################################
####### Sampling ##############################################
# use the eductaion_districtwise dataset

# n: Refers to the desired sample size
# replace: Indicates whether you are sampling with or without replacement
# random_state: Refers to the seed of the random number

sampled_data = education_districtwise.sample(n=50, replace=True, random_state=31208)

# Compute sample mean

estimate1 = sampled_data['OVERALL_LI'].mean()

# compute sample mean of another sample, the random seed has to be changed to another number.
estimate2 = education_districtwise['OVERALL_LI'].sample(n=50, replace=True, random_state=56810).mean()

print('The computed sample mean for district literacy for two different samples is: ', estimate1, estimate2)


# Compute the mean of a sampling distribution with 10,000 samples
estimate_list = []
for i in range(10000):
   estimate_list.append(education_districtwise['OVERALL_LI'].sample(n=50, replace=True).mean())
estimate_df = pd.DataFrame(data={'estimate': estimate_list})

mean_sample_means = estimate_df['estimate'].mean()
print(' Population mean is' , mean_overall_li, 'While the estimate based on 10000samples is ', mean_sample_means)

# Visualizing

plt.hist(estimate_df['estimate'], bins=25, density=True, alpha=0.4, label = "histogram of sample means of 10000 random samples")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100) # generate a grid of 100 values from xmin to xmax.
p = stats.norm.pdf(x, mean_sample_means, stats.tstd(estimate_df['estimate']))
plt.plot(x, p,'k', linewidth=2, label = 'normal curve from central limit theorem')
plt.axvline(x=mean_overall_li, color='g', linestyle = 'solid', label = 'population mean')
plt.axvline(x=estimate1, color='r', linestyle = '--', label = 'sample mean of the first random sample')
plt.axvline(x=mean_sample_means, color='b', linestyle = ':', label = 'mean of sample means of 10000 random samples')
plt.title("Sampling distribution of sample mean")
plt.xlabel('sample mean')
plt.ylabel('density')
plt.legend(bbox_to_anchor=(1.04,1))
plt.show()


### Part II ###
population_mean = epa_data['aqi'].mean()

sampled_data = epa_data.sample(n=50, replace=True, random_state=42)

print(sampled_data.head(10))

sample_mean = sampled_data['aqi'].mean()
print('Mean for the first sample evaluated is', sample_mean)


# get 10000 samples for the mean
estimate_list = []
for s in range (10000):
    estimate_list.append(epa_data['aqi'].sample(n=50, replace = True).mean())

estimate_df = pd.DataFrame(data={'estimate':estimate_list})
print(estimate_df)

mean_sample_means = estimate_df['estimate'].mean()
print('The mean of means is ', mean_sample_means)
print('The population mean is ',population_mean)


# Output the distribution using a histogram
estimate_df['estimate'].hist()
standard_error = sampled_data['aqi'].std() / np.sqrt(len(sampled_data))
print('The standard error is ', standard_error)


plt.figure(figsize=(8,5))
plt.hist(estimate_df['estimate'], bins=25, density=True, alpha=0.4, label = "histogram of sample means of 10000 random samples")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100) # generate a grid of 100 values from xmin to xmax.
p = stats.norm.pdf(x, population_mean, standard_error)
plt.plot(x, p, 'k', linewidth=2, label = 'normal curve from central limit theorem')
plt.axvline(x=population_mean, color='m', linestyle = 'solid', label = 'population mean')
plt.axvline(x=sample_mean, color='r', linestyle = '--', label = 'sample mean of the first random sample')
plt.axvline(x=mean_sample_means, color='b', linestyle = ':', label = 'mean of sample means of 10000 random samples')
plt.title("Sampling distribution of sample mean")
plt.xlabel('sample mean')
plt.ylabel('density')
plt.legend(bbox_to_anchor=(1.04,1));
plt.show()

######## Week 4 ##################################################################################
##################################################################################################

#when working with a large sample size, say larger than 30, you can construct a confidence interval for
# the mean using scipy.stats.norm.interval(). This function includes the following arguments:

#alpha: The confidence level
#loc: The sample mean
#scale: The sample standard error

sample_mean = sampled_data['OVERALL_LI'].mean()

estimated_standard_error = sampled_data['OVERALL_LI'].std() / np.sqrt(sampled_data.shape[0])
print('CI 95% ', (stats.norm.interval(alpha=0.95, loc=sample_mean, scale=estimated_standard_error)))

print('CI 99% ', (stats.norm.interval(alpha=0.99, loc=sample_mean, scale=estimated_standard_error)))

aqi = pd.read_csv('c4_epa_air_quality.csv')

print("Use describe() to summarize AQI")
print(aqi.describe(include='all'))

print("For a more thorough examination of observations by state use values_counts()")
print(aqi['state_name'].value_counts())

rre_states = ['California','Florida','Michigan','Ohio','Pennsylvania','Texas']
# Subset `aqi` to only consider only RRE states
aqi_rre = aqi[aqi['state_name'].isin(rre_states)]
# Find the mean aqi for each of the RRE states.

aqi_rre.groupby(['state_name']).agg({"aqi":"mean","state_name":"count"}) #alias as aqi_rre

sns.boxplot(x=aqi_rre["state_name"],y=aqi_rre["aqi"])
plt.show()


aqi_ca = aqi[aqi['state_name']=='California']

sample_mean = aqi_ca['aqi'].mean()
print('The aqi sample mean for California is ', sample_mean)

# margin of error = z * standard error

# Z-score CI 90% is 1.65, for 95% is 1.96 and 99% is 2.55
confidence_level = 0.95
confidence_level
z_value = 1.96
z_value_alt1 = 1.65
z_value_alt2 = 2.55

standard_error = aqi_ca['aqi'].std() / np.sqrt(aqi_ca.shape[0])
print("standard error:")
print(standard_error)

# calculate your margin of error.
margin_of_error = standard_error * z_value
print("margin of error with a 95% CI:")
print(margin_of_error)

margin_of_error_alt1 = standard_error * z_value_alt1
margin_of_error_alt2 = standard_error * z_value_alt2
print('The margin of error with a 90% and 95% CI is', margin_of_error_alt1, 'and', margin_of_error_alt2, ' respectively')

upper_ci_limit = sample_mean + margin_of_error
lower_ci_limit = sample_mean - margin_of_error
print('The CI for 9*5% is ', (lower_ci_limit, upper_ci_limit))


print('Recalculated CI is ', stats.norm.interval(alpha=confidence_level, loc=sample_mean, scale=standard_error))



############ Week 5 ################################################################
####################################################################################

### working with the education_districtwise dataset

## filter data to on to two states (the largest ones) :  STATE21 and STATE28.

state21 = education_districtwise[education_districtwise['STATNAME'] == "STATE21"]

state28 = education_districtwise[education_districtwise['STATNAME'] == "STATE28"]

#simulate random sampling, using 20 districts n =20

sampled_state21 = state21.sample(n=20, replace = True, random_state=13490)
sampled_state28 = state28.sample(n=20, replace = True, random_state=39103)

state21_mean = sampled_state21['OVERALL_LI'].mean()
state28_mean = sampled_state28['OVERALL_LI'].mean()

## hypothesis test
## 𝐻0: There is no difference in the mean district literacy rates between STATE21 and STATE28.
## 𝐻𝐴: There is a difference in the mean district literacy rates between STATE21 and STATE28.
## Significance level -- 0.05 was chosen
## p-value -probability of observing results as or more extreme than those observed when the null hypothesis is true.
## use scipy.stats.ttest_ind() to calculate p-value

print(stats.ttest_ind(a=sampled_state21['OVERALL_LI'], b=sampled_state28['OVERALL_LI'], equal_var=False))
# pvalue smaller than the significance level (0.64 vs 5%) and thus the null hypothesis is rejected


########
print(epa_data.describe(include = "all"))
print("For a more thorough examination of observations by state use values_counts()")
print(epa_data['state_name'].value_counts())
# Hypothesis 1: ROA is considering a metropolitan-focused approach. Within California,
# they want to know if the mean AQI in Los Angeles County is statistically different from the rest of California
# create data frames for each sample being compared
ca_la = epa_data[epa_data['county_name']=='Los Angeles']
ca_other = epa_data[(epa_data['state_name']=='California') & (epa_data['county_name']!='Los Angeles')]

# set significance level
significance_level = 0.05
# p-value calculation
print(stats.ttest_ind(a=ca_la['aqi'], b=ca_other['aqi'], equal_var=False))
print('With a p-value (0.049) being less than 0.05 (as your significance level is 5%), reject the null hypothesis in favor of the alternative hypothesis.'
      'Therefore, a metropolitan strategy may make sense in this case.')
#AQI in Los Angeles County was in fact different from the rest of California.

#Hypothesis 2: With limited resources, ROA has to choose between New York and Ohio for their next regional office. Does New York have a lower AQI than Ohio?

ny = epa_data[epa_data['state_name']=='New York']
ohio = epa_data[epa_data['state_name']=='Ohio']



# significance leval is still 5%

tstat, pvalue = stats.ttest_ind(a=ny['aqi'], b=ohio['aqi'], alternative='less', equal_var=False)

print(tstat, pvalue)

# With a p-value (0.030) of less than 0.05 (as your significance level is 5%) and a t-statistic < 0 (-2.036),
# reject the null hypothesis in favor of the alternative hypothesis.
# Therefore, you can conclude at the 5% significance level that New York has a lower mean AQI than Ohio.


## Hypothesis 3: A new policy will affect those states with a mean AQI of 10 or greater. Can you rule out Michigan from being affected by this new policy?

#  comparing one sample mean relative to a particular value in one direction. Thus use a one-sample 𝑡-test. Note that the function is different

mi = epa_data[epa_data['state_name']== 'Michigan']
tstat, pvalue = stats.ttest_1samp(mi['aqi'], 10, alternative='greater')
print(tstat)
print(pvalue)

# What is your p-value for hypothesis 3, and what does this indicate for your null hypothesis?
# With a p-value (0.940) being greater than 0.05 (as your significance level is 5%) and a t-statistic < 0 (-1.74), fail to reject the null hypothesis.
#
# Therefore, you cannot conclude at the 5% significance level that Michigan's mean AQI is greater than 10.
# This implies that Michigan would not be affected by the new policy.
