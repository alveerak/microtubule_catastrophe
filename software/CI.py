import bokeh.io

import numpy as np
import pandas as pd

import scipy.stats

import numba

import bokeh.io
import holoviews as hv
import bokeh_catplot

import bebi103
import CI

hv.extension("bokeh")
bokeh.io.output_notebook()
bebi103.hv.set_defaults()

def plot_ECDF(df):

    # Plot ECDF of labeled and unlabeled tubulin catastrophe times
    plt = bokeh_catplot.ecdf(
        cats=["labeled"],
        data=df,
        val="time to catastrophe (s)",
        conf_int=True,
        style="staircase",
        title="",
    )
    plt.xaxis.axis_label = "Time to Catastrophe (s)"
    plt.legend.title = "Labeled"
    return bokeh.io.show(plt)

@numba.njit
def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))


@numba.njit
def draw_bs_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from a 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out


def comp_conf_int(df):
    rg = np.random.default_rng()

    # Extract data as numpy arrays
    time_labeled = df.loc[df["labeled"] == True, "time to catastrophe (s)"].values
    time_unlabeled = df.loc[df["labeled"] == False, "time to catastrophe (s)"].values

    # Draw bootstrapping samples
    bs_reps_mean_labeled = CI.draw_bs_reps_mean(time_labeled, size=10000)
    bs_reps_mean_unlabeled = CI.draw_bs_reps_mean(time_unlabeled, size=10000)

    # 95% confidence intervals
    mean_labeled_conf_int = np.percentile(bs_reps_mean_labeled, [2.5, 97.5])
    mean_unlabeled_conf_int = np.percentile(bs_reps_mean_unlabeled, [2.5, 97.5])

    print("""
    Mean time to catastrophe for labeled tubulin 95% conf int (s):   [{0:.2f}, {1:.2f}]
    Mean time to catastrophe for unlabeled tubulin 95% conf int (s): [{2:.2f}, {3:.2f}]
    """.format(*(tuple(mean_labeled_conf_int) + tuple(mean_unlabeled_conf_int)))
    )

@numba.njit
def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)
    return concat_data[: len(x)], concat_data[len(x) :]


@numba.njit
def draw_perm_reps_diff_mean(x, y, size=1):
    """Generate array of difference in mean permuation replicates."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.abs(np.mean(x_perm) - np.mean(y_perm))
    return out


@numba.njit
def draw_perm_reps_diff_var(x, y, size=1):
    """Generate array of difference in variance permuation replicates."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.abs(np.var(x_perm) - np.var(y_perm))
    return out

def hyp_test(df):
    # Extract data as numpy arrays
    time_labeled = df.loc[df["labeled"] == True, "time to catastrophe (s)"].values
    time_unlabeled = df.loc[df["labeled"] == False, "time to catastrophe (s)"].values
    
    # Compute difference in variance for original data set
    diff_var = np.abs(np.var(time_labeled) - np.var(time_unlabeled))

    # Draw replicates
    perm_reps_var = CI.draw_perm_reps_diff_var(time_labeled, time_unlabeled, size=1000000)

    # Compute difference in variance p-value
    p_val = np.sum(perm_reps_var >= diff_var) / len(perm_reps_var)
    print("Difference in variances p-value =", p_val)

    # Compute difference in mean for original data set
    diff_mean = np.abs(np.mean(time_labeled) - np.mean(time_unlabeled))

    # Draw replicates
    perm_reps_mean = CI.draw_perm_reps_diff_mean(time_labeled, time_unlabeled, size=1000000)

    # Compute difference in mean p-value
    p_val = np.sum(perm_reps_mean >= diff_mean) / len(perm_reps_mean)
    print("Difference in means p-value =", p_val)

    # Compute combined p-value
    p_val = np.sum((perm_reps_mean >= diff_mean) * (perm_reps_var >= diff_var)) / len(
        perm_reps_mean
    )

    print("Combined p-value =", p_val)

def mean_time_conf_int(df):
    # Extract data as numpy arrays
    time_labeled = df.loc[df["labeled"] == True, "time to catastrophe (s)"].values
    time_unlabeled = df.loc[df["labeled"] == False, "time to catastrophe (s)"].values
    
    # Calculate the necessary normal distribution params
    n_labeled = np.size(time_labeled)
    n_unlabeled = np.size(time_unlabeled)

    mean_labeled = np.mean(time_labeled)
    mean_unlabeled = np.mean(time_unlabeled)

    var_labeled = (1 / (n_labeled * (n_labeled - 1))) * np.sum(
        (time_labeled - mean_labeled) ** 2
    )
    var_unlabeled = (1 / (n_unlabeled * (n_unlabeled - 1))) * np.sum(
        (time_unlabeled - mean_unlabeled) ** 2
    )
    std_labeled = np.sqrt(var_labeled)
    std_unlabeled = np.sqrt(var_unlabeled)

    # Calculate 95% probability mass interval
    interval_labeled = scipy.stats.norm.interval(0.95, mean_labeled, std_labeled)
    interval_unlabeled = scipy.stats.norm.interval(0.95, mean_unlabeled, std_unlabeled)


    print("""
    Mean time to catastrophe for labeled tubulin 95% conf int (s):   [{0:.2f}, {1:.2f}]
    Mean time to catastrophe for unlabeled tubulin 95% conf int (s): [{2:.2f}, {3:.2f}]
    """.format(*(tuple(interval_labeled) + tuple(interval_unlabeled)))
    )

# Write own ECDF function
def ecdf(x, data):
    """Return the ECDF value for 1D data set at x."""
    # Get sorted data with number of instances of each data point
    value, count = np.unique(data, return_counts=True)

    # Calculate ECDF y coords with ratio of cumulative sum to total data
    ecdf = np.cumsum(count) / len(data)

    # Convert x to numpy array
    if isinstance(x, int):
        x = np.array([x])
    else:
        x = np.asarray(x)

    y = np.zeros(np.size(x))

    # Loop through lookup points in x
    for cnt, lookup in enumerate(x):
        # Find closest value in data
        index = (np.abs(value - lookup)).argmin()

        # Check if closest value is greater than lookup
        if lookup < value[index]:
            # If lookup isn't below the lowest data point shift index down
            if index != 0:
                index -= 1
            # Else set ECDF value to 0
            else:
                y[cnt] = 0
                continue
        # Set ECDF value
        y[cnt] = ecdf[index]

    # Return data as DataFrame
    df = pd.DataFrame({"x": x, "y": y})
    return df

def upper_lower_bounds(df):
    # Extract data as numpy arrays
    time_labeled = df.loc[df["labeled"] == True, "time to catastrophe (s)"].values
    time_unlabeled = df.loc[df["labeled"] == False, "time to catastrophe (s)"].values
    
    # Calculate the necessary normal distribution params
    n_labeled = np.size(time_labeled)
    n_unlabeled = np.size(time_unlabeled)
    
    alpha = 0.05
    eps = np.sqrt((1 / (2 * n_labeled)) * np.log(2 / alpha))

    # Calculate ECDF values for labeled tubulin
    ecdf_labeled = CI.ecdf(time_labeled, time_labeled)
    ecdf_labeled["bounds"] = "Data"

    # Calculate lower and upper bounds of 95% interval
    L_labeled = pd.DataFrame(
        {
            "x": ecdf_labeled.x,
            "y": [i - eps if i - eps > 0 else 0 for i in ecdf_labeled.y],
            "bounds": "Lower",
        }
    )
    U_labeled = pd.DataFrame(
        {
            "x": ecdf_labeled.x,
            "y": [i + eps if i + eps < 1 else 1 for i in ecdf_labeled.y],
            "bounds": "Upper",
        }
    )

    # Append the bounds to original DataFrame
    ecdf_labeled = pd.concat([ecdf_labeled, L_labeled, U_labeled], ignore_index=True)
    ecdf_labeled["label"] = "Labeled"

    # Repeat steps for unlabeled tubulin
    ecdf_unlabeled = CI.ecdf(time_unlabeled, time_unlabeled)
    ecdf_unlabeled["bounds"] = "Data"
    L_unlabeled = pd.DataFrame(
        {
            "x": ecdf_unlabeled.x,
            "y": [i - eps if i - eps > 0 else 0 for i in ecdf_unlabeled.y],
            "bounds": "Lower",
        }
    )
    U_unlabeled = pd.DataFrame(
        {
            "x": ecdf_unlabeled.x,
            "y": [i + eps if i + eps < 1 else 1 for i in ecdf_unlabeled.y],
            "bounds": "Upper",
        }
    )
    ecdf_unlabeled = pd.concat(
        [ecdf_unlabeled, L_unlabeled, U_unlabeled], ignore_index=True
    )
    ecdf_unlabeled["label"] = "Unlabeled"

    # Sort data for Curve plot
    ecdf_labeled = ecdf_labeled.sort_values(by=["x"])
    ecdf_unlabeled = ecdf_unlabeled.sort_values(by=["x"])
    ecdf_labeled = ecdf_labeled.reset_index(drop=True)
    ecdf_unlabeled = ecdf_unlabeled.reset_index(drop=True)

    # Plot all data and respective bounds
    plot_ecdf_func = hv.Curve(
        data=pd.concat([ecdf_labeled, ecdf_unlabeled]),
        kdims=["x", "y"],
        vdims=["bounds", "label"],
    ).groupby(["bounds", "label"]).options(
        xlabel="Catastrophe Time (sec)",
        ylabel="Cumulative Distribution",
        padding=0.05,
        #     alpha=1
    ).overlay()

    return plot_ecdf_func

