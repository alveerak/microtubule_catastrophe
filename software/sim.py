import numpy as np
import pandas as pd

import scipy.stats

import bokeh.io
import holoviews as hv
import bokeh_catplot
import ECDF

import bebi103

hv.extension("bokeh")
bokeh.io.output_notebook()
bebi103.hv.set_defaults()

def simulation(b1, b2, df_total):
    # Loop through values of both betas
    for i in range(len(b1)):
        # Generate 150 events of consecutive poisson processes
        for j in range(150):
            b1_t = scipy.stats.expon.rvs(loc=0, scale=1 / b1[i])
            b2_t = scipy.stats.expon.rvs(loc=0, scale=1 / b2[i])
            c_t = b1_t + b2_t

            # Create temporary dataframe with info to append to external dataframe
            d_temp = {
                "B1": b1[i],
                "B2": b2[i],
                "B2/B1": ["{0}/{1}".format(b2[i], b1[i])],
                "B1 Time": [b1_t],
                "B2 Time": [b2_t],
                "C Time": [c_t],
            }
            df = pd.DataFrame(d_temp)
            df_total = df_total.append(df, ignore_index=True)
    return df_total

def plot_sim_ECDF(df_total):
    # Plot ECDF of cumulative times of consecutive Poisson events
    plt = bokeh_catplot.ecdf(
        cats=["B2/B1"],
        data=df_total,
        val="C Time",
        style="staircase",
        title="Consecutive Poisson Processes",
    )
    plt.xaxis.axis_label = "Cumulative Time (β1^-1)"
    plt.legend.title = "β2/β1"
    bokeh.io.show(plt)


def two_poisson_cdf(x, b1, b2):
    """
    Return theoretical CDF of two sequential Poisson processes with
    rate of arrivals b1 and b2 over the range x.
    """
    coeff = b1 * b2 / (b2 - b1)
    term1 = (1 / b1) * (1 - np.exp(-b1 * x))
    term2 = (1 / b2) * (1 - np.exp(-b2 * x))
    return coeff * (term1 - term2)

def compare_plots(b1, b2, df_total):
    # Create ECDF from randomly generated cumulative times for chosen beta pair
    df_subset = df_total.loc[(df_total["B1"] == b1) & (df_total["B2"] == b2)]
    df_ecdf = ECDF.ecdf_vals(df_subset["C Time"])
    df_ecdf["Type"] = "Random"

    # Create ECDF dataframe of theoretical CDF in same range
    t_max = np.round(df_subset["C Time"].max(), decimals=2)
    t = np.linspace(0, t_max, 150)
    cdf = two_poisson_cdf(t, b1, b2)
    d_theory = {"x": t, "y": cdf, "Type": "Theory"}
    df_theory = pd.DataFrame(d_theory)

    # Combine both ECDF dataframes
    df_ecdf = df_ecdf.append(df_theory, ignore_index=True)

    # Plot both ECDFs
    overlay = hv.Curve(
        data=df_ecdf,
        kdims=["x", "y"],
        vdims=["Type"]
    ).groupby(
        "Type"
    ).options(
        title="Two Poisson ECDF (β1 = 100, β2 = 200)",
        xlabel="Catastrophe Time (sec)",
        ylabel="ECDF",
    ).overlay(
    )
    return overlay