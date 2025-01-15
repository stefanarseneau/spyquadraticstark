import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stefan.mplstyle'))

def plot_with_shaded_error(ax, x, y, yerr, plot_kwargs, scatter_kwargs, fill_kwargs):
    x = np.array(x)
    y = np.array(y)

    if isinstance(yerr, tuple):
        lower_error, upper_error = yerr
        lower_bound = y - np.array(lower_error)
        upper_bound = y + np.array(upper_error)
    else:
        lower_bound = y - np.array(yerr)
        upper_bound = y + np.array(yerr)

    #if color is None:
    #    color = next(plt.gca()._get_lines.prop_cycler)['color']

    ax.plot(x, y, **plot_kwargs)
    ax.scatter(x, y, **scatter_kwargs)
    ax.fill_between(x, lower_bound, upper_bound, **fill_kwargs)

def parameter_stark_plot(allstark, paramdata, plot_zzceti = True,
                        x_label = r'$T_{eff}$ [$K$]', xlims = (7500, 30000), ylims = (-50, 50),
                        data_kwargs = {'c' : 'k', 'fmt' : 'o', 'capsize' : 4, 'alpha' : 0.1},
                        plot_kwargs = {'color' : 'k'}, 
                        scatter_kwargs = {'label' : 'data', 'color' : 'k', 'marker' : 'o'}, 
                        fill_kwargs = {'color' : 'k', 'alpha' : 0.2},
                        legend_kwargs = {'framealpha' : 0, 'loc' : 'upper right'}):
    # read in the data tuples
    mean, bin_stark, e_bin_stark = paramdata
    teff, stark, e_stark = allstark
    # begin plotting
    fig, ax = plt.subplots(ncols = 1, figsize=(8,6))
    plot_with_shaded_error(ax, mean, bin_stark, yerr=e_bin_stark, plot_kwargs=plot_kwargs, scatter_kwargs=scatter_kwargs, fill_kwargs=fill_kwargs)
    ax.errorbar(teff, stark, yerr=e_stark, **data_kwargs)
    ax.axhline(y=0, c='k', ls = '--')
    ax.set_ylabel(r'$v_{Stark}$ $[km/s]$')
    ax.set_xlabel(x_label)
    y_min, y_max = ax.get_ylim()
    if plot_zzceti:
        ax.fill_betweenx([y_min, y_max], 10500, 12500, color="red", alpha=0.5, zorder=0, label='ZZ Ceti Instability\nStrip (Approx)')
    ax.legend(**legend_kwargs)
    ax.set_ylim(y_min,y_max)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    return fig, ax

