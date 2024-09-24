
viridis = mpl.colormaps["viridis"].resampled(256)
newcolors = viridis(np.linspace(0,1, 256))
newcmp = ListedColormap(newcolors)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="7%", pad=0.04)
cbar = fig.colorbar(pc, cax=cax, extend="both")

cbar = fig.colorbar(pc, cax=cax)

label_format = r"${:.1f}$"
ticks_loc = np.linspace(-1, 1, 20)

cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
cbar.ax.set_yticklabels([label_format.format(x) for x in ticks_loc], fontsize=13)
cbar.ax.set_ylabel(r"$\mathcal{D}_{euc}(CV_i,CV_j)$", fontsize=13)
