def Hello(x):
    print(x)

def biplot(pca, scaling = 0, plot_loading_labels = True, color = None, alpha_scores = 1):
    scores, loadings = eigen_scaling(pca, scaling=scaling)
    # Plot scores
    if color is None:
        sns.relplot(
            x = "comp_0",
            y = "comp_1",
            palette = "muted",
            alpha = alpha_scores,
            data = scores_,
        )
    else:
        scores_ = scores.copy()
        scores_["group"] = color
        sns.relplot(
            x = "comp_0",
            y = "comp_1",
            hue = "group",
            palette = "muted",
            alpha = alpha_scores,
            data = scores_,
        )

    # Plot loadings
    if plot_loading_labels:
        loading_labels = pca.loadings.index

    for i in range(loadings.shape[0]):
        plt.arrow(
            0, 0,
            loadings.iloc[i, 0],
            loadings.iloc[i, 1],
            color = 'black',
            alpha = 0.7,
            linestyle = '-',
            head_width = loadings.values.max() / 50,
            width = loadings.values.max() / 2000,
            length_includes_head = True
        )
        if plot_loading_labels:
            plt.text(
                loadings.iloc[i, 0]*1.05,
                loadings.iloc[i, 1]*1.05,
                loading_labels[i],
                color = 'black',
                ha = 'center',
                va = 'center',
                fontsize = 10
            );

    # range of the plot
    scores_loadings = np.vstack([scores.values[:, :2], loadings.values[:, :2]])
    xymin = scores_loadings.min(axis=0) * 1.2
    xymax = scores_loadings.max(axis=0) * 1.2

    plt.axhline(y = 0, color = 'k', linestyle = 'dotted', linewidth=0.75)
    plt.axvline(x = 0, color = 'k', linestyle = 'dotted', linewidth=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(xymin[0], xymax[0])
    plt.ylim(xymin[1], xymax[1]);