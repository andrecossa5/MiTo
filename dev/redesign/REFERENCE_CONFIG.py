"""MiTo simulation reference configuration (molecule-level draw)."""
SIM_MOL = dict(
    af_beta=(3, 57), af_beta_min=0.04, af_depth_decay=0.85,
    af_beta_noise=(2, 1500), af_negative_frac=0.002,
    mean_coverage=200, coverage_cell_cv=0.2, coverage_site_cv=0.8,
    mean_molecules=40, molecule_cv=0.5, molecule_coverage_exponent=0.0,
    overdispersion_background=1.2, overdispersion_noise_frac=0.5,
    noise_prevalence=(0.05, 0.2),
)
