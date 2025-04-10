#!/usr/bin/env python3

import os
import time
import pickle
import ast
import numpy as np
import argparse
import jax
import jax.numpy as jnp

# Import custom argument parsing from utils
import utils

# Set JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)

# Import modules from jimgw
from jimgw.jim import Jim
from jimgw.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    # UniformSpherePrior,
    RayleighPrior,
)
from jimgw.transforms import PeriodicTransform, BoundToUnbound
from jimgw.single_event.detector import H1, L1, V1, GroundBased2G
from jimgw.single_event.likelihood import TransientLikelihoodFD, HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
from jimgw.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
    # DistanceToSNRWeightedDistanceTransform,
    # GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    # GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    # SphereSpinToCartesianSpinTransform,
    SpinAnglesToCartesianSpinTransform,
)

def run_pe(args: argparse.Namespace, verbose: bool = False):
    """
    Main function to run parameter estimation.
    - Loads the bilby_pipe DataDump.
    - Sets up interferometer data.
    - Configures waveform, priors, and transforms.
    - Runs the Jim sampler.
    - Saves and plots the results.
    """
    total_time_start = time.time()
    print("Arguments:")
    print(args)

    # Create output directory if it does not exist
    event_outdir = os.path.join(args.outdir, args.event_id)
    if not os.path.exists(event_outdir):
        os.makedirs(event_outdir)

    # -------------------------------
    # Load bilby_pipe DataDump
    # -------------------------------
    print(f"Fetching bilby_pipe DataDump for event {args.event_id}")
    data_dump_path = f"../bilby_runs/outdir/{args.event_id}/data"
    data_file = os.listdir(data_dump_path)[0]
    with open(os.path.join(data_dump_path, data_file), "rb") as f:
        data_dump = pickle.load(f)

    # Extract chirp mass bounds and luminosity distance upper bound from priors
    Mc_lower = float(data_dump.priors_dict['chirp_mass'].minimum)
    Mc_upper = float(data_dump.priors_dict['chirp_mass'].maximum)
    print(f"Setting the Mc bounds to be {Mc_lower} and {Mc_upper}")

    dL_upper = float(data_dump.priors_dict['luminosity_distance'].maximum)
    print(f"The dL upper bound is {dL_upper}")

    if verbose:
        print("Metadata:")
        print(data_dump.meta_data)

    # Compute duration and get trigger info
    duration = float(data_dump.interferometers[0].strain_data.time_array[-1] -
                     data_dump.interferometers[0].strain_data.time_array[0])
    post_trigger = float(data_dump.meta_data['command_line_args']['post_trigger_duration'])
    gps = float(data_dump.trigger_time)
    fmin = ast.literal_eval(data_dump.meta_data['command_line_args']['minimum_frequency'])
    fmax = ast.literal_eval(data_dump.meta_data['command_line_args']['maximum_frequency'])

    # -------------------------------
    # Frequency handling for interferometers
    # -------------------------------
    ifos_list_string = list(data_dump.interferometers.meta_data.keys())
    ifos_diff_fmin = []
    try:
        # Check if all interferometers have the same fmin
        if all(fmin[ifos_list_string[0]] == fmin[ifo] for ifo in ifos_list_string[1:]):
            print("fmin is the same for all ifos")
            fmin = fmin[ifos_list_string[0]]
        else:
            print("fmin is different for different ifos")
            ifos_diff_fmin = [ifo for ifo in ifos_list_string if fmin[ifo] != min(fmin.values())]
            fmin = min(fmin.values())
    except Exception:
        fmin = float(fmin)

    ifos_diff_fmax = []
    try:
        # Check if all interferometers have the same fmax
        if all(fmax[ifos_list_string[0]] == fmax[ifo] for ifo in ifos_list_string[1:]):
            print("fmax is the same for all ifos")
            fmax = fmax[ifos_list_string[0]]
        else:
            print("fmax is different for different ifos")
            ifos_diff_fmax = [ifo for ifo in ifos_list_string if fmax[ifo] != max(fmax.values())]
            fmax = max(fmax.values())
    except Exception:
        fmax = float(fmax)

    if verbose:
        print("fmin:")
        print(fmin)
        print("fmax:")
        print(fmax)

    # -------------------------------
    # Load interferometer data from HDF5 files
    # -------------------------------
    ifos: list[GroundBased2G] = []
    for i, ifo_string in enumerate(ifos_list_string):
        # Ensure the interferometer name matches the metadata key
        ifo_bilby = data_dump.interferometers[i]
        assert ifo_bilby.name == ifo_string, f"ifo_bilby.name: {ifo_bilby.name} != ifo_string: {ifo_string}"
        print("Adding interferometer", ifo_string)
        # Append the interferometer instance using eval (assumes ifo_string is defined globally)
        eval(f'ifos.append({ifo_string})')

        # Apply frequency mask to data, PSD, and frequencies
        frequencies = ifo_bilby.frequency_array
        data = ifo_bilby.frequency_domain_strain
        psd = ifo_bilby.power_spectral_density_array
        mask = (frequencies >= fmin) & (frequencies <= fmax)
        frequencies = frequencies[mask]
        data = data[mask]
        psd = psd[mask]

        # Set attributes for the interferometer object
        ifos[i].frequencies = frequencies
        ifos[i].data = data
        ifos[i].psd = psd

        # Handle different fmin and fmax per interferometer if necessary
        if ifo_string in ifos_diff_fmin:
            print(f"Setting frequencies below {fmin} to inf for {ifo_string}")
            ifos[i].psd = jnp.where(frequencies < fmin, jnp.inf, ifos[i].psd)
        if ifo_string in ifos_diff_fmax:
            print(f"Setting frequencies above {fmax} to inf for {ifo_string}")
            ifos[i].psd = jnp.where(frequencies > fmax, jnp.inf, ifos[i].psd)

        if verbose:
            print(f"Checking data for {ifo_string}")
            print(f"Data shape: {ifos[i].data.shape}")
            print(f"PSD shape: {ifos[i].psd.shape}")
            print(f"Frequencies shape: {ifos[i].frequencies.shape}")
            print("Data:", ifos[i].data)
            print("PSD:", ifos[i].psd)
            print("Frequencies:", ifos[i].frequencies)

    if verbose:
        print(f"Running PE on event {args.event_id}")
        print(f"Duration: {duration}")
        print(f"GPS: {gps}")
        print(f"Chirp mass: [{Mc_lower}, {Mc_upper}]")

    # -------------------------------
    # Select waveform based on event ID
    # -------------------------------
    ref_freq = float(data_dump.meta_data['command_line_args']['reference_frequency'])
    
    print("Using IMRPhenomPv2 waveform")
    waveform = RippleIMRPhenomPv2(f_ref=ref_freq)

    # -------------------------------
    # Set up priors for parameters
    # -------------------------------
    prior = []

    # Mass priors
    Mc_prior = UniformPrior(Mc_lower, Mc_upper, parameter_names=["M_c"])
    q_min, q_max = 0.125, 1.0
    q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])
    prior += [Mc_prior, q_prior]

    # Spin priors
    # s1_prior = UniformSpherePrior(parameter_names=["s1"])
    # s2_prior = UniformSpherePrior(parameter_names=["s2"])
    # iota_prior = SinePrior(parameter_names=["iota"])
    # prior += [s1_prior, s2_prior, iota_prior]
    theta_jn_prior = SinePrior(parameter_names=["theta_jn"])
    phi_jl_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phi_jl"])
    tilt_1_prior = SinePrior(parameter_names=["tilt_1"])
    tilt_2_prior = SinePrior(parameter_names=["tilt_2"])
    phi_12_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phi_12"])
    a_1_prior = UniformPrior(0.0, 1.0, parameter_names=["a_1"])
    a_2_prior = UniformPrior(0.0, 1.0, parameter_names=["a_2"])
    prior += [theta_jn_prior, phi_jl_prior, tilt_1_prior, tilt_2_prior, phi_12_prior, a_1_prior, a_2_prior]

    # Extrinsic priors
    dL_prior = PowerLawPrior(1.0, dL_upper, 2.0, parameter_names=["d_L"])
    t_c_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
    phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
    psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
    ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
    dec_prior = CosinePrior(parameter_names=["dec"])
    prior += [dL_prior, t_c_prior, phase_c_prior, psi_prior, ra_prior, dec_prior]

    # Extra priors for periodic parameters
    prior += [
        RayleighPrior(1.0, parameter_names=["periodic_1"]),
        RayleighPrior(1.0, parameter_names=["periodic_2"]),
        RayleighPrior(1.0, parameter_names=["periodic_3"]),
        RayleighPrior(1.0, parameter_names=["periodic_4"]),
        RayleighPrior(1.0, parameter_names=["periodic_5"]),
    ]

    # Combine all priors into a single prior object
    prior = CombinePrior(prior)

    # -------------------------------
    # Define sample and likelihood transforms
    # -------------------------------
    sample_transforms = [
        # Transformations for masses
        BoundToUnbound(name_mapping=(["M_c"], ["M_c_unbounded"]), original_lower_bound=Mc_lower, original_upper_bound=Mc_upper),
        BoundToUnbound(name_mapping=(["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max),

        # Transformations for spins
        # BoundToUnbound(name_mapping=(["iota"], ["iota_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        # PeriodicTransform(name_mapping=(["periodic_1", "s1_phi"], ["s1_phi_base_x", "s1_phi_base_y"]), xmin=0.0, xmax=2 * jnp.pi),
        # PeriodicTransform(name_mapping=(["periodic_2", "s2_phi"], ["s2_phi_base_x", "s2_phi_base_y"]), xmin=0.0, xmax=2 * jnp.pi),
        # BoundToUnbound(name_mapping=(["s1_theta"], ["s1_theta_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        # BoundToUnbound(name_mapping=(["s2_theta"], ["s2_theta_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        # BoundToUnbound(name_mapping=(["s1_mag"], ["s1_mag_unbounded"]), original_lower_bound=0.0, original_upper_bound=0.99),
        # BoundToUnbound(name_mapping=(["s2_mag"], ["s2_mag_unbounded"]), original_lower_bound=0.0, original_upper_bound=0.99),
        BoundToUnbound(name_mapping=(["theta_jn"], ["theta_jn_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        PeriodicTransform(name_mapping=(["periodic_1", "phi_jl"], ["phi_jl_x", "phi_jl_y"]), xmin=0.0, xmax=2 * jnp.pi),
        BoundToUnbound(name_mapping=(["tilt_1"], ["tilt_1_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        BoundToUnbound(name_mapping=(["tilt_2"], ["tilt_2_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        PeriodicTransform(name_mapping=(["periodic_2", "phi_12"], ["phi_12_x", "phi_12_y"]), xmin=0.0, xmax=2 * jnp.pi),
        BoundToUnbound(name_mapping=(["a_1"], ["a_1_unbounded"]), original_lower_bound=0.0, original_upper_bound=1.0),
        BoundToUnbound(name_mapping=(["a_2"], ["a_2_unbounded"]), original_lower_bound=0.0, original_upper_bound=1.0),

        # Transformations for sky position
        SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
        BoundToUnbound(name_mapping=(["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        PeriodicTransform(name_mapping=(["periodic_3", "azimuth"], ["azimuth_x", "azimuth_y"]), xmin=0.0, xmax=2 * jnp.pi),

        # Transformations for luminosity distance
        # DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
        BoundToUnbound(name_mapping=(["d_L"], ["d_L_unbounded"]), original_lower_bound=1.0, original_upper_bound=dL_upper),

        # Transformations for time
        # GeocentricArrivalTimeToDetectorArrivalTimeTransform(tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]),
        BoundToUnbound(name_mapping=(["t_c"], ["t_c_unbounded"]), original_lower_bound=-0.1, original_upper_bound=0.1),

        # Transformations for phase
        # GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
        # PeriodicTransform(name_mapping=(["periodic_4", "phase_det"], ["phase_det_x", "phase_det_y"]), xmin=0.0, xmax=2 * jnp.pi),
        PeriodicTransform(name_mapping=(["periodic_4", "phase_c"], ["phase_c_x", "phase_c_y"]), xmin=0.0, xmax=2 * jnp.pi),

        # Transformations for polarization angle
        PeriodicTransform(name_mapping=(["periodic_5", "psi"], ["psi_base_x", "psi_base_y"]), xmin=0.0, xmax=jnp.pi),
    ]

    # Likelihood transforms
    likelihood_transforms = [
        MassRatioToSymmetricMassRatioTransform,
        # SphereSpinToCartesianSpinTransform("s1"),
        # SphereSpinToCartesianSpinTransform("s2"),
        SpinAnglesToCartesianSpinTransform(freq_ref=ref_freq),
    ]

    # -------------------------------
    # Setup Likelihood based on relative binning flag
    # -------------------------------
    if args.relative_binning:
        print("Using relative binning")
        likelihood = HeterodynedTransientLikelihoodFD(
            ifos, waveform=waveform, n_bins=1_000, trigger_time=gps,
            duration=duration, post_trigger_duration=post_trigger,
            prior=prior, sample_transforms=sample_transforms,
            likelihood_transforms=likelihood_transforms,
            popsize=10, n_steps=50
        )
    else:
        print("Using normal likelihood")
        likelihood = TransientLikelihoodFD(
            ifos, waveform=waveform, trigger_time=gps,
            duration=duration, post_trigger_duration=post_trigger
        )

    # -------------------------------
    # Configure the sampler and training parameters
    # -------------------------------
    mass_matrix = jnp.eye(prior.n_dim)
    local_sampler_arg = {"step_size": mass_matrix * 3e-3}
    n_loop_training = 100
    n_epochs = 10
    learning_rate = 1e-3

    jim = Jim(
        likelihood,
        prior,
        sample_transforms=sample_transforms,
        likelihood_transforms=likelihood_transforms,
        n_loop_training=n_loop_training,
        n_loop_production=10,
        n_local_steps=100,
        n_global_steps=1_000,
        n_chains=1_000,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        n_max_examples=30000,
        n_flow_sample=100000,
        momentum=0.9,
        batch_size=30000,
        use_global=True,
        keep_quantile=0.0,
        train_thinning=1,
        output_thinning=10,
        local_sampler_arg=local_sampler_arg,
    )

    # Run the sampler
    rng_key = jax.random.PRNGKey(123)
    jim.sample(rng_key)
    jim.print_summary()

    # -------------------------------
    # Postprocessing: Save samples and evaluate log probabilities
    # -------------------------------
    total_time_end = time.time()
    print(f"Time taken: {total_time_end - total_time_start} seconds = {(total_time_end - total_time_start) / 60} minutes")

    samples = jim.get_samples()
    downsample_indices = np.random.choice(
        samples[list(samples.keys())[0]].shape[0], 5000
    )
    samples = {key: samples[key][downsample_indices] for key in samples.keys()}
    np.savez(os.path.join(event_outdir, "samples.npz"), **samples)

    log_prob = jim.sampler.get_sampler_state(training=False)['log_prob'].reshape(-1)[downsample_indices]
    log_prior = jax.vmap(prior.log_prob)(samples)[downsample_indices]
    np.savez(os.path.join(event_outdir, "result.npz"), log_prior=log_prior, log_prob=log_prob)

    # Sample from the flow and reverse transforms for final samples
    rng_key, subkey = jax.random.split(rng_key)
    nf_samples = np.array(jim.sampler.sample_flow(subkey, 5000))
    nf_samples = jax.vmap(jim.add_name)(nf_samples)
    for transform in reversed(sample_transforms):
        nf_samples = jax.vmap(transform.backward)(nf_samples)
    jnp.savez(os.path.join(event_outdir, "nf_samples.npz"), **nf_samples)

    # -------------------------------
    # Plotting diagnostics
    # -------------------------------
    import matplotlib.pyplot as plt

    # Retrieve training state from sampler
    out_train = jim.sampler.get_sampler_state(training=True)
    chains = np.array(out_train["chains"])
    global_accs = np.array(out_train["global_accs"])
    local_accs = np.array(out_train["local_accs"])
    loss_vals = np.array(out_train["loss_vals"])

    # Plot 2d projection of two chains and diagnostic metrics
    axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
    plt.sca(axs[0])
    plt.title("2d projection of 2 chains")
    plt.plot(chains[0, :, 0], chains[0, :, 1], "o-", alpha=0.5, ms=2)
    plt.plot(chains[1, :, 0], chains[1, :, 1], "o-", alpha=0.5, ms=2)
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")

    plt.sca(axs[1])
    plt.title("NF loss")
    plt.plot(loss_vals.reshape(-1))
    plt.xlabel("iteration")

    plt.sca(axs[2])
    plt.title("Local Acceptance")
    plt.plot(local_accs.mean(0))
    plt.xlabel("iteration")

    plt.sca(axs[3])
    plt.title("Global Acceptance")
    plt.plot(global_accs.mean(0))
    plt.xlabel("iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(event_outdir, "training.jpg"))

    # Plot log probability evolution during production
    log_prob_prod = jim.sampler.get_sampler_state(training=False)['log_prob']
    plt.figure()
    mean = log_prob_prod.mean(0)
    std = log_prob_prod.std(0)
    plt.plot(mean)
    plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.3)
    plt.xlabel("iteration")
    plt.ylabel("log_prob")
    plt.title("Production log_prob")
    plt.savefig(os.path.join(event_outdir, "log_prob.jpg"))

def main():
    """Entry point: parse arguments and run the parameter estimation."""
    parser = utils.get_parser()
    args = parser.parse_args()
    run_pe(args)

if __name__ == "__main__":
    main()
