#!/usr/bin/env python3

import os
import time
import pickle
import argparse
import jax
import jax.numpy as jnp

# Import custom argument parsing from utils
import utils

# Set JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)

# Import modules from jimgw
from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
    RayleighPrior,
    SimpleConstrainedPrior,
)
from jimgw.core.transforms import PeriodicTransform, BoundToUnbound
from jimgw.core.single_event.detector import detector_preset
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.likelihood import TransientLikelihoodFD, HeterodynedTransientLikelihoodFD
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    SphereSpinToCartesianSpinTransform,
    # SpinAnglesToCartesianSpinTransform,
)

def run_pe(args: argparse.Namespace):
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
    bilby_data_dump_path = f"../bilby_runs/outdir/{args.event_id}/data"
    bilby_data_file = os.listdir(bilby_data_dump_path)[0]
    with open(os.path.join(bilby_data_dump_path, bilby_data_file), "rb") as f:
        bilby_data_dump = pickle.load(f)

    # Extract chirp mass bounds and luminosity distance upper bound from priors
    Mc_lower = float(bilby_data_dump.priors_dict['chirp_mass'].minimum)
    Mc_upper = float(bilby_data_dump.priors_dict['chirp_mass'].maximum)
    print(f"Setting the Mc bounds to be {Mc_lower} and {Mc_upper}")

    dL_upper = float(bilby_data_dump.priors_dict['luminosity_distance'].maximum)
    print(f"Setting dL upper bound to be {dL_upper}")

    # -------------------------------
    # Load interferometer data from HDF5 files
    # -------------------------------

    bilby_ifos = bilby_data_dump.interferometers
    jim_ifos = [detector_preset[ifo.name] for ifo in bilby_ifos]
    gps = bilby_data_dump.trigger_time
    fmin = bilby_ifos[0].minimum_frequency
    fmax = bilby_ifos[0].maximum_frequency

    for ifo_b, ifo_j in zip(bilby_ifos, jim_ifos):
        delta_t = ifo_b.time_array[1] - ifo_b.time_array[0]

        # This line trigger the freq_domain_strain calculation
        # which then trigger the window factor computation
        # which is then multiply to the bilby psd_array as final output.
        _ = ifo_b.frequency_domain_strain

        # set analysis data
        jim_data = Data.from_fd(
            fd=ifo_b.frequency_domain_strain,
            frequencies=ifo_b.frequency_array,
            epoch=ifo_b.start_time,
            name=ifo_b.name + '_fd_data',)


        jim_psd = PowerSpectrum(
            values=ifo_b.power_spectral_density_array,
            frequencies=ifo_b.frequency_array,
            name=ifo_b.name + '_psd',
        )
        ifo_j.set_data(jim_data)
        ifo_j.set_psd(jim_psd)


    # -------------------------------
    # Select waveform based on event ID
    # -------------------------------
    ref_freq = float(bilby_data_dump.meta_data['command_line_args']['reference_frequency'])
    
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
    s1_prior = UniformSpherePrior(parameter_names=["s1"])
    s2_prior = UniformSpherePrior(parameter_names=["s2"])
    iota_prior = SinePrior(parameter_names=["iota"])
    prior += [s1_prior, s2_prior, iota_prior]
    # theta_jn_prior = SinePrior(parameter_names=["theta_jn"])
    # phi_jl_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phi_jl"])
    # tilt_1_prior = SinePrior(parameter_names=["tilt_1"])
    # tilt_2_prior = SinePrior(parameter_names=["tilt_2"])
    # phi_12_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phi_12"])
    # a_1_prior = UniformPrior(0.0, 1.0, parameter_names=["a_1"])
    # a_2_prior = UniformPrior(0.0, 1.0, parameter_names=["a_2"])
    # prior += [theta_jn_prior, phi_jl_prior, tilt_1_prior, tilt_2_prior, phi_12_prior, a_1_prior, a_2_prior]

    # Extrinsic priors
    dL_prior = SimpleConstrainedPrior([PowerLawPrior(1.0, dL_upper, 2.0, parameter_names=["d_L"])])
    t_c_prior = SimpleConstrainedPrior([UniformPrior(-0.1, 0.1, parameter_names=["t_c"])])
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
        # Transformations for luminosity distance
        DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=jim_ifos),

        # Transformations for phase
        GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=jim_ifos[0]),
        PeriodicTransform(name_mapping=(["periodic_4", "phase_det"], ["phase_det_x", "phase_det_y"]), xmin=0.0, xmax=2 * jnp.pi),
        # PeriodicTransform(name_mapping=(["periodic_4", "phase_c"], ["phase_c_x", "phase_c_y"]), xmin=0.0, xmax=2 * jnp.pi),

        # Transformations for time
        GeocentricArrivalTimeToDetectorArrivalTimeTransform(gps_time=gps, ifo=jim_ifos[0]),

        # Transformations for sky position
        SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=jim_ifos),
        BoundToUnbound(name_mapping=(["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        PeriodicTransform(name_mapping=(["periodic_3", "azimuth"], ["azimuth_x", "azimuth_y"]), xmin=0.0, xmax=2 * jnp.pi),

        # Transformations for polarization angle
        PeriodicTransform(name_mapping=(["periodic_5", "psi"], ["psi_base_x", "psi_base_y"]), xmin=0.0, xmax=jnp.pi),

        # Transformations for masses
        BoundToUnbound(name_mapping=(["M_c"], ["M_c_unbounded"]), original_lower_bound=Mc_lower, original_upper_bound=Mc_upper),
        BoundToUnbound(name_mapping=(["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max),

        # Transformations for spins
        BoundToUnbound(name_mapping=(["iota"], ["iota_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        PeriodicTransform(name_mapping=(["periodic_1", "s1_phi"], ["s1_phi_base_x", "s1_phi_base_y"]), xmin=0.0, xmax=2 * jnp.pi),
        PeriodicTransform(name_mapping=(["periodic_2", "s2_phi"], ["s2_phi_base_x", "s2_phi_base_y"]), xmin=0.0, xmax=2 * jnp.pi),
        BoundToUnbound(name_mapping=(["s1_theta"], ["s1_theta_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        BoundToUnbound(name_mapping=(["s2_theta"], ["s2_theta_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        BoundToUnbound(name_mapping=(["s1_mag"], ["s1_mag_unbounded"]), original_lower_bound=0.0, original_upper_bound=0.99),
        BoundToUnbound(name_mapping=(["s2_mag"], ["s2_mag_unbounded"]), original_lower_bound=0.0, original_upper_bound=0.99),
        # BoundToUnbound(name_mapping=(["theta_jn"], ["theta_jn_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        # PeriodicTransform(name_mapping=(["periodic_1", "phi_jl"], ["phi_jl_x", "phi_jl_y"]), xmin=0.0, xmax=2 * jnp.pi),
        # BoundToUnbound(name_mapping=(["tilt_1"], ["tilt_1_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        # BoundToUnbound(name_mapping=(["tilt_2"], ["tilt_2_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        # PeriodicTransform(name_mapping=(["periodic_2", "phi_12"], ["phi_12_x", "phi_12_y"]), xmin=0.0, xmax=2 * jnp.pi),
        # BoundToUnbound(name_mapping=(["a_1"], ["a_1_unbounded"]), original_lower_bound=0.0, original_upper_bound=1.0),
        # BoundToUnbound(name_mapping=(["a_2"], ["a_2_unbounded"]), original_lower_bound=0.0, original_upper_bound=1.0),
    ]

    # Likelihood transforms
    likelihood_transforms = [
        SphereSpinToCartesianSpinTransform("s1"),
        SphereSpinToCartesianSpinTransform("s2"),
        # SpinAnglesToCartesianSpinTransform(freq_ref=ref_freq),
        MassRatioToSymmetricMassRatioTransform,
    ]

    # -------------------------------
    # Setup Likelihood based on relative binning flag
    # -------------------------------
    if args.relative_binning:
        print("Using relative binning")
        likelihood = HeterodynedTransientLikelihoodFD(
            jim_ifos, waveform=waveform, n_bins=1_000, trigger_time=gps,
            prior=prior, sample_transforms=sample_transforms,
            likelihood_transforms=likelihood_transforms,
            f_min=fmin, f_max=fmax
        )
    else:
        print("Using normal likelihood")
        likelihood = TransientLikelihoodFD(
            jim_ifos, waveform=waveform, trigger_time=gps,
            f_min=fmin, f_max=fmax,
        )

    # -------------------------------
    # Configure the sampler and training parameters
    # -------------------------------
    mass_matrix = jnp.eye(prior.n_dims)

    jim = Jim(
        likelihood,
        prior,
        sample_transforms=sample_transforms,
        likelihood_transforms=likelihood_transforms,
        rng_key=jax.random.PRNGKey(123),
        n_chains=500,
        n_local_steps=100,
        n_global_steps=1000,
        n_training_loops=100,
        n_production_loops=10,
        n_epochs=20,
        mala_step_size=mass_matrix * 2e-3,
        rq_spline_hidden_units=[128, 128],
        rq_spline_n_bins=10,
        rq_spline_n_layers=8,
        learning_rate=1e-3,
        batch_size=10000,
        n_max_examples=10000,
        n_NFproposal_batch_size=5,
        local_thinning=1,
        global_thinning=1,
        verbose=True,
    )

    # Run the sampler
    jim.sample()
    jim.print_summary()

    # -------------------------------
    # Postprocessing: Save samples and evaluate log probabilities
    # -------------------------------
    total_time_end = time.time()
    print(f"Time taken: {total_time_end - total_time_start} seconds = {(total_time_end - total_time_start) / 60} minutes")

    # ---------------------------------
    # Save samples, results, and chains
    # ---------------------------------
    samples = jim.get_samples()
    samples = {key: samples[key] for key in samples.keys()}
    jim_prior = jax.vmap(prior.log_prob)(samples)

    resources = jim.sampler.resources
    logprob_train = resources["log_prob_training"].data
    logprob_prod = resources["log_prob_production"].data
    local_acceptance_train = resources["local_accs_training"].data
    local_acceptance_prod = resources["local_accs_production"].data
    global_acceptance_train = resources["global_accs_training"].data
    global_acceptance_prod = resources["global_accs_production"].data
    chains = resources["positions_production"].data

    jnp.savez(f"{event_outdir}/results.npz",
            log_prob_training=logprob_train,
            log_prob_production=logprob_prod,
            local_accs_training=local_acceptance_train,
            local_accs_production=local_acceptance_prod,
            global_accs_training=global_acceptance_train,
            global_accs_production=global_acceptance_prod,
            chains=chains,
            samples=samples,
            log_prior=jim_prior,
            )

def main():
    """Entry point: parse arguments and run the parameter estimation."""
    run_pe(utils.get_parser().parse_args())

if __name__ == "__main__":
    main()
