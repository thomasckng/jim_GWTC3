#!/usr/bin/env python3

import jax

# Set JAX to use 64-bit precision
jax.config.update("jax_enable_x64", True)

import os

# Set JAX to use 95% of the available memory
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import time
import pickle
import argparse
import matplotlib.pyplot as plt
import jax.numpy as jnp
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
from bilby.gw.result import CBCResult

# Import custom argument parsing from utils
import utils

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
from jimgw.core.single_event.detector import get_detector_preset
from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.likelihood import (
    BaseTransientLikelihoodFD,
    HeterodynedTransientLikelihoodFD,
)
from jimgw.core.single_event.waveform import RippleIMRPhenomPv2
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
    SphereSpinToCartesianSpinTransform,
    SpinAnglesToCartesianSpinTransform,
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

    # -------------------------------
    # Load bilby_pipe DataDump
    # -------------------------------
    print(f"Fetching bilby_pipe DataDump for event {args.event_id}")
    bilby_data_dump_path = f"../bilby_runs/outdir/{args.event_id}/data"
    bilby_data_file = os.listdir(bilby_data_dump_path)[0]
    with open(os.path.join(bilby_data_dump_path, bilby_data_file), "rb") as f:
        bilby_data_dump = pickle.load(f)

    # Extract chirp mass bounds and luminosity distance upper bound from priors
    Mc_lower = float(bilby_data_dump.priors_dict["chirp_mass"].minimum)
    Mc_upper = float(bilby_data_dump.priors_dict["chirp_mass"].maximum)
    print(f"Setting the Mc bounds to be {Mc_lower} and {Mc_upper}")

    dL_upper = float(bilby_data_dump.priors_dict["luminosity_distance"].maximum)
    print(f"Setting dL upper bound to be {dL_upper}")

    # -------------------------------
    # Load interferometer data from bilby_pipe DataDump
    # -------------------------------

    bilby_ifos = bilby_data_dump.interferometers
    jim_ifos = [get_detector_preset()[ifo.name] for ifo in bilby_ifos]
    gps = bilby_data_dump.trigger_time
    fmins = [ifo.minimum_frequency for ifo in bilby_ifos]
    fmaxs = [ifo.maximum_frequency for ifo in bilby_ifos]

    if not all(f == fmins[0] for f in fmins):
        print("Warning: Not all ifos have the same minimum frequency. Using min(fmin).")
    if not all(f == fmaxs[0] for f in fmaxs):
        print("Warning: Not all ifos have the same maximum frequency. Using max(fmax).")

    fmin = min(fmins)
    fmax = max(fmaxs)

    for ifo_b, ifo_j in zip(bilby_ifos, jim_ifos):
        if args.use_bilby_psd:
            print(f"Using bilby PSD for {ifo_b.name}")
            # This line triggers the freq_domain_strain calculation
            # which then triggers the window factor computation
            # which is then multiplied to the bilby psd_array as final output.
            _ = ifo_b.frequency_domain_strain

            jim_psd = PowerSpectrum(
                values=ifo_b.power_spectral_density_array,
                frequencies=ifo_b.frequency_array,
                name=ifo_b.name + "_psd",
            )
            ifo_j.set_psd(jim_psd)
        else:
            print(f"Computing PSD from GWOSC data for {ifo_b.name}")
            # Load PSD data from GWOSC
            psd_start = bilby_data_dump.meta_data['command_line_args']['psd_start_time']
            psd_duration = float(bilby_data_dump.meta_data['command_line_args']['psd_length']) * ifo_b.duration
            psd_start = ifo_b.start_time - psd_duration if psd_start is None else float(psd_start)

            psd_data = Data.from_gwosc(ifo_b.name, psd_start, psd_start + psd_duration)
            if jnp.isnan(psd_data.td).any():
                raise ValueError(
                    f"PSD data for {ifo_b.name} contains NaNs."
                )
            
            alpha = 2 * float(bilby_data_dump.meta_data['command_line_args']['tukey_roll_off']) / ifo_b.duration
            psd_data.set_tukey_window(alpha=alpha)

            # set an NFFT corresponding to the analysis segment duration
            psd_fftlength = ifo_b.duration * ifo_b.sampling_frequency
            psd_noverlap = psd_fftlength * float(bilby_data_dump.meta_data['command_line_args']['psd_fractional_overlap'])
            ifo_j.set_psd(psd_data.to_psd(nperseg=psd_fftlength, noverlap=psd_noverlap, average='median'))

        if args.use_bilby_data:
            print(f"Using bilby frequency domain strain data for {ifo_b.name}")
            # This line triggers the freq_domain_strain calculation
            # which then triggers the window factor computation
            # which is then multiplied to the bilby psd_array as final output.
            _ = ifo_b.frequency_domain_strain

            # set analysis data from bilby
            jim_data = Data.from_fd(
                fd=ifo_b.frequency_domain_strain,
                frequencies=ifo_b.frequency_array,
                epoch=ifo_b.start_time,
                name=ifo_b.name + "_fd_data",
            )
            ifo_j.set_data(jim_data)
        else:
            print(f"Loading analysis data from GWOSC for {ifo_b.name}")
            # set analysis data from GWOSC
            data = Data.from_gwosc(ifo_b.name, ifo_b.start_time, ifo_b.start_time + ifo_b.duration)
            ifo_j.set_data(data)

    # -------------------------------
    # Compare data and PSD between bilby and jim
    # -------------------------------
    psd_source = "bilby" if args.use_bilby_psd else "GWOSC"
    data_source = "bilby" if args.use_bilby_data else "GWOSC"
    print(f"Creating comparison plots between bilby and jim data/PSD...")
    print(f"Jim PSD source: {psd_source}, Jim data source: {data_source}")
    
    # Create comparison plots (4 rows: data, PSD, data residual, PSD residual)
    fig, axes = plt.subplots(4, len(bilby_ifos), figsize=(5*len(bilby_ifos), 16))
    if len(bilby_ifos) == 1:
        axes = axes.reshape(4, 1)
    
    for i, (ifo_b, ifo_j) in enumerate(zip(bilby_ifos, jim_ifos)):
        # Trigger FFT calculation for Jim data
        ifo_j.data.fft()
        
        # Interpolate Jim data to bilby frequency grid for residual calculation
        jim_data_interp = jnp.interp(ifo_b.frequency_array, ifo_j.data.frequencies, ifo_j.data.fd)
        jim_psd_interp = jnp.interp(ifo_b.frequency_array, ifo_j.psd.frequencies, ifo_j.psd.values)
        
        # Plot data comparison
        axes[0, i].loglog(ifo_b.frequency_array, jnp.abs(ifo_b.frequency_domain_strain), 
                         label='Bilby data', alpha=0.7)
        axes[0, i].loglog(ifo_j.data.frequencies, jnp.abs(ifo_j.data.fd), 
                         label=f'Jim data ({data_source})', alpha=0.7)
        axes[0, i].set_xlabel('Frequency [Hz]')
        axes[0, i].set_ylabel('|Strain| [strain/√Hz]')
        axes[0, i].set_title(f'{ifo_b.name} - Data Comparison')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xlim(fmin, fmax)

        # Plot data residual (real and imaginary parts)
        data_residual = ifo_b.frequency_domain_strain - jim_data_interp
        axes[1, i].semilogx(ifo_b.frequency_array, jnp.real(data_residual), 
                           label='Real part', alpha=0.7)
        axes[1, i].semilogx(ifo_b.frequency_array, jnp.imag(data_residual), 
                           label='Imaginary part', alpha=0.7)
        axes[1, i].set_ylim(-1e-25, 1e-25)
        axes[1, i].set_xlabel('Frequency [Hz]')
        axes[1, i].set_ylabel('Strain Residual [strain/√Hz]')
        axes[1, i].set_title(f'{ifo_b.name} - Data Residual (Bilby - Jim/{data_source})')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xlim(fmin, fmax)
        
        # Plot PSD comparison
        axes[2, i].loglog(ifo_b.frequency_array, ifo_b.power_spectral_density_array, 
                         label='Bilby PSD', alpha=0.7)
        axes[2, i].loglog(ifo_j.psd.frequencies, ifo_j.psd.values, 
                         label=f'Jim PSD ({psd_source})', alpha=0.7)
        axes[2, i].set_xlabel('Frequency [Hz]')
        axes[2, i].set_ylabel('PSD [strain²/Hz]')
        axes[2, i].set_title(f'{ifo_b.name} - PSD Comparison')
        axes[2, i].legend()
        axes[2, i].grid(True, alpha=0.3)
        axes[2, i].set_xlim(fmin, fmax)
        
        # Plot PSD residual (absolute fractional difference)
        psd_residual_frac = (ifo_b.power_spectral_density_array - jim_psd_interp) / ifo_b.power_spectral_density_array
        psd_residual_abs_frac = jnp.abs(psd_residual_frac)
        axes[3, i].semilogx(ifo_b.frequency_array, psd_residual_abs_frac, 
                           label='Absolute fractional difference', alpha=0.7, color='red')    
        axes[3, i].set_xlabel('Frequency [Hz]')
        axes[3, i].set_ylabel('|PSD Fractional Residual|')
        axes[3, i].set_title(f'{ifo_b.name} - PSD Residual |Bilby - Jim/{psd_source}|/Bilby')
        axes[3, i].legend()
        axes[3, i].grid(True, alpha=0.3)
        axes[3, i].set_xlim(fmin, fmax)
    
    plt.tight_layout()
    plt.savefig(f"{event_outdir}/data_psd_comparison.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print("Data/PSD comparison plot saved.")
    
    # -------------------------------
    # Set up waveform
    # -------------------------------
    ref_freq = float(bilby_data_dump.meta_data["command_line_args"]["reference_frequency"])

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
    s1_prior = UniformSpherePrior(parameter_names=["s1"], max_mag=0.99)
    s2_prior = UniformSpherePrior(parameter_names=["s2"], max_mag=0.99)
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
    # dL_prior = PowerLawPrior(1.0, dL_upper, 2.0, parameter_names=["d_L"])
    t_c_prior = SimpleConstrainedPrior([UniformPrior(-0.1, 0.1, parameter_names=["t_c"])])
    # t_c_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
    phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
    psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
    ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
    dec_prior = CosinePrior(parameter_names=["dec"])
    prior += [dL_prior,
              t_c_prior,
              phase_c_prior, psi_prior, ra_prior, dec_prior]

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
        # BoundToUnbound(name_mapping=(["d_L"], ["d_L_unbounded"]), original_lower_bound=0.0, original_upper_bound=dL_upper),

        # Transformations for phase
        GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=jim_ifos[0]),
        PeriodicTransform(name_mapping=(["periodic_4", "phase_det"], ["phase_det_x", "phase_det_y"]), xmin=0.0, xmax=2 * jnp.pi),
        # PeriodicTransform(name_mapping=(["periodic_4", "phase_c"], ["phase_c_x", "phase_c_y"]), xmin=0.0, xmax=2 * jnp.pi),

        # Transformations for time
        GeocentricArrivalTimeToDetectorArrivalTimeTransform(gps_time=gps, ifo=jim_ifos[0]),
        # BoundToUnbound(name_mapping=(["t_c"], ["t_c_unbounded"]), original_lower_bound=-0.1, original_upper_bound=0.1),

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
    # Setup Likelihood based on relative binning mode
    # -------------------------------
    if args.relative_binning == 0:
        print("Using normal likelihood")
        likelihood = BaseTransientLikelihoodFD(
            detectors=jim_ifos,
            waveform=waveform,
            trigger_time=gps,
            f_min=fmin,
            f_max=fmax,
        )
    elif args.relative_binning == 1:
        print("Using heterodyned likelihood with fixed reference parameters (bilby maxL sample)")
        # Load bilby maxL sample as ref_params
        bilby_result_path = f"../bilby_runs/outdir/{args.event_id}/final_result"
        bilby_result_file = os.listdir(bilby_result_path)[0]
        bilby_result = CBCResult.from_hdf5(os.path.join(bilby_result_path, bilby_result_file)).posterior
        max_L_sample = bilby_result.loc[bilby_result["log_likelihood"].idxmax()]
        ref_params = {
            "M_c": max_L_sample["chirp_mass"],
            "eta": max_L_sample["symmetric_mass_ratio"],
            "s1_x": max_L_sample["spin_1x"],
            "s1_y": max_L_sample["spin_1y"],
            "s1_z": max_L_sample["spin_1z"],
            "s2_x": max_L_sample["spin_2x"],
            "s2_y": max_L_sample["spin_2y"],
            "s2_z": max_L_sample["spin_2z"],
            "iota": max_L_sample["iota"],
            "d_L": max_L_sample["luminosity_distance"],
            "t_c": max_L_sample["geocent_time"] - gps,
            "phase_c": max_L_sample["phase"],
            "psi": max_L_sample["psi"],
            "ra": max_L_sample["ra"],
            "dec": max_L_sample["dec"],
        }
        likelihood = HeterodynedTransientLikelihoodFD(
            jim_ifos,
            waveform=waveform,
            n_bins=1_000,
            trigger_time=gps,
            f_min=fmin,
            f_max=fmax,
            ref_params=ref_params,
        )
    elif args.relative_binning == 2:
        print("Using heterodyned likelihood with optimised reference parameters")
        likelihood = HeterodynedTransientLikelihoodFD(
            detectors=jim_ifos,
            waveform=waveform,
            n_bins=1_000,
            trigger_time=gps,
            f_min=fmin,
            f_max=fmax,
            popsize=1000,
            n_steps=10000,
            prior=prior,
            sample_transforms=sample_transforms,
            likelihood_transforms=likelihood_transforms,
        )

    # -------------------------------
    # Configure the sampler and training parameters
    # -------------------------------
    parameter_names = prior.parameter_names
    for sample_transform in sample_transforms:
        sample_transform.propagate_name(parameter_names)

    mass_matrix = jnp.eye(prior.n_dims)
    # q_index = parameter_names.index("q")
    # mass_matrix = mass_matrix.at[q_index, q_index].set(1e-3)
    dL_index = parameter_names.index("d_L")
    mass_matrix = mass_matrix.at[dL_index, dL_index].set(1e4)
    tc_index = parameter_names.index("t_c")
    mass_matrix = mass_matrix.at[tc_index, tc_index].set(1e-1)
    mass_matrix *= 2e-3
    print("Initial mass matrix (diagonal):")
    print({k: v for k, v in zip(parameter_names, jnp.diag(mass_matrix))})

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
        mala_step_size=mass_matrix,
        rq_spline_hidden_units=[128, 128],
        rq_spline_n_bins=10,
        rq_spline_n_layers=8,
        learning_rate=1e-3,
        batch_size=10000,
        n_max_examples=10000,
        n_NFproposal_batch_size=5,
        local_thinning=1,
        global_thinning=10,
        n_temperatures=0,
        verbose=True,
    )

    # Run the sampler
    jim.sample(jim.sample_initial_condition())

    total_time_end = time.time()
    print(f"Time taken: {total_time_end - total_time_start} seconds = {(total_time_end - total_time_start) / 60} minutes")

    # -------------------------------
    # Postprocessing
    # -------------------------------
    samples = jim.get_samples()
    downsample_indices = jax.random.choice(
        jax.random.PRNGKey(123),
        jnp.arange(samples[list(samples.keys())[0]].shape[0]),
        shape=(5000,),
        replace=False,
    )
    samples = {key: samples[key][downsample_indices] for key in samples.keys()}
    transformed_samples = samples.copy()
    for transform in likelihood_transforms:
        transformed_samples = jax.vmap(transform.forward)(transformed_samples)
    log_likelihood = jax.vmap(jim.likelihood.evaluate)(transformed_samples, {})
    samples["log_L"] = log_likelihood
    jnp.savez(f"{event_outdir}/samples.npz", **samples)

    resources = jim.sampler.resources
    # logprob_train = resources["log_prob_training"].data[::10, ::10]
    logprob_prod = resources["log_prob_production"].data[::10, ::10]
    # local_acceptance_train = resources["local_accs_training"].data[::10, ::10]
    local_acceptance_prod = resources["local_accs_production"].data[::10, ::10]
    # global_acceptance_train = resources["global_accs_training"].data[::10, ::10]
    global_acceptance_prod = resources["global_accs_production"].data[::10, ::10]
    chains = resources["positions_production"].data[::10, ::10, :]

    jnp.savez(f"{event_outdir}/results.npz",
            #   log_prob_training=logprob_train,
              log_prob_production=logprob_prod,
            #   local_accs_training=local_acceptance_train,
              local_accs_production=local_acceptance_prod,
            #   global_accs_training=global_acceptance_train,
              global_accs_production=global_acceptance_prod,
              chains=chains,
              )
    
    return jim


def main():
    """Entry point: parse arguments and run the parameter estimation."""
    return run_pe(utils.get_parser().parse_args())


if __name__ == "__main__":
    jim = main()
