import os
import time
import utils
import pickle
import ast
import numpy as np

print("Importing JAX")
import jax
import jax.numpy as jnp
print("Importing JAX successful")

print(f"Checking for CUDA: JAX devices {jax.devices()}")

from jimgw.jim import Jim
from jimgw.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    UniformSpherePrior,
    RayleighPrior,
)
from jimgw.transforms import PeriodicTransform
from jimgw.single_event.detector import H1, L1, V1, GroundBased2G
from jimgw.single_event.likelihood import TransientLikelihoodFD, HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2, RippleIMRPhenomD
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    SphereSpinToCartesianSpinTransform,
    MassRatioToSymmetricMassRatioTransform,
    DistanceToSNRWeightedDistanceTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
    GeocentricArrivalPhaseToDetectorArrivalPhaseTransform,
)

jax.config.update("jax_enable_x64", True)

import argparse

def run_pe(args: argparse.Namespace,
           verbose: bool = False):
    
    total_time_start = time.time()
    print("args:")
    print(args)
    
    # Make the outdir
    if not os.path.exists(os.path.join(args.outdir, args.event_id)):
        os.makedirs(os.path.join(args.outdir, args.event_id))
        
    # Fetch bilby_pipe DataDump
    print(f"Fetching bilby_pipe DataDump for event {args.event_id}")
    file = os.listdir(f"../bilby_runs/outdir/{args.event_id}/data")[0]
    with open(f"../bilby_runs/outdir/{args.event_id}/data/{file}", "rb") as f:
        data_dump = pickle.load(f)
    Mc_lower = float(data_dump.priors_dict['chirp_mass'].minimum)
    Mc_upper = float(data_dump.priors_dict['chirp_mass'].maximum)
    
    print(f"Setting the Mc bounds to be {Mc_lower} and {Mc_upper}")
    
    dL_upper = float(data_dump.priors_dict['luminosity_distance'].maximum)
    print(f"The dL upper bound is {dL_upper}")
    
    if verbose:
        print("metadata:")
        print(data_dump.meta_data)
    
    duration = float(data_dump.interferometers[0].strain_data.time_array[-1] - data_dump.interferometers[0].strain_data.time_array[0])
    post_trigger = float(data_dump.meta_data['command_line_args']['post_trigger_duration'])
    gps = float(data_dump.trigger_time)
    fmin: dict[str, float] = ast.literal_eval(data_dump.meta_data['command_line_args']['minimum_frequency'])
    fmax: dict[str, float] = ast.literal_eval(data_dump.meta_data['command_line_args']['maximum_frequency'])
    
    ifos_list_string = list(data_dump.interferometers.meta_data.keys())
    
    ifos_diff_fmin = []
    try:
        if all(fmin[ifos_list_string[0]] == fmin[ifos_list_string[i]] for i in range(1, len(ifos_list_string))):
            print("fmin is the same for all ifos")
            fmin = fmin[ifos_list_string[0]]
        else:
            print("fmin is different for different ifos")
            ifos_diff_fmin = [ifo for ifo in ifos_list_string if fmin[ifo] != min(fmin.values())]
            fmin = min(fmin.values())
    except:
        fmin = float(fmin)

    ifos_diff_fmax = []    
    try:
        if all(fmax[ifos_list_string[0]] == fmax[ifos_list_string[i]] for i in range(1, len(ifos_list_string))):
            print("fmax is the same for all ifos")
            fmax = fmax[ifos_list_string[0]]
        else:
            print("fmax is different for different ifos")
            ifos_diff_fmax = [ifo for ifo in ifos_list_string if fmax[ifo] != max(fmax.values())]
            fmax = max(fmax.values())
    except:
        fmax = float(fmax)

    
    if verbose:
        print("fmin:")
        print(fmin)
        
        print("fmax:")
        print(fmax)
    
    # Load the HDF5 files from the ifos dict url and open it:
    ifos: list[GroundBased2G] = []
    for i, ifo_string in enumerate(ifos_list_string):

        ifo_bilby = data_dump.interferometers[i]
        assert ifo_bilby.name == ifo_string, f"ifo_bilby.name: {ifo_bilby.name} != ifo_string: {ifo_string}"

        print("Adding interferometer ", ifo_string)
        eval(f'ifos.append({ifo_string})')
    
        frequencies, data, psd = ifo_bilby.frequency_array, ifo_bilby.frequency_domain_strain, ifo_bilby.power_spectral_density_array
        
        mask = (frequencies >= fmin) & (frequencies <= fmax)
        frequencies = frequencies[mask]
        data = data[mask]
        psd = psd[mask]
    
        ifos[i].frequencies = frequencies
        ifos[i].data = data
        ifos[i].psd = psd

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
            
            print(f"Data: {ifos[i].data}")
            print(f"PSD: {ifos[i].psd}")
            print(f"Frequencies: {ifos[i].frequencies}")
    
    if verbose:
        print(f"Running PE on event {args.event_id}")
        print(f"Duration: {duration}")
        print(f"GPS: {gps}")
        print(f"Chirp mass: [{Mc_lower}, {Mc_upper}]")
    
    if args.event_id[-2:] == "_D":
        print("Using IMRPhenomD waveform")
        waveform = RippleIMRPhenomD(f_ref=float(data_dump.meta_data['command_line_args']['reference_frequency']))
    else:
        print("Using IMRPhenomPv2 waveform")
        waveform = RippleIMRPhenomPv2(f_ref=float(data_dump.meta_data['command_line_args']['reference_frequency']))

    ###########################################
    ########## Set up priors ##################
    ###########################################

    prior = []

    # Mass prior
    Mc_prior = UniformPrior(Mc_lower, Mc_upper, parameter_names=["M_c"])
    q_min, q_max = 0.125, 1.0
    q_prior = UniformPrior(q_min, q_max, parameter_names=["q"])

    prior = prior + [Mc_prior, q_prior]

    # Spin prior
    if args.event_id[-2:] == "_D":
        s1_prior = UniformPrior(-0.99, 0.99, parameter_names=["s1_z"])
        s2_prior = UniformPrior(-0.99, 0.99, parameter_names=["s2_z"])
    else:
        s1_prior = UniformSpherePrior(parameter_names=["s1"])
        s2_prior = UniformSpherePrior(parameter_names=["s2"])
    iota_prior = SinePrior(parameter_names=["iota"])

    prior = prior + [
        s1_prior,
        s2_prior,
        iota_prior,
    ]

    # Extrinsic prior
    dL_prior = PowerLawPrior(1.0, dL_upper, 2.0, parameter_names=["d_L"])
    t_c_prior = UniformPrior(-0.1, 0.1, parameter_names=["t_c"])
    phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
    psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
    ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
    dec_prior = CosinePrior(parameter_names=["dec"])

    prior = prior + [
        dL_prior,
        t_c_prior,
        phase_c_prior,
        psi_prior,
        ra_prior,
        dec_prior,
    ]

    # Extra prior for periodic parameters
    r_1_prior = RayleighPrior(1.0, parameter_names=["periodic_1"])
    r_2_prior = RayleighPrior(1.0, parameter_names=["periodic_2"])
    r_3_prior = RayleighPrior(1.0, parameter_names=["periodic_3"])

    prior = prior + [
        r_1_prior,
        r_2_prior,
        r_3_prior,
    ]

    if args.event_id[-2:] != "_D":
        prior = prior + [
            RayleighPrior(1.0, parameter_names=["periodic_4"]),
            RayleighPrior(1.0, parameter_names=["periodic_5"]),
        ]

    prior = CombinePrior(prior)

    # Defining Transforms

    sample_transforms = [
        DistanceToSNRWeightedDistanceTransform(gps_time=gps, ifos=ifos, dL_min=dL_prior.xmin, dL_max=dL_prior.xmax),
        GeocentricArrivalPhaseToDetectorArrivalPhaseTransform(gps_time=gps, ifo=ifos[0]),
        GeocentricArrivalTimeToDetectorArrivalTimeTransform(tc_min=t_c_prior.xmin, tc_max=t_c_prior.xmax, gps_time=gps, ifo=ifos[0]),
        SkyFrameToDetectorFrameSkyPositionTransform(gps_time=gps, ifos=ifos),
        BoundToUnbound(name_mapping = (["M_c"], ["M_c_unbounded"]), original_lower_bound=Mc_lower, original_upper_bound=Mc_upper),
        BoundToUnbound(name_mapping = (["q"], ["q_unbounded"]), original_lower_bound=q_min, original_upper_bound=q_max),
        BoundToUnbound(name_mapping = (["iota"], ["iota_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
        PeriodicTransform(name_mapping = (["periodic_1", "psi"], ["psi_base_x", "psi_base_y"]), xmin=0.0, xmax=jnp.pi),
        # BoundToUnbound(name_mapping = (["psi"], ["psi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
        PeriodicTransform(name_mapping = (["periodic_2", "phase_det"], ["phase_det_x", "phase_det_y"]), xmin=0.0, xmax=2 * jnp.pi),
        # BoundToUnbound(name_mapping = (["phase_det"], ["phase_det_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
        BoundToUnbound(name_mapping = (["zenith"], ["zenith_unbounded"]), original_lower_bound=0.0, original_upper_bound=jnp.pi),
        PeriodicTransform(name_mapping = (["periodic_3", "azimuth"], ["azimuth_x", "azimuth_y"]), xmin=0.0, xmax=2 * jnp.pi),
        # BoundToUnbound(name_mapping = (["azimuth"], ["azimuth_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    ]
    if args.event_id[-2:] == "_D":
        sample_transforms = sample_transforms + [
            BoundToUnbound(name_mapping = (["s1_z"], ["s1_z_unbounded"]), original_lower_bound=-0.99, original_upper_bound=0.99),
            BoundToUnbound(name_mapping = (["s2_z"], ["s2_z_unbounded"]), original_lower_bound=-0.99, original_upper_bound=0.99),
        ]
    else:
        sample_transforms = sample_transforms + [
            PeriodicTransform(name_mapping = (["periodic_4", "s1_phi"], ["s1_phi_base_x", "s1_phi_base_y"]), xmin=0.0, xmax=2 * jnp.pi),
            PeriodicTransform(name_mapping = (["periodic_5", "s2_phi"], ["s2_phi_base_x", "s2_phi_base_y"]), xmin=0.0, xmax=2 * jnp.pi),
            # BoundToUnbound(name_mapping = (["s1_phi"], ["s1_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
            # BoundToUnbound(name_mapping = (["s2_phi"], ["s2_phi_unbounded"]) , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
            BoundToUnbound(name_mapping = (["s1_theta"], ["s1_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
            BoundToUnbound(name_mapping = (["s2_theta"], ["s2_theta_unbounded"]) , original_lower_bound=0.0, original_upper_bound=jnp.pi),
            BoundToUnbound(name_mapping = (["s1_mag"], ["s1_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
            BoundToUnbound(name_mapping = (["s2_mag"], ["s2_mag_unbounded"]) , original_lower_bound=0.0, original_upper_bound=0.99),
        ]

    likelihood_transforms = [
        MassRatioToSymmetricMassRatioTransform,
    ]
    if args.event_id[-2:] != "_D":
        likelihood_transforms = likelihood_transforms + [
            SphereSpinToCartesianSpinTransform("s1"),
            SphereSpinToCartesianSpinTransform("s2"),
        ]

    # likelihood = TransientLikelihoodFD(
    #     ifos, waveform=waveform, trigger_time=gps, duration=duration, post_trigger_duration=post_trigger
    # )
    
    likelihood = HeterodynedTransientLikelihoodFD(ifos, 
                                                  waveform=waveform, 
                                                  n_bins = 1_000, 
                                                  trigger_time=gps, 
                                                  duration=duration, 
                                                  post_trigger_duration=post_trigger, 
                                                  prior=prior, 
                                                  sample_transforms=sample_transforms,
                                                  likelihood_transforms=likelihood_transforms,
                                                  popsize=10,
                                                  n_steps=50)


    mass_matrix = jnp.eye(prior.n_dim)
    local_sampler_arg = {"step_size": mass_matrix * 3e-3}

    # Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

    # import optax

    n_loop_training = 100
    n_epochs = 10
    # total_epochs = n_epochs * n_loop_training
    # start = total_epochs // 10
    # learning_rate = optax.polynomial_schedule(
    #     1e-3, 1e-4, 4.0, total_epochs - start, transition_begin=start
    # )
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
        # strategies=[Adam_optimizer,"default"],
    )

    rng_key = jax.random.PRNGKey(123)
    jim.sample(rng_key)
    jim.print_summary()

    total_time_end = time.time()
    print(f"Time taken: {total_time_end - total_time_start} seconds = {(total_time_end - total_time_start) / 60} minutes")
    
    # Postprocessing comes here
    samples = jim.get_samples()
    downsample_indices = np.random.choice(samples[list(samples.keys())[0]].shape[0], 5000)
    samples = {key: samples[key][downsample_indices] for key in samples.keys()}
    jnp.savez(os.path.join(args.outdir, args.event_id, "samples.npz"), **samples)
    log_prob = jim.sampler.get_sampler_state(training=False)['log_prob'].reshape(-1)[downsample_indices]
    log_prior = jax.vmap(prior.log_prob)(samples)[downsample_indices]
    jnp.savez(os.path.join(args.outdir, args.event_id, "result.npz"), log_prior=log_prior, log_prob=log_prob)
    rng_key, subkey = jax.random.split(rng_key)
    nf_samples = np.array(jim.sampler.sample_flow(subkey, 5000))
    nf_samples = jax.vmap(jim.add_name)(nf_samples)
    for transform in reversed(sample_transforms):
        nf_samples = jax.vmap(transform.backward)(nf_samples)
    jnp.savez(os.path.join(args.outdir, args.event_id, "nf_samples.npz"), **nf_samples)

    # Plotting
    import matplotlib.pyplot as plt
    
    out_train = jim.sampler.get_sampler_state(training=True)

    chains = np.array(out_train["chains"])
    global_accs = np.array(out_train["global_accs"])
    local_accs = np.array(out_train["local_accs"])
    loss_vals = np.array(out_train["loss_vals"])
    
    # Plot 2 chains in the plane of 2 coordinates for first visual check
    axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
    plt.sca(axs[0])
    plt.title("2d proj of 2 chains")

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
    plt.savefig(f"{args.outdir}/{args.event_id}/training.jpg")

    # Plot log_prob vs chain index
    log_prob = jim.sampler.get_sampler_state(training=False)['log_prob'] # (n_chains, n_samples)
    plt.figure()
    plt.plot(log_prob.mean(0))
    mean = log_prob.mean(0)
    std = log_prob.std(0)
    plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.3)
    plt.xlabel("chain index")
    plt.ylabel("log_prob")
    plt.title("Production log_prob")
    plt.savefig(f"{args.outdir}/{args.event_id}/log_prob.jpg")


def main():
    parser = utils.get_parser()
    args = parser.parse_args()
    run_pe(args)
    
if __name__ == "__main__":
    main()
