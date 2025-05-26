import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
print(os.getcwd())
import time
import gc
import torch
import bisect

#import empirical cdf
# import sys
# sys.path.append(os.path.abspath('../../src'))

from data_processing import gen_ckpt_steps, move_dict_to_device, get_other_err, get_mop_ratios_ckpt, compute_ratio
# sys.path.append(os.path.abspath('..'))

from check_ecdf import get_empirical_cdf

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# def get_seg_starts_per_config(experiment, valA, valC, state_dim, ckpt, print_seg_starts=False, nope=False, needle=False, fin_seg_ext=False, haystack_len=19, train_conv=False, datasource="val", paren_swap=False, fix_needle=False, opposite_ortho=False, irrelevant_tokens=False, same_tokens=False, new_hay_insert=False):
#     # load the sys choices etc
#     errs_dir = "../outputs/GPT2" + ("_NoPE" if nope else "") + "/" + experiment + f"/prediction_errors{valC}_step={ckpt}.ckpt"
#     # + ("train_conv_" if needle else "")
#     errs_loc = errs_dir + f"/" + ("train_conv_" if train_conv else "") + ("single_system_" if not needle else "") + (f"needle_haystack_len_{haystack_len}_{datasource}_" if needle else "") + ("fin_seg_ext_" if needle and fin_seg_ext else "") + f"{valA}_state_dim_{state_dim}_" + ("fix_needle_" if fix_needle else "") + ("opposite_ortho_" if opposite_ortho else "") + ("new_hay_insert_" if new_hay_insert else "") + ("irrelevant_tokens_" if irrelevant_tokens else "") + ("same_tokens_" if same_tokens else "") + ("paren_swap_" if paren_swap else "")  + f"sys_choices_sys_dict_tok_seg_lens_seg_starts" + ("_example_0" if needle else "") + ".pkl"

#     if not os.path.exists(errs_loc):
#         print(f"errs_loc {errs_loc} does not exist")
#         return None
#     else:
#         with open(errs_loc, "rb") as f:
#             data = pickle.load(f)
#             seg_starts_per_config = data['seg_starts_per_config']
#             if print_seg_starts:
#                 print(f"seg_starts_per_config: {seg_starts_per_config}")
                
#         return seg_starts_per_config

def get_seg_starts_per_config(config, ex=0):
    if config.datasource == "train_systems" or config.datasource == "backstory_train" or config.datasource == "train":
        dataset_typ = config.dataset_typ
    elif config.datasource == "val":
        dataset_typ = config.val_dataset_typ

    # path to interleaved data file
    if config.needle_in_haystack:
        interleaving = f"haystack_len_{config.num_sys_haystack}"
    else:
        interleaving = f"multi_cut"

    #load multi_sys_ys from interleaved_traces file
    interleave_traces_dict_path = os.path.join(f"/data/shared/ICL_Kalman_Experiments/train_and_test_data/{dataset_typ}/" + ("backstory_" if config.mem_suppress and config.backstory else "") + ("masked_" if config.mem_suppress and config.masking else "") + ("unmasked_" if config.mem_suppress and not config.masking else "") + f"moss_{config.datasource}_" + ("ortho_sync_" if config.val_dataset_typ == "ortho_sync" else "") + ("fix_needle_" if config.fix_needle else "") + ("opposite_ortho_" if config.opposite_ortho else "") + ("irrelevant_tokens_" if config.irrelevant_tokens else "") + ("same_tokens_" if config.same_tokens else "") + ("new_hay_insert_" if config.new_hay_insert else "")+ ("paren_swap_" if config.paren_swap else "") + ("zero_cut_" if config.zero_cut else "") + f"interleaved_traces_{dataset_typ}{config.C_dist}_{interleaving}.pkl")

    if os.path.exists(interleave_traces_dict_path):
        with open(interleave_traces_dict_path, "rb") as f:
            interleave_traces_dict = pickle.load(f)
            seg_starts_per_config_per_example = interleave_traces_dict["seg_starts_per_config"]

        if config.needle_in_haystack:
            seg_starts_per_config = seg_starts_per_config_per_example[0] # all of the examples have the same segment starts for needle in haystack tests

        else:
            seg_starts_per_config = seg_starts_per_config_per_example[ex] #get the segment starts for the example ex

    return seg_starts_per_config

def train_conv_plots(experiments, trainAs, kal_ckpt, valA, C_dist, num_val_systems, compute_more_ckpts=False, ind=250, min_ckpt=79, max_ckpt=79000, interval=79, nx=10, needle_in_haystack=False, single_system=False, max_ir_len=3, nope=False, batch_size=512, gpus=1, zero_cut=False):
    num_preds = 3+(3*2) #len(experiments) #number of predictors to plot

    colors = ['#000000', '#005CAB', '#E31B23', '#FFC325', '#00A651', '#9B59B6']


    plot_time = time.ctime()

    #create a figure with subplots for each of the m indexes for the cdfs
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    filename = f'training_dist_comparison_val_{valA}_state_dim_{nx}_val_sys_{num_val_systems}_{time.time()}.pdf'

    parent_path = "../outputs/GPT2" + ("_NoPE" if nope else "") + "/"

    filepath = os.path.abspath(f"../outputs/train_conv/{filename}")
    print(filepath)

    print(f"quantiles 5 path exists?: {os.path.exists(parent_path + experiments[0] + '/train_conv/quantiles_5.npz')}")

    ckpt_steps = gen_ckpt_steps(min_ckpt, max_ckpt, interval)

    i = 0
    for experiment in experiments:
        if (not (os.path.exists(parent_path + experiment + "/train_conv/quantiles.npz")) or (single_system and not os.path.exists(parent_path + experiment + "/train_conv/quantiles_5.npz"))) or compute_more_ckpts:
            kal_err = None #initialize kalman error
            pred_ckpts = []
            quantiles = []
            if single_system:
                quantiles_5 = []
                quantiles_20 = []
            print("\n\ni", i)
            if not needle_in_haystack and not (valA == "ortho" or valA == "ident"): 
                    kal_err = get_other_err(valA, C_dist, kal_ckpt[i], experiment, "Kalman_rem", nx=nx, single_system=single_system, nope=nope)


                    if single_system:
                        seg_starts_per_config = get_seg_starts_per_config(experiment, valA, C_dist, nx, kal_ckpt[i], print_seg_starts=True, nope=nope)
                        seg_starts = seg_starts_per_config[0]

                        ols_quantile = {}
                        ols_quantile_5 = {}
                        ols_quantile_20 = {}
                        for ir in range(1, max_ir_len+1):

                            ols_errs = get_other_err(valA, C_dist, kal_ckpt[i], experiment, f"OLS_ir_{ir}", nx=nx, single_system=single_system, nope=nope)
                            ols_err_rat = compute_ratio(ind=ind, err=ols_errs, kalman_err=kal_err, single_system=single_system)

                            if len(seg_starts) > 1:
                                ols_quantile[ir] = ols_err_rat[:, seg_starts[1] + 1] #take the quantile at the start of the second segment
                                ols_quantile_5[ir] = ols_err_rat[:, seg_starts[1] + 5] #take the quantile 5 steps after the start of the second segment
                                ols_quantile_20[ir] = ols_err_rat[:, seg_starts[1] + 20] #take the quantile 20 steps after the start of the second segment

                            if isinstance(ols_quantile[ir], torch.Tensor):
                                ols_quantile[ir] = ols_quantile[ir].cpu().numpy()
                            if isinstance(ols_quantile_5[ir], torch.Tensor):
                                ols_quantile_5[ir] = ols_quantile_5[ir].cpu().numpy()
                            if isinstance(ols_quantile_20[ir], torch.Tensor):
                                ols_quantile_20[ir] = ols_quantile_20[ir].cpu().numpy()


                    
            for ckpt_step in ckpt_steps:

                mop_err, pred_ckpt = get_mop_ratios_ckpt(valA, C_dist, ckpt_step, experiment, nx=nx, single_system=single_system, nope=nope)
                if pred_ckpt:

                    if needle_in_haystack and not (valA == "ortho" or valA == "ident"):
                        kal_err = get_other_err(valA, C_dist, ckpt_step, experiment, "Kalman", nx=nx, single_system=single_system, nope=nope)

                    quantile = compute_ratio(ind=ind, err=mop_err, kalman_err=kal_err, single_system=single_system)
                    if single_system:

                        print(f"quantile shape before seg start choice: {quantile.shape}")
                        seg_starts_per_config = get_seg_starts_per_config(experiment, valA, C_dist, nx, ckpt_step, print_seg_starts=True, nope=nope)

                        print(f"seg_starts_per_config: {seg_starts_per_config}")
                        seg_starts = seg_starts_per_config[0]

                        if len(seg_starts) > 1:
                            quantile_5 = quantile[:, seg_starts[1] + 5] #take the quantile 5 steps after the start of the second segment
                            quantile_20 = quantile[:, seg_starts[1] + 20] #take the quantile 20 steps after the start of the second segment
                            quantile = quantile[:, seg_starts[1] + 1] #take the quantile at the start of the second segment
                            print(f"seg_starts[1] + 1: {seg_starts[1] + 1}")
                            print(f"quantile shape after seg start choice: {quantile.shape}")
                        else:
                            print("only one segment start so disregard ckpt")
                            continue
                        
                    
                    pred_ckpts.append(pred_ckpt)
                    print(f"quantile shape: {quantile.shape}")
                    if isinstance(quantile, torch.Tensor):
                        quantile = quantile.cpu().numpy()
                    del mop_err
                    quantiles.append(quantile)

                    if single_system:
                        if isinstance(quantile_5, torch.Tensor):
                            quantile_5 = quantile_5.cpu().numpy()
                        quantiles_5.append(quantile_5)
                        if isinstance(quantile_20, torch.Tensor):
                            quantile_20 = quantile_20.cpu().numpy()
                        quantiles_20.append(quantile_20)


                    torch.cuda.empty_cache()
                    gc.collect()
            del kal_err
            torch.cuda.empty_cache()
            gc.collect()

            quantiles = np.array(quantiles)
            if single_system:
                quantiles_5 = np.array(quantiles_5)
                quantiles_20 = np.array(quantiles_20)
            
            #save quantiles to file
            os.makedirs(parent_path + experiment + "/train_conv", exist_ok=True)
            np.savez_compressed(parent_path + experiment + "/train_conv/" + ("zero_cut_" if zero_cut else "") + "quantiles.npz", pred_ckpts=pred_ckpts, quantiles=quantiles)
            if single_system:
                np.savez_compressed(parent_path + experiment + "/train_conv/" + ("zero_cut_" if zero_cut else "") + "quantiles_5.npz", pred_ckpts=pred_ckpts, quantiles=quantiles_5)

                np.savez_compressed(parent_path + experiment + "/train_conv/" + ("zero_cut_" if zero_cut else "") + "quantiles_20.npz", pred_ckpts=pred_ckpts, quantiles=quantiles_20)

                if not (valA == "ortho" or valA == "ident"):
                    # Convert keys to strings
                    ols_quantile_str_keys = {str(k): v for k, v in ols_quantile.items()}
                    ols_quantile_5_str_keys = {str(k): v for k, v in ols_quantile_5.items()}
                    ols_quantile_20_str_keys = {str(k): v for k, v in ols_quantile_20.items()}

                    np.savez_compressed(parent_path + experiment + "/train_conv/quantiles_ols.npz", **ols_quantile_str_keys)
                    np.savez_compressed(parent_path + experiment + "/train_conv/quantiles_ols_5.npz", **ols_quantile_5_str_keys)
                    np.savez_compressed(parent_path + experiment + "/train_conv/quantiles_ols_20.npz", **ols_quantile_20_str_keys)

                    # np.savez_compressed(parent_path + experiment + "/train_conv/quantiles_ols.npz", pred_ckpts=pred_ckpts, quantiles_ols=ols_quantile, quantiles_ols_5=ols_quantile_5, quantiles_ols_20=ols_quantile_20)
        else:
            print(f"quantiles already exist for {experiment}, and single_system={single_system}")

            data = np.load(parent_path + experiment + "/train_conv/" + ("zero_cut_" if zero_cut else "") + "quantiles.npz", allow_pickle=True)
            print(f"keys in the file: {data.files}")
            pred_ckpts = data["pred_ckpts"]
            quantiles = data["quantiles"]
            print(f"quantiles shape after load: {quantiles.shape}")

            if single_system:
                print(f"loading quantiles_5 and quantiles_20")
                data = np.load(parent_path + experiment + "/train_conv/" + ("zero_cut_" if zero_cut else "") + "quantiles_5.npz", allow_pickle=True)
                quantiles_5 = data["quantiles"]

                data = np.load(parent_path + experiment + "/train_conv/" + ("zero_cut_" if zero_cut else "") + "quantiles_20.npz", allow_pickle=True)
                quantiles_20 = data["quantiles"]

                if not (valA == "ortho" or valA == "ident"):
                    ols_quantile = np.load(parent_path + experiment + "/train_conv/" + ("zero_cut_" if zero_cut else "") + "quantiles_ols.npz", allow_pickle=True)
                    ols_quantile_5 = np.load(parent_path + experiment + "/train_conv/" + ("zero_cut_" if zero_cut else "") + "quantiles_ols_5.npz", allow_pickle=True)
                    ols_quantile_20 = np.load(parent_path + experiment + "/train_conv/" + ("zero_cut_" if zero_cut else "") + "quantiles_ols_20.npz", allow_pickle=True)

                    #convert keys back to ints
                    ols_quantile = {int(k): v for k, v in ols_quantile.items()}
                    ols_quantile_5 = {int(k): v for k, v in ols_quantile_5.items()}
                    ols_quantile_20 = {int(k): v for k, v in ols_quantile_20.items()}

                    # data = np.load(parent_path + experiment + "/train_conv/quantiles_ols.npz", allow_pickle=True)
                    # ols_quantile = data["quantiles_ols"]
                    # ols_quantile_5 = data["quantiles_ols_5"]
                    # ols_quantile_20 = data["quantiles_ols_20"]


        if not (valA == "ortho" or valA == "ident"):

            quantiles -= 1
            if single_system:
                quantiles_5 -= 1
                quantiles_20 -= 1
                for ir in range(1, max_ir_len+1):
                    ols_quantile[ir] -= 1
                    ols_quantile_5[ir] -= 1
                    ols_quantile_20[ir] -= 1

        pred_ckpts = [batch_size*gpus*ckpt for ckpt in pred_ckpts] #trained on 2 GPUs
        # # for the ortho regular training run since the first ckpt was trained on 3 GPUs
        # count = 0
        # for pred_ckpt in pred_ckpts:
        #     if pred_ckpt == 3000:
        #         pred_ckpts[count] = 3*pred_ckpt
        #     else:
        #         pred_ckpts[count] = 3*3000 + 2*(pred_ckpt - 3000)

        #     count += 1

        print("quantiles shape", quantiles.shape)    
        ##plotting stuff
        ax.plot(pred_ckpts, quantiles[:,1], marker=".", linewidth=3, color= colors[0], label=(trainAs[i] if not single_system else "") + " TF" + (" 1 after" if single_system else ""), markersize=5 if valA == "gaussA" else 10)

        if not valA == "gaussA":
            plt.fill_between(pred_ckpts, quantiles[:,0], quantiles[:,2], color=colors[0], alpha=0.2) #, label='25th-75th Percentile Range')
        if single_system:
            ax.plot(pred_ckpts, quantiles_5[:,1], marker=".", linewidth=3, color= colors[1], label=(trainAs[i] if not single_system else "") + " TF" + (" 5 after" if single_system else ""), markersize=5 if valA == "gaussA" else 10)
            if not valA == "gaussA":
                plt.fill_between(pred_ckpts, quantiles_5[:,0], quantiles_5[:,2], color=colors[1], alpha=0.2) #, label='25th-75th Percentile Range')

            ax.plot(pred_ckpts, quantiles_20[:,1], marker=".", linewidth=3, color= colors[2], label=(trainAs[i] if not single_system else "") + " TF" + (" 20 after" if single_system else ""), markersize=5 if valA == "gaussA" else 10)

            if not valA == "gaussA":
                plt.fill_between(pred_ckpts, quantiles_20[:,0], quantiles_20[:,2], color=colors[2], alpha=0.2) #, label='25th-75th Percentile Range')

            if not (valA == "ortho" or valA == "ident"):
                for ir in range(2, max_ir_len+1):
                    ols_markers = ["x", "o"]
                    ax.plot(pred_ckpts, [ols_quantile[ir][1]]*len(pred_ckpts), linewidth=2, color= colors[1 + ir], label=(trainAs[i] if not single_system else "") + f" OLS ir={ir}" + (" 1 after" if single_system else ""),marker=ols_markers[ir - 2], markersize=1, linestyle="-")
                    # plt.fill_between(pred_ckpts, [ols_quantile[ir][0]]*len(pred_ckpts), [ols_quantile[ir][2]]*len(pred_ckpts), color=colors[3], alpha=0.05) #, label='25th-75th Percentile Range')

                    ax.plot(pred_ckpts, [ols_quantile_5[ir][1]]*len(pred_ckpts), linewidth=2, color= colors[1 + ir], label=(trainAs[i] if not single_system else "") + f" OLS ir={ir}" + (" 5 after" if single_system else ""), marker=ols_markers[ir - 2], markersize=0, linestyle=":")
                    # plt.fill_between(pred_ckpts, [ols_quantile_5[ir][0]]*len(pred_ckpts), [ols_quantile_5[ir][2]]*len(pred_ckpts), color=colors[4], alpha=0.05) #, label='25th-75th Percentile Range')

                    ax.plot(pred_ckpts, [ols_quantile_20[ir][1]]*len(pred_ckpts), linewidth=2, color= colors[1 + ir], label=(trainAs[i] if not single_system else "") + f" OLS ir={ir}" + (" 20 after" if single_system else ""), marker=ols_markers[ir - 2], markersize=0, linestyle="--")
                    # plt.fill_between(pred_ckpts, [ols_quantile_20[ir][0]]*len(pred_ckpts), [ols_quantile_20[ir][2]]*len(pred_ckpts), color=colors[5], alpha=0.05) #, label='25th-75th Percentile Range')

        torch.cuda.empty_cache()
        gc.collect()

        if single_system:
            print(f"no title for single system")
            # ax.set_title(f"Error" + (" Ratio" if not (valA == "ortho" or valA == "ident") else "") + " of Instance After Punctuation vs Training Iteration: " + ("Gaussian" if valA == "gaussA" else ("Orthogonal" if valA == "ortho" else ("Identity" if valA == "ident" else ""))) + " Test Distribution." + (" NoPE" if nope else ""), fontsize=8)
        else:
            ax.set_title(f"Error Ratio of Median Test System vs Training Iteration: Gaussian Test Distribution.")
        ax.grid(True)

        ax.set_ylabel("Error of Instance After Punctuation" + (" / Emp Kal Error" if not (valA == "ortho" or valA == "ident") else ""), fontsize=12)
        ax.set_xlabel("# of Training Examples", fontsize=12)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        ax.legend(loc="lower left" if single_system else "upper right")
        ax.set_yscale("log")
        ax.set_xscale("log")

        # ax.set_ylim([10e-3, 10e0])

        # Set the font size of the tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)  
        ax.tick_params(axis='both', which='minor', labelsize=12) 

        # fig.text(0.5, 0.01, f'Generated at {plot_time}', ha='center')

        plt.tight_layout()
        os.makedirs(os.path.dirname(f"../outputs/GPT2" + ("_NoPE" if nope else "") + "/" + experiment + "/figures/"), exist_ok=True)
        #save the figures
        fig.savefig(f"../outputs/GPT2" + ("_NoPE" if nope else "") + "/" + experiment + f"/figures/" + ("nope_" if nope else "") + f"{valA}_train_conv_single_sys.pdf", format='pdf', bbox_inches='tight')
        if i ==0:
            lr_med = quantiles[1,:]
            lr_pred_ckpts = pred_ckpts

        i+=1

    return lr_med, lr_pred_ckpts