import os
import torch
import numpy as np
import pandas as pd
import json
from loguru import logger

# DeepISA Imports
from deepISA.model.cnn import Conv
from deepISA.model.preprocess import compile_training_data
from deepISA.model.train import train_model
from deepISA.score.mapper import map_motifs, map_motif_finemo
from deepISA.score.single_isa import run_single_isa, calc_pred_orig

from deepISA.score.combi_isa import run_combi_isa

from deepISA.score.aggregate_isa import (
    calc_interaction,
    calc_tf_importance,
    calc_coop_score,
)

from deepISA.score.non_motifs import calc_non_motif_stats


from deepISA.utils import setup_logger, find_available_gpu

# Motif discovery (tf-modisco-lite + Fi-NeMo) -- optional, additive API.
from deepISA.discover import (
    compute_attribution,
    prepare_modisco_input,
    run_modisco,
    build_finemo_input,
    run_finemo_scan,
    load_hits_with_annotation,
    load_motifs,
)
from deepISA.discover.modisco import read_attribution_h5


# Plot functions
from deepISA.plot.interaction import (
    plot_interaction_decay, 
)

from deepISA.plot.null import (
    plot_null_isa,
    plot_motif_length,
    plot_null_interaction,
    plot_motif_distance
)

from deepISA.plot.cooperativity import (
    hist_coop_score,
    heatmap_coop_score,
    plot_motif_distance_by_category
)

from deepISA.plot.tf import (
    plot_motif_gc_by_coop,
    plot_coop_vs_importance,
    plot_partner_specificity
)


# exploring functions
from deepISA.explore.tf_family import (
    plot_coop_by_tf_pair_family,
    plot_coop_by_dbd,
    plot_intra_family_coop_score
)

from deepISA.explore.tf_pair_ppi import (
    plot_ppi_enrichment,
    plot_cofactor_recruitment,
    plot_dna_mediated_ppi
)
from deepISA.explore.tf_function import (
    plot_usf_pfs,
    plot_cell_specificity
)

ISA_STAGES = [
    "map_motifs",
    "calc_orig",
    "single_isa",
    "combi_isa",
    "calc_background",
    "aggregate_isa",
]

class QuickStart:
    def __init__(self, 
                 results_dir, 
                 fasta_path, 
                 df_input, 
                 device=None):
        # create results directory and subdirectories for data, plots, and models
        self.results_dir = results_dir
        self.data_dir = os.path.join(self.results_dir, "Data")
        self.plots_dir = os.path.join(self.results_dir, "Plots")
        self.model_dir = os.path.join(self.results_dir, "Models")
        # tf-modisco-lite / Fi-NeMo products live in their own subdirectory.
        self.motif_dir = os.path.join(self.data_dir, "Motif_discovery")
        setup_logger(self.results_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.motif_dir, exist_ok=True)
        # register paths for key files that will be generated and used throughout the pipeline
        self.files = {
            "motif_locs":    os.path.join(self.data_dir, "motif_locs.csv"),
            "non_motif_locs":os.path.join(self.data_dir, "non_motif_locs.csv"),
            "non_motif_isa": os.path.join(self.data_dir, "non_motif_isa.csv"),
            "pred_orig":     os.path.join(self.data_dir, "pred_orig.csv"),
            "isa_single":    os.path.join(self.data_dir, "motif_single_isa.csv"),
            "isa_combi":     os.path.join(self.data_dir, "motif_combi_isa.csv"),
            "non_motif_interaction":os.path.join(self.data_dir, "non_motif_interaction.csv"),
            "non_motif_tau": os.path.join(self.data_dir, "non_motif_interaction_tau.json"),
            "imp_tf":        os.path.join(self.data_dir, "tf_importance.csv"),
            "coop_tf_pair":  os.path.join(self.data_dir, "coop_tf_pair.csv"),
            "coop_tf":       os.path.join(self.data_dir, "coop_tf.csv"),
            # Motif discovery (tf-modisco-lite + Fi-NeMo) products, written by
            # the run_isa(isa_config={"method": [..., "modisco-finemo"]}) path.
            "attr_h5":       os.path.join(self.motif_dir, "attribution.h5"),
            "modisco_h5":    os.path.join(self.motif_dir, "modisco_results.h5"),
            "discovered_motifs": os.path.join(self.motif_dir, "discovered_motifs.csv"),
            "finemo_npz":    os.path.join(self.motif_dir, "finemo_input.npz"),
            "finemo_hits":   os.path.join(self.motif_dir, "finemo_hits.tsv"),
            "motif_report":  os.path.join(self.motif_dir, "MotifReport"),
            "discovered_meme": os.path.join(self.motif_dir, "discovered_motifs.meme"),
        }
        self.fasta_path = fasta_path
        self.df_input = df_input # can be either positive or negative regions
        self.df_labeled = None
        self.device = device if device is not None else find_available_gpu()
        

    def define_model(self, 
                     model_config=None,
                     model_obj=None, 
                     mode='dual',
                     alpha_genome_config=None):
        """
        Internalizes a model.
        Pass a pre-instantiated object, OR pass params (ks, cs, ds, seq_len) to
        build the internal Conv model, OR pass model_obj="AlphaGenome" with
        alpha_genome_config (YAML path or dict) to use the AlphaGenome API
        backend:

            pipe.define_model(model_obj="AlphaGenome", alpha_genome_config=config)

        The config is materialized to <model_dir>/ag_config.yaml (so the
        persistent cache lives next to it) and seq_len is auto-synced to the
        pipeline's region length. Only output_type / biosample_name (or a
        tracks list) are scientific choices; everything else has defaults.
        """
        if isinstance(model_obj, str):
            if model_obj.strip().lower() != "alphagenome":
                raise ValueError(
                    f"Unknown model_obj={model_obj!r}. Pass a model instance, "
                    "'AlphaGenome', or use model_config to build a Conv."
                )
            self._define_alphagenome(alpha_genome_config)
            return
        self.mode = mode
        if model_obj is not None:
            self.model = model_obj.to(self.device)
            logger.info("External model internalized successfully.")
        elif model_config is not None:
            self.model_config = model_config 
            # Build the internal Conv class from provided cnn.py
            self.model = Conv(self.mode, self.model_config).to(self.device) 
            logger.info(f"Internal Conv model initialized. Receptive field: {self.model.rf}")
            with open(os.path.join(self.model_dir, "model_config.json"), 'w') as f:
                json.dump(model_config, f, indent=4)

    @staticmethod
    def _default_tracks(model):
        """[0] for single-output models; every track for multi-track backends."""
        n = getattr(model, "n_tracks", None)
        return list(range(n)) if n else [0]

    def _infer_region_len(self) -> int:
        """Region length of the pipeline's input df (mode); 600 as fallback."""
        try:
            lens = (self.df_input["end"] - self.df_input["start"]).astype(int)
            if len(lens):
                return int(lens.mode().iloc[0])
        except Exception:
            pass
        return 600

    def _define_alphagenome(self, alpha_genome_config):
        """Build the AlphaGenome adapter backend from a YAML path or dict."""
        import yaml
        from deepISA.model.alpha_genome_adapter import AlphaGenomeAdapter

        if alpha_genome_config is None:
            raise ValueError(
                "model_obj='AlphaGenome' requires alpha_genome_config "
                "(a YAML path or a dict)."
            )
        if isinstance(alpha_genome_config, dict):
            cfg = dict(alpha_genome_config)
        else:
            with open(alpha_genome_config) as f:
                cfg = yaml.safe_load(f) or {}

        # The pipeline's region length is the source of truth — a mismatched
        # seq_len would make every forward() fail the length check.
        seq_len = self._infer_region_len()
        if cfg.get("seq_len") not in (None, seq_len):
            logger.info(
                f"AlphaGenome seq_len synced {cfg.get('seq_len')} -> {seq_len} "
                "to match the pipeline regions."
            )
        cfg["seq_len"] = seq_len

        # Materialize next to the model artefacts so the adapter's persistent
        # cache has a stable home across sessions (crash-safe resume).
        cfg_path = os.path.join(self.model_dir, "ag_config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        self.model = AlphaGenomeAdapter(cfg_path).to(self.device)
        logger.info(
            f"AlphaGenome backend ready (n_tracks={self.model.n_tracks}). "
            f"Config + persistent cache live next to {cfg_path}."
        )
        return self.model

    def train(self, 
              trainer_config=None,
              bw_paths=None,
              target_reg_col="target_reg",
              target_class_col=None):
        """
        Compiles training data and executes the Trainer. 
        The best model is automatically saved by the Trainer class.
        """
        if self.model is None:
            raise ValueError("Model not defined. Call define_model() first.")
        
        if trainer_config is None:
            trainer_config={
                "epochs": 10,
                "batch_size": 128,
                "patience": 3,
                "min_delta": 0.001, # minimum change in the monitored metric to qualify as an improvement
                "learning_rate": 1e-3,
                "save_one_fourth": False,
                "save_one": False,
                "log_transform":True,
                "rc_aug":True
            }

        
        train_data_path = os.path.join(self.data_dir, "Training_data")
        # 1. Compile Data
        logger.info("Compiling training data...")
        self.df_labeled = compile_training_data(
            df=self.df_input,
            fasta_path=self.fasta_path,
            out_dir=train_data_path,
            seq_len=self.model_config['seq_len'],
            bw_paths=bw_paths,
            target_reg_col=target_reg_col,
            target_class_col=target_class_col,
            log_transform=trainer_config.get('log_transform', True),
            rc_aug=trainer_config.get('rc_aug', True)
        )
        
        train_model(
            model=self.model,
            device=self.device,
            train_dat_dir=train_data_path,
            trainer_config=trainer_config,
            mode=self.mode,
            model_dir=self.model_dir, 
        )
        logger.info(f"Training complete. Check {self.model_dir} for model_best.pt")


    def load_checkpoint(self, suffix="best"):
        """Explicitly loads a specific checkpoint into self.model."""
        if self.model is None:
            raise ValueError("Model structure not defined. Call define_model first.")
        filename = f"model{suffix}.pt"
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {filename} not found at {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        logger.info(f"Successfully loaded checkpoint: {filename}")
        return self.model
    
    
    
    def _validate_start_from(self, start_from):
        if start_from not in ISA_STAGES:
            raise ValueError(f"Invalid start_from='{start_from}'. Expected one of: {ISA_STAGES}")
    
    
    def _check_isa_dependencies(self, start_from):
        # "map_motifs" and "calc_orig" start from raw inputs — nothing to check.
        if start_from in ("map_motifs", "calc_orig"):
            return
        if start_from == "single_isa":
            required = [self.files["motif_locs"], self.files["pred_orig"]]
        elif start_from == "combi_isa":
            required = [self.files["isa_single"], self.files["pred_orig"]]        
        elif start_from == "calc_background":
            required = [self.files["non_motif_locs"], self.files["isa_combi"], self.files["pred_orig"]]  
        elif start_from == "aggregate_isa":
            required = [self.files["non_motif_tau"],self.files["isa_single"], self.files["non_motif_isa"], self.files["isa_combi"], self.files["non_motif_interaction"]]  
            required = []
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            missing_str = "\n".join([f" - {path}" for path in missing])
            raise FileNotFoundError(
                f"Cannot start from stage '{start_from}' because required files are missing:\n{missing_str}"
            )
    
    
    def run_isa(self, 
            jaspar_path,
            isa_config={},
            df_pos=None,
            expressed_tfs=None,
            start_from="map_motifs"): 
            
        if self.model is None:
            raise ValueError("Model not defined. Call define_model() first.")
        
        self._validate_start_from(start_from)
        self._check_isa_dependencies(start_from)
        
        # resolve df_pos
        if df_pos is None:
            if self.df_labeled is not None and 'target_class' in self.df_labeled.columns:
                df_pos = self.df_labeled[self.df_labeled['target_class'] == 1].copy()
                logger.info(f"{len(df_pos)} positive regions identified from input data. Use as input for ISA.")
            else:
                df_pos = self.df_input.copy()
                logger.warning("Using df_input as positives for ISA. Ensure this is intended.")
        else:
            logger.info(f"{len(df_pos)} positive regions provided for ISA.")
                        
        # Default tracks: all of them for multi-track backends (e.g. an
        # AlphaGenome adapter exposes n_tracks), [0] for single-output models.
        self.tracks = isa_config.get('tracks', self._default_tracks(self.model))
        start_idx = ISA_STAGES.index(start_from)

        # Motif-location source(s): ["JASPAR"] (default, unchanged behaviour),
        # ["modisco-finemo"] (de novo discovery on model attributions), or both
        # (hits are unioned into motif_locs).
        method = isa_config.get("method", ["JASPAR"])
        if isinstance(method, str):
            method = [method]
        for m in method:
            if m not in ("JASPAR", "modisco-finemo"):
                raise ValueError(
                    f"Unknown isa_config method {m!r}. "
                    "Expected 'JASPAR' and/or 'modisco-finemo'."
                )

        # 1. Motif Mapping
        if start_idx <= ISA_STAGES.index("map_motifs"):
            motif_frames = []
            if "JASPAR" in method:
                logger.info("Running stage: map_motifs (JASPAR)")
                map_motifs(
                    regions_df=df_pos,
                    jaspar_path=jaspar_path,
                    motif_outpath=self.files["motif_locs"],
                    non_motif_outpath=self.files["non_motif_locs"],
                    motif_score_thresh=isa_config.get('motif_score_threshold', 500),
                    remap_path=isa_config.get('remap_path', None),
                    expressed_tfs=expressed_tfs
                )

                if isa_config.get('subset_by_remap', False):
                    logger.info("Subsetting motif locations by ReMap evidence...")
                    df_temp = pd.read_csv(self.files["motif_locs"])
                    df_temp = df_temp[df_temp['remap_evidence'] == True].reset_index(drop=True)
                    df_temp.to_csv(self.files["motif_locs"], index=False)
                motif_frames.append(pd.read_csv(self.files["motif_locs"]))

            if "modisco-finemo" in method:
                logger.info("Running stage: map_motifs (modisco-finemo)")
                hits_df = self._discover_motifs_finemo(df_pos, isa_config)
                finemo_df = map_motif_finemo(
                    hits_df=hits_df,
                    regions_df=df_pos,
                    motif_outpath=self.files["motif_locs"],
                    non_motif_outpath=self.files["non_motif_locs"],
                )
                motif_frames.append(finemo_df)

            if len(motif_frames) > 1:
                # Union of both sources; column order follows the JASPAR frame
                # (extra columns such as remap_evidence are kept).
                from deepISA.score.mapper import get_non_motifs
                merged = (pd.concat(motif_frames, ignore_index=True)
                          .drop_duplicates()
                          .reset_index(drop=True))
                merged.to_csv(self.files["motif_locs"], index=False)
                get_non_motifs(df_pos, merged.copy()).to_csv(
                    self.files["non_motif_locs"], index=False)
                logger.info(f"Union motif locations from {method}: "
                            f"{len(merged)} rows -> {self.files['motif_locs']}")
        else:
            logger.info(f"Skipping stage: map_motifs (start_from='{start_from}')")


        # 2. calc pred_orig
        if start_idx <= ISA_STAGES.index("calc_orig"):
            logger.info("Running stage: calc_orig")
            df_pos["region"]=df_pos["chrom"] + ":" + df_pos["start"].astype(str) + "-" + df_pos["end"].astype(str)
            calc_pred_orig(
                model=self.model,
                fasta=self.fasta_path,
                regions_df=df_pos,
                tracks=self.tracks,
                outpath=self.files["pred_orig"],
                device=self.device,
                pred_batch_size=isa_config.get("pred_batch_size", 1024),
            )
        else:
            logger.info(f"Skipping stage: calc_orig (start_from='{start_from}')")


        # 3. Single ISA now writes raw then filters to final motif_single_isa.csv
        if start_idx <= ISA_STAGES.index("single_isa"):
            logger.info("Running stage: single_isa")
            run_single_isa(
                model=self.model,
                fasta=self.fasta_path,
                motif_locs_path=self.files["motif_locs"],
                pred_orig_path=self.files["pred_orig"],
                outpath=self.files["isa_single"],
                device=self.device,
                tracks=self.tracks,
                num_regions_per_batch=isa_config.get("num_regions_per_batch", 200),
                pred_batch_size=isa_config.get("pred_batch_size", 1024),
            )

        else:
            logger.info(f"Skipping stage: single_isa (start_from='{start_from}')")

        # 4. Combination ISA
        if start_idx <= ISA_STAGES.index("combi_isa"):
            logger.info("Running stage: combi_isa")
            run_combi_isa(
                model=self.model,
                fasta=self.fasta_path,  
                single_isa_path=self.files["isa_single"],  
                outpath=self.files["isa_combi"],
                device=self.device,
                tracks=self.tracks,
                receptive_field=isa_config.get("receptive_field", getattr(self.model, "rf", 255)),
                pred_orig_path=self.files["pred_orig"],     
                num_regions_per_batch=isa_config.get("num_regions_per_batch", 200),
                pred_batch_size=isa_config.get("pred_batch_size", 1024),
            )
        else:
            logger.info(f"Skipping stage: combi_isa (start_from='{start_from}')")

        # 5. non-motif backgrounds
        if start_idx <= ISA_STAGES.index("calc_background"):
            logger.info("Running stage: calc_background")
            calc_non_motif_stats(
                model=self.model,
                fasta=self.fasta_path,
                non_motif_locs_path=self.files["non_motif_locs"],
                single_isa_path=self.files["isa_single"],
                combi_isa_path=self.files["isa_combi"],
                pred_orig_path=self.files["pred_orig"],
                non_motif_isa_outpath=self.files["non_motif_isa"],
                non_motif_interaction_outpath=self.files["non_motif_interaction"],
                device=self.device,
                tracks=self.tracks,
                num_regions_per_batch=isa_config.get("num_regions_per_batch", 200),
                pred_batch_size=isa_config.get("pred_batch_size", 1024),
                n_samples=isa_config.get("n_samples", 2048),
                n_bins=isa_config.get("n_bins", 20),
            )
        else:
            logger.info(f"Skipping stage: calc_background (start_from='{start_from}')")

        # 6. Aggregate / calculate cooperativity scores
        if start_idx <= ISA_STAGES.index("aggregate_isa"):
            with open(self.files["non_motif_tau"]) as f:
                tau_map = {int(k): v for k, v in json.load(f).items()}
            
            # make sure if null_interaction is set to non_motif_interaction, normalize_interaction must be True
            if isa_config.get("null_interaction", "distant_pairs") == "non_motif_interaction":
                logger.info("Setting 'normalize_interaction' to True because 'null_interaction' uses 'non_motif_interaction'")
                isa_config["normalize_interaction"] = True
                
            calc_interaction(
                combi_isa_path=self.files["isa_combi"],
                tracks=self.tracks,
                tau_map=tau_map,                         
                isa_thresh_quantile=isa_config.get("isa_thresh_quantile", 0.1),
                single_isa_path=self.files["isa_single"], # use to derive isa_thresh
                normalize=isa_config.get("normalize_interaction", False)
            )
            
            calc_tf_importance(
                self.files["isa_single"],
                self.files["imp_tf"],    
                non_motif_isa_path=self.files["non_motif_isa"],                    
                min_count=isa_config.get("min_count", 10),
            )

            for t in self.tracks:
                calc_coop_score(
                    self.files["isa_combi"],
                    outpath=self.files["coop_tf_pair"].replace(".csv", f"_t{t}.csv"),
                    level="tf_pair",   
                    track_idx=t,
                    min_count=isa_config.get("min_count", 10),
                    q_val_thresh=isa_config.get("q_val_thresh", 0.1),
                )
                                
                calc_coop_score(
                    self.files["isa_combi"],             
                    outpath=self.files["coop_tf"].replace(".csv", f"_t{t}.csv"),
                    level="tf",
                    track_idx=t,
                    min_count=isa_config.get("min_count", 10),
                    q_val_thresh=isa_config.get("q_val_thresh", 0.1),
                )
        else:
            logger.info(f"Skipping stage: calc_coop_score (start_from='{start_from}')")

        logger.info("ISA execution and aggregation complete.")
        
    
    def report(self, tracks=None):
        """
        Executes the full suite of visualization and exploration functions.
        """
        if tracks is not None:
            self.tracks = tracks
        if not hasattr(self, "tracks") or self.tracks is None:
            raise ValueError(
                "Tracks are not set. Pass tracks to report(tracks=[...]) "
                "or run run_isa(...) first."
            )
        logger.info("Generating comprehensive reports and plots...")
        
        # --- A. Null Distributions (null.py) ---
        
        plot_null_isa(self.files["non_motif_isa"],
                    tracks=self.tracks, 
                    outpath=os.path.join(self.plots_dir, f"null_isa.png"))
        plot_motif_length(self.files["non_motif_isa"],
                          self.files["motif_locs"],
                          outpath=os.path.join(self.plots_dir, f"motif_length.png"))
        plot_null_interaction(self.files["non_motif_interaction"],
                              tracks=self.tracks, 
                              outpath=os.path.join(self.plots_dir, f"null_interaction.png"))
        plot_motif_distance(self.files["non_motif_interaction"],
                            self.files["isa_combi"],
                            outpath=os.path.join(self.plots_dir, f"motif_distance.png"))
        # --- B. Interaction Plots (interaction.py) ---
        
        df_isa_combi = pd.read_csv(self.files["isa_combi"])
        plot_interaction_decay(df_isa_combi, 
                               self.tracks, 
                               mode='signed', 
                               outpath=os.path.join(self.plots_dir, f"interaction_decay_signed.png"))
        
        for t in self.tracks:
            t_suffix = f"_t{t}"
            # Helper to generate output paths
            def ppath(name): return os.path.join(self.plots_dir, f"{name}{t_suffix}.png")
            # 1. Load the specific results for this track
            coop_pair_path = self.files["coop_tf_pair"].replace(".csv", f"_t{t}.csv")
            coop_tf_path = self.files["coop_tf"].replace(".csv", f"_t{t}.csv")
            imp_path = self.files["imp_tf"]
            
            if not os.path.exists(coop_pair_path) or not os.path.exists(coop_tf_path):
                logger.warning(f"Results for track {t} not found. Skipping.")
                continue
                
            df_coop_pair = pd.read_csv(coop_pair_path)
            df_coop_tf = pd.read_csv(coop_tf_path)
            df_imp = pd.read_csv(imp_path)

            # --- C. Cooperativity Distribution (cooperativity.py) ---
            hist_coop_score(df_coop_pair, outpath=ppath("coop_score_hist"))
            heatmap_coop_score(df_coop_pair, outpath=ppath("coop_score_heatmap"))
            plot_motif_distance_by_category(df_coop_pair, outpath=ppath("distance_by_category"))

            # --- D. TF Importance & GC (tf.py) ---
            plot_motif_gc_by_coop(df_coop_tf, outpath=ppath("motif_gc_by_coop"))
            plot_coop_vs_importance(df_coop_tf, df_imp, 
                                     x_col="coop_score", 
                                     y_col=f"mean_isa_t{t}", 
                                     outpath=ppath("coop_vs_importance"))
            plot_partner_specificity(df_coop_pair, df_coop_tf, outpath=ppath("partner_specificity_ratio"))

            # --- E. TF Family Exploration (tf_family.py) ---
            plot_coop_by_tf_pair_family(df_coop_pair, outpath=ppath("family_coop_summary"))
            plot_coop_by_dbd(df_coop_tf, outpath=ppath("dbd_coop_summary"))
            plot_intra_family_coop_score(df_coop_pair, outpath=ppath("intra_family_distribution"))

            # --- F. TF Functional Evolution (tf_function.py) ---
            plot_usf_pfs(df_coop_tf, outpath=ppath("usf_pioneer_ecdf"))
            plot_cell_specificity(df_coop_tf, outpath=ppath("rolling_gini_specificity"))

            # --- G. PPI Validation (tf_pair_ppi.py) ---
            plot_ppi_enrichment(df_coop_pair, rank_by="coop_score", outpath=ppath("ppi_enrichment_by_coop_score"))
            plot_ppi_enrichment(df_coop_pair, rank_by="p_val", outpath=ppath("ppi_enrichment_by_p_val"))
            plot_cofactor_recruitment(df_coop_pair, outpath=ppath("ppi_violin_validation"))
            plot_dna_mediated_ppi(df_coop_pair, rank_by="coop_score", outpath=ppath("dna_ppi_enrichment_by_score"))
            plot_dna_mediated_ppi(df_coop_pair, rank_by="p_val", outpath=ppath("dna_ppi_enrichment_by_pval"))
        logger.info(f"Report complete. All plots saved to {self.plots_dir}")

    # ------------------------------------------------------------------
    # Motif discovery: tf-modisco-lite + Fi-NeMo
    # ------------------------------------------------------------------
    # These methods are *additive*: they do not touch run_isa() and can be
    # called independently or via the run_motif_discovery() orchestrator.
    def _resolve_positives(self, df_pos):
        """Pick the positive-region DataFrame, mirroring run_isa() semantics."""
        if df_pos is not None:
            logger.info(f"{len(df_pos)} positive regions provided for motif discovery.")
            return df_pos
        if self.df_labeled is not None and 'target_class' in self.df_labeled.columns:
            df_pos = self.df_labeled[self.df_labeled['target_class'] == 1].copy()
            logger.info(f"{len(df_pos)} positive regions identified from labeled data.")
            return df_pos
        logger.warning("Using df_input as positives for motif discovery. Ensure this is intended.")
        return self.df_input.copy()

    def _positives_to_onehot(self, df_pos):
        """Load FASTA once and one-hot encode all positive regions -> (N, 4, L)."""
        from deepISA.utils import get_sequences_from_df, one_hot_encode, load_fasta
        fasta = load_fasta(self.fasta_path)
        seqs = get_sequences_from_df(df_pos, fasta)
        return one_hot_encode(seqs)

    def _predict_activity(self, df_pos, track):
        """Model prediction per region on one output track -> 1D np.ndarray."""
        X = self._positives_to_onehot(df_pos)
        self.model.eval()  # dropout off -> ranking is deterministic
        preds = []
        with torch.no_grad():
            for s in range(0, len(X), 256):
                xb = torch.from_numpy(X[s:s + 256]).to(self.device)
                y = self.model(xb).detach().cpu().numpy()
                preds.append(y)
        pred = np.concatenate(preds, axis=0)
        return pred[:, track] if pred.ndim == 2 and pred.shape[1] > 1 else pred.ravel()

    def _select_modisco_regions(self, df_pos, tracks, top_frac=0.1, rank_by=None,
                                drop_N=True, max_drop_frac=0.2):
        """Curate the motif-discovery input set (mc000's ``y >= THRESH`` analog).

        N policy with a safety valve: N-containing regions are dropped only
        while they stay within ``max_drop_frac`` of the input. Beyond that the
        drop would remove too much data, so all regions are kept and their
        unknown bases are imputed with random ACGT during attribution instead.
        """
        from deepISA.discover import select_top_regions, drop_non_acgt_regions
        if drop_N and len(df_pos) > 0:
            kept, n_dropped = drop_non_acgt_regions(df_pos, self.fasta_path)
            frac = n_dropped / len(df_pos)
            if frac <= max_drop_frac:
                df_pos = kept
                if n_dropped:
                    logger.info(
                        f"Dropped {n_dropped}/{len(df_pos) + n_dropped} region(s) containing "
                        f"unknown bases (N): {frac:.0%} <= max_drop_frac {max_drop_frac:.0%}."
                    )
            else:
                logger.warning(
                    f"{n_dropped}/{len(df_pos)} ({frac:.0%}) regions contain N -- more than "
                    f"max_drop_frac {max_drop_frac:.0%}. Keeping them; unknown bases will be "
                    f"imputed with random ACGT during attribution instead of dropping."
                )
        if top_frac is not None:
            if len(df_pos) == 0:
                raise ValueError("No regions left after dropping N-containing ones.")
            if rank_by is not None:
                score = df_pos[rank_by].to_numpy(dtype=float)
                logger.info(f"Ranking {len(df_pos)} regions by column '{rank_by}'.")
            else:
                score = self._predict_activity(df_pos, tracks[0])
                logger.info(
                    f"Ranking {len(df_pos)} regions by model prediction on track {tracks[0]}."
                )
            df_pos = select_top_regions(df_pos, score, top_frac=top_frac)
            logger.info(f"Motif discovery input: top {top_frac:.0%} -> {len(df_pos)} regions.")
        return df_pos


    # ------------------------------------------------------------------
    # De novo motif discovery (tf-modisco-lite + Fi-NeMo) — internal stage
    # of run_isa(isa_config={"method": [..., "modisco-finemo"]}). Products
    # live under Data/Motif_discovery/ (self.files["attr_h5" ...]).
    # ------------------------------------------------------------------
    def _discover_motifs_finemo(self, df_pos, isa_config):
        """Discover motifs de novo and scan every input region for hits.

        Runs the former run_modisco + run_finemo bodies as one pipeline:
        input-set curation -> attribution (tangermeme DeepLIFT-SHAP) ->
        ``modisco motifs`` -> Fi-NeMo input -> ``finemo call-hits`` ->
        annotated hits (also saved to ``self.files["finemo_hits"]``).

        All knobs come from ``isa_config`` with the previous defaults:
        ``discovery_top_frac``, ``discovery_rank_by``, ``discovery_drop_N``,
        ``discovery_max_drop_frac``, ``n_refs``, ``attr_batch_size``,
        ``modisco_window``, ``modisco_n_seqlets``, ``finemo_lam``,
        ``finemo_max_steps``.

        Returns the annotated hits DataFrame (for map_motif_finemo).
        """
        if self.model is None:
            raise ValueError("Model not defined. Call define_model() first.")
        tracks = list(getattr(self, "tracks", None) or [0])

        top_frac   = isa_config.get("discovery_top_frac", 0.1)
        rank_by    = isa_config.get("discovery_rank_by", None)
        drop_N     = isa_config.get("discovery_drop_N", True)
        max_drop   = isa_config.get("discovery_max_drop_frac", 0.2)
        n_refs     = isa_config.get("n_refs", 100)
        attr_bs    = isa_config.get("attr_batch_size", 64)
        window     = isa_config.get("modisco_window", None)
        n_seqlets  = isa_config.get("modisco_n_seqlets", 50000)
        lam        = isa_config.get("finemo_lam", 0.7)
        max_steps  = isa_config.get("finemo_max_steps", 10000)

        # Fresh discovery on a curated input set; an existing attribution H5
        # is reused (same policy as before) so re-runs skip straight to finemo.
        if not os.path.exists(self.files["attr_h5"]):
            curated = self._select_modisco_regions(
                df_pos, tracks, top_frac=top_frac, rank_by=rank_by,
                drop_N=drop_N, max_drop_frac=max_drop)
            if len(curated) == 0:
                raise ValueError("Empty motif-discovery input set after curation.")
            if "region" not in curated.columns:
                # Region strings become Fi-NeMo's peak_name and are parsed back
                # into genomic coordinates by map_motif_finemo.
                curated["region"] = (curated["chrom"] + ":"
                                     + curated["start"].astype(str) + "-"
                                     + curated["end"].astype(str))
            seqs_ohe = self._positives_to_onehot(curated)
            ids = curated["region"].astype(str).tolist()
            logger.info(f"Computing attribution for {len(seqs_ohe)} regions, tracks={tracks}.")
            compute_attribution(
                model=self.model,
                seqs_ohe=seqs_ohe,
                tracks=tracks,
                device=self.device,
                n_refs=n_refs,
                batch_size=attr_bs,
                save_h5_path=self.files["attr_h5"],
                ids=ids,
            )

        # Attribution H5 -> NPZ pair for the modisco CLI (first track only:
        # averaging regression+classification attributions mixes signals).
        ohe_npz, hyp_npz = prepare_modisco_input(
            h5_path=self.files["attr_h5"],
            out_dir=os.path.join(self.motif_dir, "modisco_input"),
            track_index=0,
        )
        run_modisco(
            ohe_npz=ohe_npz,
            hyp_npz=hyp_npz,
            out_h5=self.files["modisco_h5"],
            n_seqlets=n_seqlets,
            window=window,
        )
        motifs = load_motifs(self.files["modisco_h5"], task_name="discovery")
        pd.DataFrame([
            {"motif_id": mid, "num_seqlets": m["num_seqlets"],
             "task": m["task"], "length": m["cwm"].shape[0]}
            for mid, m in motifs.items()
        ]).to_csv(self.files["discovered_motifs"], index=False)
        logger.info(f"Discovered {len(motifs)} motifs -> {self.files['discovered_motifs']}")

        # Fi-NeMo scan of the SAME regions the attribution covered. The region
        # ids stored in the H5 become Fi-NeMo's peak_name, which
        # map_motif_finemo parses back into genomic coordinates.
        seqs_4lc, hyp_4lc, _ = read_attribution_h5(
            self.files["attr_h5"], track_index=0)
        import h5py
        with h5py.File(self.files["attr_h5"], "r") as f:
            if "id" not in f:
                raise ValueError(
                    f"{self.files['attr_h5']} has no region ids (written by an "
                    "older flow). Delete it so attribution is recomputed with ids."
                )
            region_ids = [s.decode() if isinstance(s, bytes) else str(s)
                          for s in f["id"][:]]
        finemo_npz = build_finemo_input(
            seqs_ohe=seqs_4lc,
            hyp_scores=hyp_4lc,
            out_dir=os.path.dirname(self.files["finemo_npz"]),
            ids=region_ids,
        )
        hits_tsv = run_finemo_scan(
            npz_path=finemo_npz,
            out_dir=os.path.dirname(self.files["finemo_hits"]),
            motif_db_h5=self.files["modisco_h5"],
            lam=lam,
            max_steps=max_steps,
        )
        df_hits = load_hits_with_annotation(hits_tsv, self.files["modisco_h5"])
        df_hits.to_csv(self.files["finemo_hits"], sep="	", index=False)
        logger.info(f"Fi-NeMo hits ({len(df_hits)} rows) -> {self.files['finemo_hits']}")
        return df_hits
