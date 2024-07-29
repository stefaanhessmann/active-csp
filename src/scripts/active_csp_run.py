import os
import shutil
import time
import uuid
from ase.db import connect
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from deepcryspy.utils.db import write_pool, mask_relaxation_ids, get_atoms
from deepcryspy.utils.state_tracker import StateTracker
from deepcryspy.job_handlers import (
    TrainHandler,
    ComputationHandler,
    OptimizationHandler,
    PoolOptimizationHandler,
    RepresentationComputationHandler,
    StoppingCriterion,
)
from deepcryspy.selection import (
    Clustering,
    CandidateSelection,
    random_selection,
    cluster_by_representations_old,
)
from deepcryspy import Paths
import logging
from schnetpack.utils.script import log_hyperparameters, print_config


log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid4()))


def initialize_experiment(config):
    # create paths
    os.makedirs(Paths.data_dir, exist_ok=False)
    os.makedirs(Paths.work_dir, exist_ok=False)
    os.makedirs(Paths.fin_dir, exist_ok=False)

    # initialize structure pool
    train_pool = []
    if config.candidate_pool.train_pool is not None:
        train_pool = hydra.utils.instantiate(
            config.candidate_pool.train_pool.generator
        ).gen_struc(config.candidate_pool.train_pool.n_structures)
    selection_pool = []
    if config.candidate_pool.selection_pool is not None:
        selection_pool = hydra.utils.instantiate(
            config.candidate_pool.selection_pool.generator
        ).gen_struc(config.candidate_pool.selection_pool.n_structures)

    write_pool(
        Paths.candidate_pool, train_pool=train_pool, selection_pool=selection_pool
    )
    shutil.copyfile(Paths.candidate_pool, Paths.initial_pool)

    with open("config.yaml", "w") as f:
        OmegaConf.save(config, f, resolve=False)


@hydra.main(config_path="configs", config_name="main", version_base="1.2")
def run(config: DictConfig):
    if os.path.exists(os.path.join(os.path.abspath("."), "config.yaml")):
        log.info(
            f"continue run from config file located at: {os.path.abspath('.')}/config.yaml"
        )
        config = OmegaConf.load(os.path.join(os.path.abspath("."), "config.yaml"))
        job_state = StateTracker(Paths.state_file)
    else:
        initialize_experiment(config)
        job_state = StateTracker(Paths.state_file, initial_state=dict(cycle=0))

    # initialize modules
    reference_optimization_handler: OptimizationHandler = hydra.utils.instantiate(
        config.reference_optimization_handler
    )
    reference_calculation_handler: ComputationHandler = hydra.utils.instantiate(
        config.reference_calculation_handler
    )
    train_handler: TrainHandler = hydra.utils.instantiate(config.train_handler)
    pool_optimization_handler: PoolOptimizationHandler = hydra.utils.instantiate(
        config.pool_optimization_handler
    )
    representation_handler: RepresentationComputationHandler = hydra.utils.instantiate(
        config.representation_handler
    )
    stopping_criterion: StoppingCriterion = hydra.utils.instantiate(
        config.stopping_criterion
    )
    job_handlers = [
        reference_optimization_handler,
        reference_calculation_handler,
        train_handler,
        pool_optimization_handler,
        representation_handler,
    ]
    clustering: Clustering = hydra.utils.instantiate(config.clustering)
    reference_selection: CandidateSelection = hydra.utils.instantiate(
        config.reference_selection
    )
    target_selection: CandidateSelection = hydra.utils.instantiate(
        config.target_selection
    )

    # iterative optimization loop
    max_cycles = 15
    best_ids = []
    while job_state.get_state("cycle") < max_cycles:
        # handle running jobs
        while any(
            [
                handler.jobs_running()
                for handler in job_handlers
                if handler != reference_optimization_handler
            ]
        ):
            active_handlers = [
                handler for handler in job_handlers if handler.jobs_running()
            ]
            for handler in active_handlers:
                handler.handle_jobs()
            # time.sleep(60)

        # structure selection
        if reference_calculation_handler.cycle != job_state.get_state("cycle"):
            log.info(f"starting structure selection...")
            # random selection in first iteration
            if job_state.get_state("cycle") == 0:
                reference_candidates = random_selection(
                    db_path=Paths.candidate_pool,
                    n_candidates=config.globals.n_candidates,
                    use_train_ids=config.reference_selection.use_train_ids,
                    use_select_ids=config.reference_selection.use_select_ids,
                )
                optimization_candidates, idx_mask = [], []
            # guided selection
            else:
                with connect(Paths.candidate_pool) as db:
                    representations = np.array(db.metadata["representations"])
                # cluster_labels = None
                cluster_labels = cluster_by_representations_old(
                    eps=config.clustering.eps,
                    min_sampled=config.clustering.min_sampled,
                    representations=representations,
                    min_different=config.clustering.min_different,
                    max_tries=config.clustering.max_tries,
                )

                # select candidates for computation of labels with reference method
                reference_candidates = reference_selection.select(
                    db_path=Paths.candidate_pool,
                    cluster_labels=cluster_labels,
                )

                # select most stable candidates for evaluation of local minima with reference method
            #    optimization_candidates, idx_mask = target_selection.select(
            #        db_path=Paths.candidate_pool,
            #        cluster_labels=cluster_labels,
            #    )
            # relaxation_ids_unmasked = [
            #    idx for idx, idx_masked in zip(optimization_candidates, idx_mask) if not idx_masked
            # ]
            #
            # with open("selection_log.txt", "a") as file:
            #    file.write(
            #        f"cycle: {job_state.get_state('cycle')}\n"
            #        f"selected ids: {optimization_candidates}\n"
            #        f"masked ids: {idx_mask}\n"
            #        "---\n\n"
            #    )
            # stopping criterion: no more promising candidates are found
            # if len(relaxation_ids_unmasked) == 0 and job_state.get_state("cycle") > 0:
            #    reference_optimization_handler.submit_jobs(
            #        atoms_list=get_atoms(Paths.candidate_pool, optimization_candidates),
            #    )
            #    raise SystemExit("No more promising candidates")

            # mask selected clusters
            # mask_relaxation_ids(
            #    db_path=Paths.candidate_pool,
            #    relaxation_ids=relaxation_ids_unmasked,
            # )
            log.info(
                f"selection of candidates for labeling done! selected structure ids: {reference_candidates}."
            )
            job_state.update_state(selection_cycle=job_state.get_state("cycle"))

            # submit reference computations
            noise = (
                (
                    config.reference_calculation.noise
                    * config.reference_calculation.noise_factor
                    ** job_state.get_state("cycle")
                )
                if any(
                    [
                        job_state.get_state("cycle") != 0,
                        config.reference_calculation.noise_on_first_cycle,
                    ]
                )
                else 0
            )
            reference_calculation_handler.submit_jobs(
                atoms_list=get_atoms(Paths.candidate_pool, reference_candidates),
                noise=noise,
            )
            log.info(f"submitted selected structures for labeling.")
            continue

        # submit training of force field models
        if train_handler.cycle != job_state.get_state("cycle"):
            train_handler.submit_jobs()
            continue

        # submit pool optimization jobs
        if pool_optimization_handler.cycle != job_state.get_state("cycle"):

            pool_optimization_handler.submit_jobs()
            continue

        # compute feature representations
        if representation_handler.cycle != job_state.get_state("cycle"):
            representation_handler.submit_jobs()
            continue

        # save checkpoint data
        shutil.move(
            config.paths.candidate_pool,
            config.paths.candidate_pool.replace(
                ".db", f"_{job_state.get_state('cycle')}.db"
            ),
        )
        shutil.move(
            config.paths.tmp_pool,
            config.paths.candidate_pool,
        )
        shutil.move(
            config.paths.inference_models,
            config.paths.inference_models + f"_{job_state.get_state('cycle')}",
        )

        # check stopping criterion
        stop, best_ids = stopping_criterion.evaluate(cycle=job_state.get_state("cycle"))
        print(
            f"*** finished cycle: {job_state.get_state('cycle')}. stop: {stop} --- best_ids: {best_ids} ***"
        )
        if stop:
            break

        # update job state
        job_state.update_state(cycle=job_state.get_state("cycle") + 1)
        continue

    # evaluate best structures
    reference_optimization_handler.submit_jobs(
        atoms_list=get_atoms(Paths.candidate_pool, best_ids),
    )


if __name__ == "__main__":
    run()
