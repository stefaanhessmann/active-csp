import os


class Properties:
    e_rel = "relaxation_energy"
    e = "energy"
    f = "forces"
    s = "stress"
    u = "uncertainty"
    e_u = "energy_uncertainty"
    f_u = "force_uncertainty"
    s_u = "stress_uncertainty"
    struc = "structure"
    prediction_properties = [e_rel]
    relaxation_properties = [e, f, s]


class Paths:
    inference_model = "inference_model"
    input_structure = "input_structure.cif"
    reference_computation_dir = "reference_computation"
    reference_optimization_dir = "reference_optimization"
    pool_optimization_dir = "pool_optimization"
    compute_representations_dir = "compute_representations"
    train_dir = "train"
    data_dir = "data"
    work_dir = "work"
    fin_dir = "fin"
    candidate_pool = os.path.join(data_dir, "candidate_pool.db")
    initial_pool = os.path.join(data_dir, "initial_pool.db")
    input_db = "input.db"
    output_db = "output.db"
    train_db = os.path.join(data_dir, "train.db")
    models_dir = os.path.join(data_dir, "models")
    splits_dir = os.path.join(data_dir, "splits")
    inference_models = os.path.join(models_dir, "inference_models")
    state_file = "state.yaml"
    done_file = ".done"

    @classmethod
    def idx_to_dir(cls, idx: int) -> str:
        return str(idx).zfill(8)

    @classmethod
    def workdir_to_idx(cls, dirname: str) -> int:
        return int(os.path.basename(dirname))
