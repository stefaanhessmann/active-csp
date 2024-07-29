import numpy as np
from ase.constraints import full_3x3_to_voigt_6_stress
from typing import Optional, List, Callable
import schnetpack as spk
from deepcryspy.force_field import AtomsConverter
from deepcryspy import Properties
from ase.calculators.calculator import Calculator, all_changes


__all__ = [
    "relative_uncertainty",
    "Calculator",
    "DummyCalculator",
    "EnsembleCalculator",
]


def relative_uncertainty(
    mean: np.array, std: np.array, aggregation_fn: Callable = np.max
) -> np.array:
    """
    Compute relative uncertainty.
    """
    return aggregation_fn(np.abs(std)) / aggregation_fn(np.abs(mean))


class Calculator:
    """
    Base class for ase calculators.

    """

    def __init__(self):
        self.results = dict()
        self.atoms = None

    def calculation_required(
        self,
        atoms,
    ):
        if self.atoms is None or not self.atoms == atoms:
            self.reset()
            return True
        return False

    def get_forces(
        self,
        atoms,
    ):
        if self.calculation_required(atoms):
            self.calculate(atoms)
        return self.results["forces"]

    def get_potential_energy(
        self,
        atoms,
    ):
        if self.calculation_required(atoms):
            self.calculate(atoms)
        return self.results["energy"]

    def get_stress(
        self,
        atoms,
    ):
        if self.calculation_required(atoms):
            self.calculate(atoms)
        return self.results["stress"]

    def calculate(self, atoms):
        pass

    def reset(self):
        self.results = dict()
        self.atoms = None


class DummyCalculator(Calculator):
    """
    Only for testing purposes. Replace with QE, LAMMPS, VASP, ... calculator from ase!

    """

    def __init__(self):
        super(DummyCalculator, self).__init__()

    def calculate(self, atoms):
        self.results = dict(
            energy=np.random.random(1).item(),
            forces=np.random.random(atoms.positions.shape),
            stress=np.random.random((3, 3)),
        )


class EnsembleCalculator(Calculator):
    """
    Calculator for neural network models with uncertainty prediction.
    NN-models must return a prediction and an uncertainty dict.
    """

    def __init__(
        self,
        ensemble: Optional = None,
        uncertainty_fn: Callable = relative_uncertainty,
        cutoff: Optional[float] = 7.0,
        device: Optional[str] = "cpu",
        callbacks: Optional[List] = None,
    ):
        super(EnsembleCalculator, self).__init__()
        if callbacks is None:
            callbacks = []
        self.ensemble = ensemble
        self.device = device
        self.atoms_converter = AtomsConverter(
            transforms=[
                spk.transform.SkinNeighborList(
                    spk.transform.MatScipyNeighborList(cutoff=cutoff),
                ),
                spk.transform.CastTo32(),
            ]
        )
        self.callbacks = callbacks
        self.metadata = dict()
        self.uncertainty_fn = uncertainty_fn

    def calculate(
        self,
        atoms,
    ):
        for callback in self.callbacks:
            callback.on_calculation_start(self, atoms)

        # update results
        if self.ensemble is not None:
            self.results = self._get_results(atoms)
            self.metadata.update(dict(trainable=False))

        # update atoms
        self.atoms = atoms.copy()

        for callback in self.callbacks:
            callback.on_calculation_end(self)

    def _get_results(self, atoms):
        # compute properties with NN ensemble
        self.ensemble.eval()
        inputs = self.atoms_converter(atoms)
        self.ensemble.to(self.device)
        self.ensemble.eval()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        means, stds = self.ensemble(inputs)
        means = {k: v.detach().cpu().squeeze().numpy() for k, v in means.items()}
        stds = {k: v.detach().cpu().squeeze().numpy() for k, v in stds.items()}

        # add means, stds to metadata
        self.metadata.update(dict(means=means, stds=stds))

        # compute uncertainty value
        uncertainty = self._get_uncertainty(means, stds)

        # collect results
        results = {
            Properties.e: means[Properties.e],
            Properties.f: means[Properties.f],
            Properties.s: means[Properties.s],
            "uncertainty": uncertainty,
            Properties.e_u: stds[Properties.e],
            Properties.f_u: stds[Properties.f],
            Properties.s_u: stds[Properties.s],
        }
        # transform stress to voigt 6 if needed
        if results[Properties.s].shape[0] == 3:
            results[Properties.s] = full_3x3_to_voigt_6_stress(results[Properties.s])

        return results

    def _get_uncertainty(self, means, stds):
        """
        Compute uncertainty value for NN ensemble.
        This function can be overwritten to apply other rules for uncertainty.

        """
        f_mean = means[Properties.f]
        s_mean = means[Properties.s]
        f_std = stds[Properties.f]
        s_std = stds[Properties.s]

        f_rel = self.uncertainty_fn(f_mean, f_std)
        s_rel = self.uncertainty_fn(s_mean, s_std)

        return np.max([s_rel, f_rel])

    def reset(self):
        self.atoms = None
        self.results = dict()
        self.metadata = dict()
