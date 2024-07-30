import torch
import torch.nn as nn
import schnetpack.properties as structure
from schnetpack.data.loader import _atoms_collate_fn
from ase import Atoms
from typing import List


__all__ = ["AtomsConverter"]


class AtomsConverter:
    def __init__(self, transforms=None, unsqueeze_cell=True):
        # todo: remove this hack
        if transforms is None:
            transforms = [nn.Identity()]
        for transf in transforms:
            if hasattr(transf, "mode"):
                transf.mode = "pre"
        self.transform_module = torch.nn.Sequential(*transforms)
        self.unsqueeze_cell = unsqueeze_cell

    def __call__(self, atoms: Atoms or List):
        if type(atoms) == Atoms:
            atoms = [atoms]
        properties = [self._get_properties(atms) for atms in atoms]

        return _atoms_collate_fn(properties)

    def _get_properties(self, atoms: Atoms):
        """
        Similar to loading structure from dataset.
        """
        atms = atoms.copy()

        properties = {}
        properties[structure.idx] = torch.tensor([0])

        Z = atms.numbers.copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])
        properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
        properties[structure.position] = torch.tensor(atms.get_positions(wrap=True))
        properties[structure.cell] = torch.tensor(atms.cell.copy())
        if self.unsqueeze_cell:
            properties[structure.cell] = torch.unsqueeze(properties[structure.cell], 0)

        properties[structure.pbc] = torch.tensor(atms.pbc)

        # apply transforms
        properties = self.transform_module(properties)

        return properties
