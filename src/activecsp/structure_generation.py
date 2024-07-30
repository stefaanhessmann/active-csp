import json
import math
from contextlib import redirect_stdout
from io import StringIO
from logging import getLogger
import os
import random
from pymatgen.core import Structure
import numpy as np
from pyxtal import pyxtal
from pyxtal.tolerance import Tol_matrix

from pymatgen.io.ase import AseAtomsAdaptor
from ase.db import connect


__all__ = ["Rnd_struc_gen_pyxtal", "Rnd_struc_gen", "DBLoader"]


logger = getLogger("cryspy")


class DBLoader:

    def __init__(self, db_path):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"pool does not exist at {db_path}.")
        self.db_path = db_path

    def gen_struc(self, nstruc=None):
        atoms_list = []
        with connect(self.db_path) as db:
            if len(db) == 0:
                raise Exception(f"candidate pool at {self.db_path} is empty.")
            if nstruc is None:
                nstruc = len(db)
            for i in range(nstruc):
                atoms_list.append(db.get(i + 1))
        return atoms_list


class Rnd_struc_gen_pyxtal:
    """
    Random structure generation using pyxtal.

    Note:
    The implementation of this function is taken from the GitHub repository:
    https://github.com/Tomoki-YAMASHITA/CrySPY
    """

    def __init__(
        self,
        mindist,
        atype,
        dangle,
        maxlen,
        minlen,
        nat,
        natot,
        vol_mu=None,
        vol_sigma=None,
        vol_factor=None,
        symprec=0.01,
        maxcnt=50,
        spgnum="all",
        mol_file=None,
        algo=None,
    ):
        if vol_factor is None:
            vol_factor = [1.0, 1.1]
        self.mindist = mindist
        self.atype = atype
        self.dangle = dangle
        self.maxlen = maxlen
        self.mindist = mindist
        self.minlen = minlen
        self.nat = nat
        self.natot = natot
        self.vol_mu = vol_mu
        self.vol_sigma = vol_sigma
        self.vol_factor = vol_factor
        self.symprec = symprec
        self.maxcnt = maxcnt
        self.spgnum = spgnum
        self.mol_file = mol_file
        self.algo = algo

    def gen_struc(self, nstruc, id_offset=0, init_pos_path=None):
        """
        Generate random structures for given space groups

        # ---------- args
        nstruc (int): number of generated structures

        id_offset (int): default: 0
                         structure ID starts from id_offset
                         e.g. nstruc = 3, id_offset = 10
                              you obtain ID 10, ID 11, ID 12

        init_pos_path (str): default: None
                             specify a path of file
                             if you write POSCAR data of init_struc_data
                             ATTENSION: data are appended to the specified file

        # ---------- comment
        generated structure data are saved in self.init_struc_data
        """
        # ---------- initialize
        init_struc_data = {}
        # ---------- Tol_matrix
        tolmat = self._set_tol_mat(self.atype, self.mindist)
        # ---------- loop for structure generattion
        while len(init_struc_data) < nstruc:
            # ------ spgnum --> spg
            if self.spgnum == "all":
                spg = random.randint(1, 230)
            else:
                spg = random.choice(self.spgnum)
            # ------ vol_factor
            rand_vol = random.uniform(self.vol_factor[0], self.vol_factor[1])
            # ------ generate structure
            tmp_crystal = pyxtal()
            try:
                f = StringIO()
                with redirect_stdout(f):
                    tmp_crystal.from_random(
                        dim=3,
                        group=spg,
                        species=self.atype,
                        numIons=self.nat,
                        factor=rand_vol,
                        conventional=False,
                        tm=tolmat,
                    )
                s = f.getvalue().rstrip()  # to delete \n
                if s:
                    logger.warning(s)
            except Exception as e:
                logger.warning(e.args[0] + f": spg = {spg} retry.")
                continue
            if tmp_crystal.valid:
                tmp_struc = tmp_crystal.to_pymatgen(
                    resort=False
                )  # pymatgen Structure format
                # -- check the number of atoms
                if not self._check_nat(tmp_struc):
                    # (pyxtal 0.1.4) cryspy adopts "conventional=False",
                    #     which is better for DFT calculation
                    # pyxtal returns conventional cell, that is, too many atoms
                    tmp_struc = tmp_struc.get_primitive_structure()
                    # recheck nat
                    if not self._check_nat(tmp_struc):  # failure
                        continue

                # -- scale volume
                if self.vol_mu is not None:
                    vol = random.gauss(mu=self.vol_mu, sigma=self.vol_sigma)
                    tmp_struc.scale_lattice(volume=vol)

                # -- check actual space group
                try:
                    spg_sym, spg_num = tmp_struc.get_space_group_info(
                        symprec=self.symprec
                    )
                except TypeError:
                    spg_num = 0
                    spg_sym = None
                # -- register the structure in pymatgen format
                cid = len(init_struc_data) + id_offset
                init_struc_data[cid] = tmp_struc
                logger.info(
                    f"Structure ID {cid:>6} was generated."
                    f" Space group: {spg:>3} --> {spg_num:>3} {spg_sym}"
                )

        return [
            AseAtomsAdaptor.get_atoms(structure)
            for structure in init_struc_data.values()
        ]

    def _set_tol_mat(self, atype, mindist):
        tolmat = Tol_matrix()
        for i, itype in enumerate(atype):
            for j, jtype in enumerate(atype):
                if i <= j:
                    tolmat.set_tol(itype, jtype, mindist[i][j])
        return tolmat

    def _check_nat(self, struc):
        # ---------- count number of atoms in each element for check
        species_list = [a.species_string for a in struc]
        for i in range(len(self.atype)):
            if species_list.count(self.atype[i]) != self.nat[i]:
                return False  # failure
        return True


class Rnd_struc_gen:
    """
    Random structure generation w/o pyxtal

    Note:
    The implementation of this function is taken from the GitHub repository:
    https://github.com/Tomoki-YAMASHITA/CrySPY
    """

    def __init__(
        self,
        mindist,
        atype,
        dangle,
        maxlen,
        minlen,
        nat,
        natot,
        vol_mu=None,
        vol_sigma=None,
        symprec=0.01,
        maxcnt=50,
        spgnum=0,
    ):
        self.mindist = mindist
        self.atype = atype
        self.dangle = dangle
        self.maxlen = maxlen
        self.mindist = mindist
        self.minlen = minlen
        self.nat = nat
        self.natot = natot
        self.vol_mu = vol_mu
        self.vol_sigma = vol_sigma
        self.symprec = symprec
        self.maxcnt = maxcnt
        self.spgnum = spgnum

    def gen_struc(self, nstruc, id_offset=0, init_pos_path=None):
        """
        Generate random structures without space group information

        # ---------- args
        nstruc (int): number of generated structures

        id_offset (int): structure ID starts from id_offset
                         e.g. nstruc = 3, id_offset = 10
                              you obtain ID 10, ID 11, ID 12

        init_pos_path (str): specify a path of file,
                             if you write POSCAR data of init_struc_data
                             ATTENSION: data are appended to the specified file

        # ---------- comment
        generated init_struc_data is saved in self.init_struc_data
        """
        # ---------- initialize
        init_struc_data = {}
        self._get_atomlist()  # get self.atomlist
        # ---------- generate structures
        while len(init_struc_data) < nstruc:
            # ------ get spg, a, b, c, alpha, beta, gamma in self.*
            self._gen_lattice()
            # ------ get va, vb, and vc in self.*
            self._calc_latvec()
            # ------ get structure
            tmp_struc = self._gen_struc_wo_spg()
            if tmp_struc is not None:  # success of generation
                # ------ scale volume
                if self.vol_mu is not None:
                    vol = random.gauss(mu=self.vol_mu, sigma=self.vol_sigma)
                    tmp_struc.scale_lattice(volume=vol)
                    success, mindist_ij, dist = check_distance(
                        tmp_struc, self.atype, self.mindist
                    )
                    if not success:
                        type0 = self.atype[mindist_ij[0]]
                        type1 = self.atype[mindist_ij[1]]
                        logger.warning(
                            f"mindist in gen_wo_spg: {type0} - {type1}, {dist}. retry."
                        )
                        continue  # failure
                # ------ check actual space group using pymatgen
                try:
                    spg_sym, spg_num = tmp_struc.get_space_group_info(
                        symprec=self.symprec
                    )
                except TypeError:
                    spg_num = 0
                    spg_sym = None
                # ------ register the structure in pymatgen format
                cid = len(init_struc_data) + id_offset
                init_struc_data[cid] = tmp_struc
                logger.info(
                    f"Structure ID {cid:>6} was generated."
                    f" Space group: {spg_num:>3} {spg_sym}"
                )

        return [
            AseAtomsAdaptor.get_atoms(structure)
            for structure in init_struc_data.values()
        ]

    def _get_atomlist(self):
        """
        e.g. Na2Cl2
            atomlist = ['Na', 'Na', 'Cl', 'Cl']
        """
        atomlist = []
        for i in range(len(self.atype)):
            atomlist += [self.atype[i]] * self.nat[i]
        self.atomlist = atomlist

    def _gen_lattice(self):
        # ---------- for spgnum = 0: no space group
        if self.spgnum == 0:
            crystal_systems = [
                "Triclinic",
                "Monoclinic",
                "Orthorhombic",
                "Tetragonal",
                "Rhombohedral",
                "Hexagonal",
                "Cubic",
            ]
            spg = 0
            csys = random.choice(crystal_systems)
        # ---------- for spgnum 1--230
        else:
            # ------ spgnum --> spg
            if self.spgnum == "all":
                spg = random.randint(1, 230)
            else:
                spg = random.choice(self.spgnum)
            if 1 <= spg <= 2:
                csys = "Triclinic"
            elif 3 <= spg <= 15:
                csys = "Monoclinic"
            elif 16 <= spg <= 74:
                csys = "Orthorhombic"
            elif 75 <= spg <= 142:
                csys = "Tetragonal"
            elif 143 <= spg <= 167:
                # trigonal includes rhombohedral in find_wy
                csys = "Trigonal"
            elif 168 <= spg <= 194:
                csys = "Hexagonal"
            elif 195 <= spg <= 230:
                csys = "Cubic"
            else:
                logger.error("spg is wrong")
                raise SystemExit(1)
        # ---------- generate lattice constants a, b, c, alpha, beta, gamma
        if csys == "Triclinic":
            t1 = random.uniform(self.minlen, self.maxlen)
            t2 = random.uniform(self.minlen, self.maxlen)
            t3 = random.uniform(self.minlen, self.maxlen)
            t = [t1, t2, t3]
            t.sort()
            a, b, c = t
            r = random.random()
            if r < 0.5:  # Type I
                alpha = 90.0 - random.uniform(0, self.dangle)
                beta = 90.0 - random.uniform(0, self.dangle)
                gamma = 90.0 - random.uniform(0, self.dangle)
            else:  # Type II
                alpha = 90.0 + random.uniform(0, self.dangle)
                beta = 90.0 + random.uniform(0, self.dangle)
                gamma = 90.0 + random.uniform(0, self.dangle)
        elif csys == "Monoclinic":
            a = random.uniform(self.minlen, self.maxlen)
            b = random.uniform(self.minlen, self.maxlen)
            c = random.uniform(self.minlen, self.maxlen)
            if a > c:
                a, c = c, a
            alpha = gamma = 90.0
            beta = 90.0 + random.uniform(0, self.dangle)
        elif csys == "Orthorhombic":
            t1 = random.uniform(self.minlen, self.maxlen)
            t2 = random.uniform(self.minlen, self.maxlen)
            t3 = random.uniform(self.minlen, self.maxlen)
            t = [t1, t2, t3]
            t.sort()
            a, b, c = t
            alpha = beta = gamma = 90.0
        elif csys == "Tetragonal":
            a = b = random.uniform(self.minlen, self.maxlen)
            c = random.uniform(self.minlen, self.maxlen)
            alpha = beta = gamma = 90.0
        elif csys == "Trigonal":
            a = b = random.uniform(self.minlen, self.maxlen)
            c = random.uniform(self.minlen, self.maxlen)
            alpha = beta = 90.0
            gamma = 120.0
        elif csys == "Rhombohedral":
            a = b = c = random.uniform(self.minlen, self.maxlen)
            alpha = beta = gamma = 90 + random.uniform(-self.dangle, self.dangle)
        elif csys == "Hexagonal":
            a = b = random.uniform(self.minlen, self.maxlen)
            c = random.uniform(self.minlen, self.maxlen)
            alpha = beta = 90.0
            gamma = 120.0
        elif csys == "Cubic":
            a = b = c = random.uniform(self.minlen, self.maxlen)
            alpha = beta = gamma = 90.0
        self.spg = spg
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _calc_latvec(self):
        # ---------- degree to radian
        alpha_rad = math.radians(self.alpha)
        beta_rad = math.radians(self.beta)
        gamma_rad = math.radians(self.gamma)
        # ---------- calculate components
        bx = self.b * math.cos(gamma_rad)
        by = self.b * math.sin(gamma_rad)
        cx = self.c * math.cos(beta_rad)
        cy = (self.c * math.cos(alpha_rad) - cx * math.cos(gamma_rad)) / math.sin(
            gamma_rad
        )
        cz = math.sqrt(self.c * self.c - cx * cx - cy * cy)
        # ---------- lattice vector as list
        self.va = [self.a, 0.0, 0.0]
        self.vb = [bx, by, 0.0]
        self.vc = [cx, cy, cz]

    def _calc_cos(self):
        # ---------- degree to radian
        a_rad = math.radians(self.alpha)
        b_rad = math.radians(self.beta)
        g_rad = math.radians(self.gamma)
        self.cosa = math.cos(a_rad)
        self.cosb = math.cos(b_rad)
        self.cosg = math.cos(g_rad)

    def _gen_struc_wo_spg(self):
        """
        Success --> return structure data in pymatgen format
        Failure --> return None
        """
        # ---------- initialize
        cnt = 0
        incoord = []
        # ---------- generate internal coordinates
        while len(incoord) < self.natot:
            tmp_coord = np.random.rand(3)
            incoord.append(tmp_coord)
            tmp_struc = Structure(
                [self.va, self.vb, self.vc], self.atomlist[: len(incoord)], incoord
            )
            success, mindist_ij, dist = check_distance(
                tmp_struc, self.atype, self.mindist
            )
            if not success:
                type0 = self.atype[mindist_ij[0]]
                type1 = self.atype[mindist_ij[1]]
                logger.warning(
                    f"mindist in _gen_struc_wo_spg: {type0} - {type1}, {dist}. retry."
                )
                incoord.pop(-1)  # cancel
                cnt += 1
                if self.maxcnt < cnt:
                    return None
        return tmp_struc

    def _fw_input(self):
        with open("input", "w") as f:
            f.write("nspecies {}\n".format(len(self.atype)))
            f.write("species_name")
            for aa in self.atype:
                f.write("  {}".format(aa))
            f.write("\n")
            f.write("species_num")
            for i in self.nat:
                f.write("  {}".format(i))
            f.write("\n")
            f.write("spacegroup  {}\n".format(self.spg))
            f.write("originchoice  1\n")
            f.write("\n")
            f.write("a  {}\n".format(self.a))
            f.write("b  {}\n".format(self.b))
            f.write("c  {}\n".format(self.c))
            f.write("cosa  {}\n".format(self.cosa))
            f.write("cosb  {}\n".format(self.cosb))
            f.write("cosc  {}\n".format(self.cosg))
            f.write("\n")
            # f.write('selectone true\n')
            f.write("randomseed auto\n")

    def _gen_struc_with_spg(self):
        """
        Success --> return True, structure data
        Failure --> return False, _
        """
        # ---------- load POS_WY_SKEL_ALL.json
        with open("POS_WY_SKEL_ALL.json", "r") as f:
            wydata = json.load(f)
        # ---------- generate structure
        plat = wydata["primitivevector"]
        clat = wydata["conventionalvector"]
        n_uniq, wydata_eq_atom = self._get_wydata_eq_atom(wydata)
        eq_atomnames = {}
        eq_positions = {}
        # ---------- equivalent atom loop
        for key, value in sorted(n_uniq.items(), key=lambda x: x[1]):
            # ------ distribute eq atoms. first, special (num_uniqvar = 0),
            #            then, others
            cnt = 0
            while True:
                eq_atomnames[key], eq_positions[key] = self._gen_eq_atoms(
                    wydata_eq_atom[key]
                )
                # -- sort in original order
                atomnames = []
                positions = []
                for key_a, value_a in sorted(eq_atomnames.items()):
                    atomnames += eq_atomnames[key_a]
                    positions += eq_positions[key_a]
                # -- Cartesian coordinate; use clat (not plat)
                cart = []
                for p in positions:
                    v = np.zeros(3)
                    for i in range(3):
                        a = np.array(clat[i])
                        v += p[i] * a
                    cart.append(v)
                # -- check minimum distance
                spgstruc = Structure(plat, atomnames, cart, coords_are_cartesian=True)
                success, mindist_ij, dist = check_distance(
                    spgstruc, self.atype, self.mindist
                )
                if not success:
                    type0 = self.atype[mindist_ij[0]]
                    type1 = self.atype[mindist_ij[1]]
                    logger.warning(
                        f"mindist in _gen_struc_with_spg: {type0} - {type1}, {dist}. retry."
                    )
                    # failure
                    # num_uniqvar = 0 --> value == 0
                    cnt = self.maxcnt + 1 if value == 0 else cnt + 1
                    if self.maxcnt < cnt:
                        return False, spgstruc  # spgstruc is dummy
                else:
                    break  # break while loop --> next eq atoms
        return True, spgstruc

    def _get_wydata_eq_atom(self, wydata):
        i = 0  # count eq_atom, not atom
        n_uniq = {}  # num_uniqvar each eq_atom
        wydata_eq_atom = {}  # wydata each eq_atom
        for specie in wydata["atoms"]:
            for wydata2 in specie:  # equivalent atom loop
                n_uniq[i] = wydata2[0]["num_uniqvar"]
                wydata_eq_atom[i] = wydata2
                i += 1
        return n_uniq, wydata_eq_atom

    def _gen_eq_atoms(self, wydata2):
        eq_atomnames = []
        eq_positions = []
        rval = np.random.random_sample(3)
        for each in wydata2:
            pos = []
            for ch in each["xyzch"]:
                if ch == "-2x":
                    pos.append(-2.0 * rval[0])
                elif ch == "-x+y":
                    pos.append(-rval[0] + rval[1])
                elif ch == "-z":
                    pos.append(-rval[2])
                elif ch == "-y":
                    pos.append(-rval[1])
                elif ch == "-x":
                    pos.append(-rval[0])
                elif ch == "0":
                    pos.append(0.0)
                elif ch == "x":
                    pos.append(rval[0])
                elif ch == "y":
                    pos.append(rval[1])
                elif ch == "z":
                    pos.append(rval[2])
                elif ch == "x-y":
                    pos.append(rval[0] - rval[1])
                elif ch == "2x":
                    pos.append(2.0 * rval[0])
                else:
                    logger.error("unknown ch in conversion in gen_wycoord")
                    raise SystemExit(1)
            pos = np.array(pos)
            eq_positions.append(pos + each["add"])
            eq_atomnames.append(each["name"])
        return eq_atomnames, eq_positions

    def _rm_files(self, files=["input", "POS_WY_SKEL_ALL.json"]):
        for rfile in files:
            if os.path.isfile(rfile):
                os.remove(rfile)


def check_distance(struc, atype, mindist, check_all=False):
    """
    # ---------- args
    struc: structure data in pymatgen format
    atype (list): e.g. ['Li', 'Co, 'O']
    mindist (2d list) : e.g. [[2.0, 2.0, 1.2],
                              [2.0, 2.0, 1.2],
                              [1.2, 1.2, 1.5]]
    check_all (bool) : if True, check all atom pairs, return dist_list.
                       if False, stop when (dist < mindist) is found,
                                 return True or False (see below)
    # ---------- return
    (check_all=False) True, None, None: nothing smaller than mindist
    (check_all=False) False, (i, j), dist: something smaller than mindst
                                     here, (i, j) means mindist(i, j)
    (check_all=True) dist_list: if dist_list is vacant,
                                    nothing smaller than mindist
    """

    # ---------- initialize
    if check_all:
        dist_list = []  # [(i, j, dist), (i, j, dist), ...]

    # ---------- in case there is only one atom
    if struc.num_sites == 1:
        dist = min(struc.lattice.abc)
        if dist < mindist[0][0]:
            if check_all:
                dist_list.append((0, 0, dist))
                return dist_list
            return False, (0, 0), dist
        return True, None, None

    # ---------- normal case
    for i in range(struc.num_sites):
        for j in range(i):
            dist = struc.get_distance(j, i)
            i_type = atype.index(struc[i].species_string)
            j_type = atype.index(struc[j].species_string)
            if dist < mindist[i_type][j_type]:
                if check_all:
                    dist_list.append((j, i, dist))
                else:
                    return False, (i_type, j_type), dist

    # ---------- return
    if check_all:
        if dist_list:
            dist_list.sort()  # sort
            return dist_list
        else:
            return dist_list  # dist_list is vacant list
    return True, None, None


if __name__ == "__main__":
    from ase.visualize import view

    generator = Rnd_struc_gen_pyxtal(
        mindist=[[1.0]],
        atype=["Si"],
        dangle=30,
        maxlen=4,
        minlen=2,
        nat=[2],
        natot=2,
        vol_mu=None,
        vol_sigma=None,
        vol_factor=[1.0, 1.1],
        symprec=0.01,
        maxcnt=50,
        spgnum="all",
        mol_file=None,
        algo=None,
    )
    atoms = generator.gen_struc(10)
    view(atoms)
