import tencirchem as tcc
from tencirchem.static.ucc import *
from functools import partial
from itertools import product
from collections import defaultdict
from time import time
import logging
from typing import Any, Tuple, Callable, List, Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import comb
import pandas as pd
from openfermion import jordan_wigner, FermionOperator, QubitOperator
from pyscf.gto.mole import Mole
from pyscf.scf import RHF
from pyscf.scf import ROHF
from pyscf.scf.hf import RHF as RHF_TYPE
from pyscf.scf.rohf import ROHF as ROHF_TYPE
from pyscf.cc.addons import spatial2spin
from pyscf.mcscf import CASCI
from pyscf import fci
import tensorcircuit as tc

from tencirchem.constants import DISCARD_EPS
from tencirchem.molecule import _Molecule
from tencirchem.utils.misc import reverse_qop_idx, scipy_opt_wrap, rdm_mo2ao, canonical_mo_coeff
from tencirchem.utils.circuit import get_circuit_dataframe
from tencirchem.static.engine_ucc import (
    get_civector,
    get_statevector,
    apply_excitation,
    translate_init_state,
)
from tencirchem.static.hamiltonian import (
    get_integral_from_hf,
    get_h_from_integral,
    get_hop_from_integral,
    get_hop_hcb_from_integral,
)
from tencirchem.static.ci_utils import get_ci_strings, get_ex_bitstring, get_addr, get_init_civector
from tencirchem.static.evolve_tensornetwork import get_circuit
from .engine_expval import get_energy,get_energy_and_grad


class EXPVAL(UCC):
    def __init__(self,mol: Union[Mole, RHF],init_method="mp2",active_space=None,mo_coeff=None,hcb=False,engine=None,run_hf=True,run_mp2=True,run_ccsd=True,run_fci=True):
        r"""
        Initialize the class with molecular input.

        Parameters
        ----------
        mol: Mole or RHF
            The molecule as PySCF ``Mole`` object or the PySCF ``RHF`` object
        init_method: str, optional
            How to determine the initial amplitude guess. Accepts ``"mp2"`` (default), ``"ccsd"``, ``"fe"``
            and ``"zeros"``.
        active_space: Tuple[int, int], optional
            Active space approximation. The first integer is the number of electrons and the second integer is
            the number or spatial-orbitals. Defaults to None.
        mo_coeff: np.ndarray, optional
            Molecule coefficients. If provided then RHF is skipped.
            Can be used in combination with the ``init_state`` attribute.
            Defaults to None which means RHF orbitals are used.
        hcb: bool, optional
            Whether force electrons to pair as hard-core boson (HCB). Default to False.
        engine: str, optional
            The engine to run the calculation. See :ref:`advanced:Engines` for details.
        run_hf: bool, optional
            Whether run HF for molecule orbitals. Defaults to ``True``.
            The argument has no effect if ``mol`` is a ``RHF`` object.
        run_mp2: bool, optional
            Whether run MP2 for initial guess and energy reference. Defaults to ``True``.
        run_ccsd: bool, optional
            Whether run CCSD for initial guess and energy reference. Defaults to ``True``.
        run_fci: bool, optional
            Whether run FCI  for energy reference. Defaults to ``True``.

        See Also
        --------
        tencirchem.UCCSD
        tencirchem.KUPCCGSD
        tencirchem.PUCCD
        """
        # process mol
        if isinstance(mol, _Molecule):
            self.mol = mol
            self.mol.verbose = 0
            self.hf: RHF = None
        elif isinstance(mol, Mole):
            # to set verbose = 0
            self.mol = mol.copy()
            # be cautious when modifying mol. Custom mols are common in practice
            self.mol.verbose = 0
            self.hf: RHF = None
        elif isinstance(mol, RHF_TYPE):
            self.hf: RHF = mol
            self.mol = self.hf.mol
            mol = self.mol
        else:
            raise TypeError(
                f"Unknown input type {type(mol)}. If you're performing open shell calculations, "
                "please use ROHF instead."
            )

        if active_space is None:
            active_space = (mol.nelectron, int(mol.nao))

        self.hcb = hcb
        self.spin = self.mol.spin
        if hcb:
            assert self.spin == 0
        self.n_qubits = 2 * active_space[1]
        if hcb:
            self.n_qubits //= 2

        # process activate space
        self.active_space = active_space
        self.n_elec = active_space[0]
        self.active = active_space[1]
        self.inactive_occ = (mol.nelectron - active_space[0]) // 2
        assert (mol.nelectron - active_space[0]) % 2 == 0
        self.inactive_vir = mol.nao - active_space[1] - self.inactive_occ
        frozen_idx = list(range(self.inactive_occ)) + list(range(mol.nao - self.inactive_vir, mol.nao))

        # process backend
        self._check_engine(engine)

        if engine is None:
            # no need to be too precise
            if self.n_qubits <= 16:
                engine = "civector"
            else:
                engine = "civector-large"
        self.engine = engine

        # classical quantum chemistry
        # hf
        if self.hf is not None:
            self.e_hf = self.hf.e_tot
            self.hf.mo_coeff = canonical_mo_coeff(self.hf.mo_coeff)
        elif run_hf:
            if self.spin == 0:
                self.hf = RHF(self.mol)
            else:
                self.hf = ROHF(self.mol)
            # avoid serialization warnings for `_Molecule`
            self.hf.chkfile = None
            # run this even when ``mo_coeff is not None`` because MP2 and CCSD
            # reference energy might be desired
            self.e_hf = self.hf.kernel(dump_chk=False)
            self.hf.mo_coeff = canonical_mo_coeff(self.hf.mo_coeff)
        else:
            self.e_hf = None
            # otherwise, can't run casci.get_h2eff() based on HF
            self.hf = RHF(self.mol)
            self.hf._eri = mol.intor("int2e", aosym="s8")
            if mo_coeff is None:
                raise ValueError("Must provide MO coefficient if HF is skipped")

        # mp2
        if run_mp2 and not isinstance(self.hf, ROHF_TYPE):
            mp2 = self.hf.MP2()
            if frozen_idx:
                mp2.frozen = frozen_idx
            e_corr_mp2, mp2_t2 = mp2.kernel()
            self.e_mp2 = self.e_hf + e_corr_mp2
        else:
            self.e_mp2 = None
            mp2_t2 = None
            if init_method is not None and init_method.lower() == "mp2":
                raise ValueError("Must run RHF and MP2 to use MP2 as the initial guess method")

        # ccsd
        if run_ccsd and not isinstance(self.hf, ROHF_TYPE):
            ccsd = self.hf.CCSD()
            if frozen_idx:
                ccsd.frozen = frozen_idx
            e_corr_ccsd, ccsd_t1, ccsd_t2 = ccsd.kernel()
            self.e_ccsd = self.e_hf + e_corr_ccsd
        else:
            self.e_ccsd = None
            ccsd_t1 = ccsd_t2 = None
            if init_method is not None and init_method.lower() == "ccsd":
                raise ValueError("Must run CCSD to use CCSD as the initial guess method")

        # MP2 and CCSD rely on canonical HF orbitals but FCI doesn't
        # so set custom mo_coeff after MP2 and CCSD and before FCI
        if mo_coeff is not None:
            # use user defined coefficient
            self.hf.mo_coeff = canonical_mo_coeff(mo_coeff)

        # fci
        if run_fci:
            fci = CASCI(self.hf, self.active_space[1], self.active_space[0])
            fci.max_memory = 32000
            res = fci.kernel()
            self.e_fci = res[0]
            self.civector_fci = res[2].ravel()
        else:
            self.e_fci = None
            self.civector_fci = None

        self.e_nuc = mol.energy_nuc()

        # Hamiltonian related
        self.hamiltonian_lib = {}
        self.int1e = self.int2e = None
        # e_core includes nuclear repulsion energy
        self.hamiltonian, self.e_core, _ = self._get_hamiltonian_and_core(self.engine)

        # initial guess
        self.t1 = np.zeros([self.no, self.nv])
        self.t2 = np.zeros([self.no, self.no, self.nv, self.nv])
        self.init_method = init_method
        if init_method is None or init_method in ["zeros", "zero"]:
            pass
        elif init_method.lower() == "ccsd":
            self.t1, self.t2 = ccsd_t1, ccsd_t2
        elif init_method.lower() == "fe":
            self.t2 = compute_fe_t2(self.no, self.nv, self.int1e, self.int2e)
        elif init_method.lower() == "mp2":
            self.t2 = mp2_t2
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        # circuit related
        self._init_state_bra = None
        self._init_state_ket = None
        self._ex_ops_bra = None
        self._ex_ops_ket = None
        self._param_ids_bra = None
        self._param_ids_ket = None
        self.init_guess = None
        del self._init_state
        del self._ex_ops
        del self._param_ids
        del self._params
        # optimization related
        self.scipy_minimize_options = None
        # optimization result
        self.opt_res = None
        # for manually set
        self._params_bra = None
        self._params_ket = None
    def energy(self, params_bra: Tensor = None,params_ket: Tensor = None, engine: str = None) -> float:
        """
        Evaluate the total energy.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        energy: float
            Total energy

        See Also
        --------
        civector: Get the configuration interaction (CI) vector.
        statevector: Evaluate the circuit state vector.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> round(uccsd.energy([0, 0]), 8)  # HF state
        -1.11670614
        """
        self._sanity_check()
        params_bra = self._check_params_argument(params_bra)
        params_ket = self._check_params_argument(params_ket)
        if params_bra is self.params_bra and self.opt_res is not None:
            return self.opt_res.e
        hamiltonian, _, engine = self._get_hamiltonian_and_core(engine)
        e = get_energy(hamiltonian, self.n_qubits, self.n_elec_s,self.hcb,params_bra,params_ket, self.ex_ops_bra,self.ex_ops_ket, self.param_ids_bra,self.param_ids_ket,  self.init_state_bra,self.init_state_ket, engine)
        return float(e) + self.e_core
    def energy_and_grad(self, params_bra: Tensor = None,params_ket: Tensor = None, engine: str = None) -> Tuple[float, Tensor]:
        """
        Evaluate the total energy and parameter gradients.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        energy: float
            Total energy
        grad: Tensor
            The parameter gradients

        See Also
        --------
        civector: Get the configuration interaction (CI) vector.
        statevector: Evaluate the circuit state vector.
        energy: Evaluate the total energy.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> e, g = uccsd.energy_and_grad([0, 0])
        >>> round(e, 8)
        -1.11670614
        >>> g  # doctest:+ELLIPSIS
        array([..., ...])
        """
        self._sanity_check()
        params_bra = self._check_params_argument(params_bra)
        params_ket = self._check_params_argument(params_ket)
        hamiltonian, _, engine = self._get_hamiltonian_and_core(engine)
        e, g = get_energy_and_grad(hamiltonian, self.n_qubits, self.n_elec_s,self.hcb,params_bra,params_ket, self.ex_ops_bra,self.ex_ops_ket, self.param_ids_bra,self.param_ids_ket,  self.init_state_bra,self.init_state_ket, engine,) 
        return float(e + self.e_core), tc.backend.numpy(g)

    def _check_params_argument(self, params, strict=True):
        if params is None:
            if self.params is not None:
                params = self.params_ket + self.params_bra
            else:
                if strict:
                    raise ValueError("Run the `.kernel` method to determine the parameters first")
                else:
                    if self.init_guess is not None:
                        params = self.init_guess
                    else:
                        params = np.zeros(self.n_params)

        if len(params) != self.n_params:
            raise ValueError(f"Incompatible parameter shape. {self.n_params} is desired. Got {len(params)}")
        return tc.backend.convert_to_tensor(params).astype(tc.rdtypestr)   

    def _sanity_check(self):
        if self.ex_ops_ket is None or self.param_ids_ket is None:
            raise ValueError("`ex_ops_ket` or `param_ids_ket` not defined")
        if self.ex_ops_bra is None or self.param_ids_bra is None:
            raise ValueError("`ex_ops_bra` or `param_ids_bra` not defined")
        
        if self.param_ids_bra is not None and (len(self.ex_ops_bra) != len(self.param_ids_bra)):
            raise ValueError(
                f"Excitation operator size {len(self.ex_ops_bra)} and parameter size {len(self.param_ids_bra)} do not match"
            )
        if self.param_ids_ket is not None and (len(self.ex_ops_ket) != len(self.param_ids_ket)):
            raise ValueError(
                f"Excitation operator size {len(self.ex_ops_ket)} and parameter size {len(self.param_ids_ket)} do not match"
            )
    
    def civector(self, params: Tensor = None, engine: str = None,ket: bool = True) -> Tensor:
        """
        Evaluate the configuration interaction (CI) vector.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        civector: Tensor
            Corresponding CI vector

        See Also
        --------
        statevector: Evaluate the circuit state vector.
        energy: Evaluate the total energy.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.civector([0, 0])  # HF state
        array([1., 0., 0., 0.])
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        if ket:
            ex_ops  = self.ex_ops_ket
            param_ids = self.param_ids_ket
            init_state = self.init_state_ket
        else:
            ex_ops  = self.ex_ops_bra
            param_ids = self.param_ids_bra
            init_state = self.init_state_bra
        civector = get_civector(
            params, self.n_qubits, self.n_elec_s, ex_ops, param_ids, self.hcb, init_state, engine
        )
        return civector

    def _statevector_to_civector(self, statevector=None,ket: bool = True):
        if statevector is None:
            civector = self.civector(ket=ket)
        else:
            if len(statevector) == self.statevector_size:
                ci_strings = self.get_ci_strings()
                civector = statevector[ci_strings]
            else:
                if len(statevector) == self.civector_size:
                    civector = statevector
                else:
                    raise ValueError(f"Incompatible statevector size: {len(statevector)}")

        civector = tc.backend.numpy(tc.backend.convert_to_tensor(civector))
        return civector
    # since there's ci_vector method
    ci_strings = get_ci_strings

    def statevector(self, params: Tensor = None, engine: str = None, ket: bool = True) -> Tensor:
        """
        Evaluate the circuit state vector.

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter
            and :func:`kernel` must be called before.
        engine: str, optional
            The engine to use. Defaults to ``None``, which uses ``self.engine``.

        Returns
        -------
        statevector: Tensor
            Corresponding state vector

        See Also
        --------
        civector: Evaluate the configuration interaction (CI) vector.
        energy: Evaluate the total energy.
        energy_and_grad: Evaluate the total energy and parameter gradients.

        Examples
        --------
        >>> from tencirchem import UCCSD
        >>> from tencirchem.molecule import h2
        >>> uccsd = UCCSD(h2)
        >>> uccsd.statevector([0, 0])  # HF state
        array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        """
        self._sanity_check()
        params = self._check_params_argument(params)
        self._check_engine(engine)
        if engine is None:
            engine = self.engine
        if ket:
            ex_ops = self.ex_ops_ket
            param_ids = self.param_ids_ket
            init_state = self.init_state_ket
        else:
            ex_ops = self.ex_ops_bra
            param_ids = self.param_ids_bra
            init_state = self.init_state_bra
        statevector = get_statevector(
            params, self.n_qubits, self.n_elec_s, ex_ops, param_ids, self.hcb, init_state, engine
        )
        return statevector

    def get_circuit(self, params: Tensor = None, decompose_multicontrol: bool = False, trotter: bool = False, ket: bool = True) -> tc.Circuit:
        """
        Get the circuit as TensorCircuit ``Circuit`` object

        Parameters
        ----------
        params: Tensor, optional
            The circuit parameters. Defaults to None, which uses the optimized parameter.
            If :func:`kernel` is not called before, the initial guess is used.
        decompose_multicontrol: bool, optional
            Whether decompose the Multicontrol gate in the circuit into CNOT gates.
            Defaults to False.
        trotter: bool, optional
            Whether Trotterize the UCC factor into Pauli strings.
            Defaults to False.
        ket: bool, optional
            Whether to return ket or bra circuit
        Returns
        -------
        circuit: :class:`tc.Circuit`
            The quantum circuit.
        """
        if ket:
            ex_ops = self.ex_ops_ket
            param_ids = self.param_ids_ket
            init_state = self.init_state_ket
        else:
            ex_ops = self.ex_ops_bra
            param_ids = self.param_ids_bra
            init_state = self.init_state_bra
        if ex_ops is None:
            raise ValueError("Excitation operators not defined")
        params = self._check_params_argument(params, strict=False)
        return get_circuit(params,self.n_qubits,self.n_elec_s,ex_ops,param_ids,self.hcb,init_state,decompose_multicontrol=decompose_multicontrol,trotter=trotter)

    def print_circuit(self,ket:bool=True):
        """
        Prints the circuit information. If you wish to print the circuit diagram,
        use :func:`get_circuit` and then call ``draw()`` such as ``print(ucc.get_circuit().draw())``.
        """
        c = self.get_circuit(ket=ket)
        df = get_circuit_dataframe(c)

        def format_flop(f):
            return f"{f:.3e}"

        formatters = {"flop": format_flop}
        print(df.to_string(index=False, formatters=formatters))

    def get_excitation_dataframe(self,ket:bool=True) -> pd.DataFrame:
        '''
        ket:bool Whether to refered to the ket (True) or bra (False)
        '''
        columns = ["excitation", "configuration", "parameter", "initial guess"]
        if ket:
            if self.ex_ops_ket is None:
                return pd.DataFrame(columns=columns)

            if self.params_ket is None:
                # optimization not done
                params = [None] * len(self.init_guess)
            else:
                params = self.params_ket

            if self.param_ids is None:
                # see self.n_params
                param_ids = range(len(self.ex_ops_ket))
            else:
                param_ids = self.param_ids_ket
            ex_ops =self.ex_ops_ket
        else:
            if self.ex_ops_bra is None:
                return pd.DataFrame(columns=columns)

            if self.params_bra is None:
                # optimization not done
                params = [None] * len(self.init_guess)
            else:
                params = self.params_bra

            if self.param_ids is None:
                # see self.n_params
                param_ids = range(len(self.ex_ops_bra))
            else:
                param_ids = self.param_ids_bra
            ex_ops =self.ex_ops_bra
        
        data_list = []

        for i, ex_op in zip(param_ids, ex_ops):
            bitstring = get_ex_bitstring(self.n_qubits, self.n_elec_s, ex_op, self.hcb)
            data_list.append((ex_op, bitstring, params[i], self.init_guess[i]))
        return pd.DataFrame(data_list, columns=columns)

    def print_excitations(self):
        print('Bra: ',self.get_excitation_dataframe(ket=False).to_string())
        print('Ket: ',self.get_excitation_dataframe(ket=True).to_string())

    def print_ansatz(self):
        df_dict = {
            "#qubits": [self.n_qubits],
            "#params": [self.n_params],
            "#excitations": [len(self.ex_ops)],
        }
        if self.init_state_bra is None:
            df_dict["initial condition bra"] = "RHF"
        else:
            df_dict["initial condition bra"] = "custom"

        if self.init_state_ket is None:
            df_dict["initial condition ket"] = "RHF"
        else:
            df_dict["initial condition ket"] = "custom"
        print(pd.DataFrame(df_dict).to_string(index=False))

    def get_init_state_dataframe(self,init_state=None, coeff_epsilon: float = DISCARD_EPS) -> pd.DataFrame:
        """
        Returns initial state information dataframe.

        Parameters
        ----------
        coeff_epsilon: float, optional
            The threshold to screen out states with small coefficients.
            Defaults to 1e-12.

        Returns
        -------
        pd.DataFrame

        See Also
        --------
        init_state: The circuit initial state before applying the excitation operators.

        Examples
        --------
        >>> from tencirchem import UCC
        >>> from tencirchem.molecule import h2
        >>> ucc = UCC(h2)
        >>> ucc.init_state = [0.707, 0, 0, 0.707]
        >>> ucc.get_init_state_dataframe()   # doctest: +NORMALIZE_WHITESPACE
             configuration  coefficient
        0          0101        0.707
        1          1010        0.707
        """
        columns = ["configuration", "coefficient"]
        if init_state is None:
            init_state = get_init_civector(self.civector_size)
        ci_strings = self.get_ci_strings()
        ci_coeffs = translate_init_state(init_state, self.n_qubits, ci_strings)
        data_list = []
        for ci_string, coeff in zip(ci_strings, ci_coeffs):
            if np.abs(coeff) < coeff_epsilon:
                continue
            ci_string = bin(ci_string)[2:]
            ci_string = "0" * (self.n_qubits - len(ci_string)) + ci_string
            data_list.append((ci_string, coeff))
        return pd.DataFrame(data_list, columns=columns)

    def print_init_state_bra(self):
        print('Bra: ',self.get_init_state_dataframe(self.init_state_bra).to_string())
    
    def print_init_state_ket(self):
        print('Ket: ',self.get_init_state_dataframe(self.init_state_ket).to_string())

    def print_init_state(self):
        self.print_init_state_bra()
        self.print_init_state_ket()
    
    def print_summary(self, include_circuit: bool = False):
        """
        Print a summary of the class.

        Parameters
        ----------
        include_circuit: bool
            Whether include the circuit section.

        """
        print("################################ Ansatz ###############################")
        self.print_ansatz()
        if self.init_state_bra is not None or self.init_state_ket is not None:
            print("############################ Initial Condition ########################")
            self.print_init_state_bra()
            self.print_init_state_ket()
        if include_circuit:
            print("############################### Circuit ###############################")
            self.print_circuit()
        print("############################### Energy ################################")
        self.print_energy()
        print("############################# Excitations #############################")
        self.print_excitations()
        print("######################### Optimization Result #########################")
        if self.opt_res is None:
            print("Optimization not run (.opt_res is None)")
        else:
            print(self.opt_res)

    @property
    def init_state_bra(self) -> Tensor:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        get_init_state_dataframe: Returns initial state information dataframe.
        """
        if self._init_state_bra is not None:
            return self._init_state_bra
        else:
            return self._init_state_ket

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra):
        self._init_state_bra = init_state_bra

    @property
    def init_state_ket(self) -> Tensor:
        """
        The circuit initial state before applying the excitation operators. Usually RHF.

        See Also
        --------
        get_init_state_dataframe: Returns initial state information dataframe.
        """
        return self._init_state_ket

    @init_state_ket.setter
    def init_state_ket(self, init_state_ket):
        self._init_state_ket = init_state_ket

    @property
    def ex_ops_bra(self) -> Tensor:
        """
        Excitation operators applied to the bra.
        """
        if self._ex_ops_bra is not None:
            return self._ex_ops_bra
        else: return self._ex_ops_ket

    @ex_ops_bra.setter
    def ex_ops_bra(self, ex_ops_bra):
        self._ex_ops_bra = ex_ops_bra
    
    @property
    def ex_ops_ket(self) -> Tensor:
        """
        Excitation operators applied to the ket.
        """
        return self._ex_ops_ket

    @ex_ops_ket.setter
    def ex_ops_ket(self, ex_ops_ket):
        self._ex_ops_ket = ex_ops_ket

    @property
    def params_bra(self) -> Tensor:
        """The circuit parameters."""
        if self._params_bra is not None:
            return self._params_bra
        elif self._params_ket is not None:
            return self._params_ket
        elif self.opt_res is not None:
            return self.opt_res.x
        return None

    @params_bra.setter
    def params_bra(self, params_bra):
        self._params_bra = params_bra
    
    @property
    def params_ket(self) -> Tensor:
        """The circuit parameters."""
        if self._params_ket is not None:
            return self._params_ket
        if self.opt_res is not None:
            return self.opt_res.x
        return None

    @params_ket.setter
    def params_ket(self, params_ket):
        self._params_ket = params_ket

    @property
    def param_ids_bra(self) -> List[int]:
        """The mapping from excitations operators to parameters."""
        if self._param_ids_bra is None:
            if self.ex_ops_bra is None:
                if self._param_ids_ket is not None:
                    return self.param_ids_ket()
                raise ValueError("Excitation operators not defined")
            else:
                return tuple(range(len(self.ex_ops_bra)))
        return self._param_ids_bra

    @param_ids_bra.setter
    def param_ids_bra(self, v):
        self._param_ids_bra = v

    @property
    def param_ids_ket(self) -> List[int]:
        """The mapping from excitations operators to parameters."""
        if self._param_ids_ket is None:
            if self.ex_ops_ket is None:
                raise ValueError("Excitation operators not defined")
            else:
                return tuple(range(len(self.ex_ops_ket)))
        return self._param_ids_bra

    @param_ids_ket.setter
    def param_ids_ket(self, v):
        self._param_ids_ket = v

    @property
    def param_to_ex_ops_bra(self):
        d = defaultdict(list)
        for i, j in enumerate(self.param_ids_bra):
            d[j].append(self.ex_ops_bra[i])
        return d
    
    @property
    def param_to_ex_ops_ket(self):
        d = defaultdict(list)
        for i, j in enumerate(self.param_ids_ket):
            d[j].append(self.ex_ops_ket[i])
        return d
    
    @property
    def n_params(self) -> int:
        """The number of parameter in the ansatz/circuit."""
        # this definition ensures that `param[param_id]` is always valid
        if not self.param_ids_bra and not self.param_ids_ket:
            return 0
        return max(self.param_ids_ket+self.param_ids_bra) + 1