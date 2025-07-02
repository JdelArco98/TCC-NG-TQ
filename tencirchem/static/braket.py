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
    get_energy,get_energy_and_grad,
)
from tencirchem.static.hamiltonian import (
    get_integral_from_hf,
    get_h_from_integral,
    get_hop_from_integral,
    get_hop_hcb_from_integral,
)
from tencirchem.static.ci_utils import get_ci_strings, get_ex_bitstring, get_addr, get_init_civector
from tencirchem.static.evolve_tensornetwork import get_circuit
# from tencirchem.static.engine_braket import get_energy,get_energy_and_grad
from .engine_braket import get_energy,get_energy_and_grad

class EXPVAL(UCC):
    def __init__(
        self,
        mol: Union[Mole, RHF],
        init_method="mp2",
        active_space=None,
        aslst=None,
        mo_coeff=None,
        mode="fermion",
        engine=None,
        run_hf=True,
        run_mp2=True,
        run_ccsd=True,
        run_fci=True,
    ):
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
        aslst: List[int], optional
            Pick orbitals for the active space. Defaults to None which means the orbitals are sorted by energy.
            The orbital index is 0-based.

            .. note::
                See `PySCF document <https://pyscf.org/user/mcscf.html#picking-an-active-space>`_
                for choosing the active space orbitals. Here orbital index is 0-based, whereas in PySCF by default it
                is 1-based.
        mo_coeff: np.ndarray, optional
            Molecule coefficients. If provided then RHF is skipped.
            Can be used in combination with the ``init_state`` attribute.
            Defaults to None which means RHF orbitals are used.
        mode: str, optional
            How to deal with particle symmetry, such as whether force electrons to pair as hard-core boson (HCB).
            Possible values are ``"fermion"``, ``"qubit"`` and ``"hcb"``.
            Default to ``"fermion"``.
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

        """
        super().__init__(mol=mol,
        init_method=init_method,
        active_space=active_space,
        aslst=aslst,
        mo_coeff=mo_coeff,
        mode=mode,
        engine=engine,
        run_hf=run_hf,
        run_mp2=run_mp2,
        run_ccsd=run_ccsd,
        run_fci=run_fci)
        
        # circuit related
        self._init_state_bra = None
        self._init_state  = None
        self._ex_ops_bra = None
        self._ex_ops  = None
        self._param_ids_bra = None
        self._param_ids  = None
        self.init_guess_bra = None
        self.init_guess  = None
        # optimization related
        self.scipy_minimize_options = None
        # optimization result
        self.opt_res = None
        # for manually set
        self._params_bra = None
        self._params  = None
    
    def kernel(self) -> float:
        """
        The kernel to perform the VQE algorithm.
        The L-BFGS-B method in SciPy is used for optimization
        and configuration is possible by setting the ``self.scipy_minimize_options`` attribute.

        Returns
        -------
        e: float
            The optimized energy
        """
        assert len(self.param_ids) == len(self.ex_ops)
        assert len(self.param_ids_bra) == len(self.ex_ops_bra)

        energy_and_grad, stating_time = self.get_opt_function(with_time=True)

        if self.init_guess  is None:
            self.init_guess  = np.zeros(self.n_params )
        if self.init_guess_bra is None:
            self.init_guess_bra = np.zeros(self.n_params_bra)

        # optimization options
        if self.scipy_minimize_options is None:
            # quite strict
            options = {"ftol": 1e1 * np.finfo(tc.rdtypestr).eps, "gtol": 1e2 * np.finfo(tc.rdtypestr).eps}
        else:
            options = self.scipy_minimize_options

        logger.info("Begin optimization")
        time1 = time()
        opt_res = minimize(energy_and_grad, x0=self.init_guess, jac=True, method="L-BFGS-B", options=options)
        time2 = time()

        if not opt_res.success:
            logger.warning("Optimization failed. See `.opt_res` for details.")

        opt_res["staging_time"] = stating_time
        opt_res["opt_time"] = time2 - time1
        opt_res["init_guess"] = self.init_guess +self.init_guess_bra
        opt_res["e"] = float(opt_res.fun)
        self.opt_res = opt_res
        # prepare for future modification
        self.params = opt_res.x.copy() #TODO: split params in bra and ket
        return opt_res.e
    
    def energy(self, params_bra: Tensor = None,params : Tensor = None, engine: str = None) -> float:
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
        params  = self._check_params_argument(params )
        if params_bra is None:
            params_bra = params 
        params_bra = self._check_params_argument(params_bra)
        if (params_bra is self.params_bra or params_bra is self.params ) and params  is self.params  and self.opt_res is not None:
            return self.opt_res.e
        hamiltonian, _, engine = self._get_hamiltonian_and_core(engine)
        e = get_energy(hamiltonian=hamiltonian, n_qubits=self.n_qubits, n_elec_s=self.n_elec_s,engine= engine,mode=self.mode,
                       ex_ops=self.ex_ops , params=self.params ,params_bra=self.params_bra,
                       param_ids=self.param_ids ,init_state=self.init_state , ex_ops_bra=self.ex_ops_bra, 
                       param_ids_bra=self.param_ids_bra,  init_state_bra=self.init_state_bra)
        return float(e) + self.e_core
    
    def energy_and_grad(self, params : Tensor = None, params_bra: Tensor = None, engine: str = None) -> Tuple[float, Tensor]:
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
        params  = self._check_params_argument(params)
        hamiltonian, _, engine = self._get_hamiltonian_and_core(engine)
        e, g = get_energy_and_grad(hamiltonian=hamiltonian, n_qubits=self.n_qubits, n_elec_s=self.n_elec_s, engine=engine, mode=self.mode, 
                                   ex_ops=self.ex_ops , params=self.params ,params_bra=self.params_bra,param_ids=self.param_ids , 
                                   init_state=self.init_state , ex_ops_bra=self.ex_ops_bra, param_ids_bra=self.param_ids_bra, init_state_bra= self.init_state_bra) 
        return float(e + self.e_core), tc.backend.numpy(g)

    def _check_params_argument(self, params, strict=True):
        if params is None:
            if self.params is not None:
                params = self.params  + self.params_bra
            else:
                if strict:
                    raise ValueError("Run the `.kernel` method to determine the parameters first")
                else:
                    if self.init_guess  is not None and self.init_guess_bra is not None:
                        params = self.init_guess  + self.init_guess_bra
                    else:
                        params = np.zeros(self.n_params ) + np.zeros(self.n_params_bra)
        if len(params) != self.n_params  and len(params) != self.n_params_bra and len(params) != self.n_params_bra+self.n_params:
            raise ValueError(f"Incompatible parameter shape. {self.n_params_bra} , {self.n_params }  or {self.n_params_bra+self.n_params} is desired. Got {len(params)}")
        return tc.backend.convert_to_tensor(params).astype(tc.rdtypestr)   

    def _sanity_check(self):
        if self.ex_ops  is None or self.param_ids  is None:
            raise ValueError("`ex_ops ` or `param_ids ` not defined")
        if self.ex_ops_bra is None or self.param_ids_bra is None:
            raise ValueError("`ex_ops_bra` or `param_ids_bra` not defined")
        
        if self.param_ids_bra is not None and (len(self.ex_ops_bra) != len(self.param_ids_bra)):
            raise ValueError(
                f"Excitation operator size {len(self.ex_ops_bra)} and parameter size {len(self.param_ids_bra)} do not match"
            )
        if self.param_ids  is not None and (len(self.ex_ops ) != len(self.param_ids )):
            raise ValueError(
                f"Excitation operator size {len(self.ex_ops )} and parameter size {len(self.param_ids )} do not match"
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
            ex_ops  = self.ex_ops 
            param_ids = self.param_ids 
            init_state = self.init_state 
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
            ex_ops = self.ex_ops 
            param_ids = self.param_ids 
            init_state = self.init_state 
        else:
            ex_ops = self.ex_ops_bra
            param_ids = self.param_ids_bra
            init_state = self.init_state_bra
        statevector = get_statevector(
            params, self.n_qubits, self.n_elec_s, ex_ops, param_ids, self.mode, init_state, engine
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
            ex_ops = self.ex_ops 
            param_ids = self.param_ids 
            init_state = self.init_state 
        else:
            ex_ops = self.ex_ops_bra
            param_ids = self.param_ids_bra
            init_state = self.init_state_bra
        if ex_ops is None:
            raise ValueError("Excitation operators not defined")
        params = self._check_params_argument(params, strict=False)
        return get_circuit(params,self.n_qubits,self.n_elec_s,ex_ops,param_ids,self.mode,init_state,decompose_multicontrol=decompose_multicontrol,trotter=trotter)

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

    def get_init_state_dataframe(self, init_state, coeff_epsilon: float = DISCARD_EPS) -> pd.DataFrame:
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
            if self.init_state is None:
                init_state = get_init_civector(self.civector_size)
            else:
                init_state = self.init_state
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

    def get_excitation_dataframe(self,ket:bool=True) -> pd.DataFrame:
        '''
        ket:bool Whether to refered to the ket (True) or bra (False)
        '''
        columns = ["excitation", "configuration", "parameter", "initial guess"]
        if ket:
            if self.ex_ops  is None:
                return pd.DataFrame(columns=columns)

            if self.params  is None:
                # optimization not done
                params = [None] * len(self.init_guess)
            else:
                params = self.params 

            if self.param_ids is None:
                # see self.n_params
                param_ids = range(len(self.ex_ops ))
            else:
                param_ids = self.param_ids 
            ex_ops =self.ex_ops 
            init_guess = self.init_guess 
        else:
            if self.ex_ops_bra is None:
                return pd.DataFrame(columns=columns)

            if self.params_bra is None:
                # optimization not done
                params = [None] * len(self.init_guess_bra)
            else:
                params = self.params_bra

            if self.param_ids is None:
                # see self.n_params
                param_ids = range(len(self.ex_ops_bra))
            else:
                param_ids = self.param_ids_bra
            ex_ops =self.ex_ops_bra
            init_guess = self.init_guess_bra
            
        data_list = []

        for i, ex_op in zip(param_ids, ex_ops):
            bitstring = get_ex_bitstring(self.n_qubits, self.n_elec_s, ex_op, self.mode)
            data_list.append((ex_op, bitstring, params[i], init_guess[i]))
        return pd.DataFrame(data_list, columns=columns)

    def print_excitations(self):
        print('Bra: ',self.get_excitation_dataframe(ket=False).to_string())
        print('Ket: ',self.get_excitation_dataframe(ket=True).to_string())

    def print_ansatz(self):
        df_dict = {
            "#qubits": [self.n_qubits],
            "#params_ket": [self.n_params],
            "#params_bra": [self.n_params_bra],
            "#excitations_ket": [len(self.ex_ops)],
            "#excitations_bra": [len(self.ex_ops_bra)],
        }
        if self.init_state_bra is None:
            df_dict["initial condition bra"] = "RHF"
        else:
            df_dict["initial condition bra"] = "custom"

        if self.init_state is None:
            df_dict["initial condition ket"] = "RHF"
        else:
            df_dict["initial condition ket"] = "custom"
        print(pd.DataFrame(df_dict).to_string(index=False))

    def print_init_state_bra(self):
        print('Bra: ',self.get_init_state_dataframe(self.init_state_bra).to_string())
    
    def print_init_state_ket(self):
        print('Ket: ',self.get_init_state_dataframe(self.init_state).to_string())

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
        if self.init_state_bra is not None or self.init_state  is not None:
            print("############################ Initial Condition ########################")
            self.print_init_state_bra()
            self.print_init_state()
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
            return self._init_state 

    @init_state_bra.setter
    def init_state_bra(self, init_state_bra):
        self._init_state_bra = init_state_bra

    @property
    def ex_ops_bra(self) -> Tensor:
        """
        Excitation operators applied to the bra.
        """
        if self._ex_ops_bra is not None:
            return self._ex_ops_bra
        else: return self.ex_ops 

    @ex_ops_bra.setter
    def ex_ops_bra(self, ex_ops_bra):
        self._ex_ops_bra = ex_ops_bra

    @property
    def params_bra(self) -> Tensor:
        """The circuit parameters."""
        if self._params_bra is not None:
            return self._params_bra
        elif self.params  is not None:
            return self.params 
        elif self.opt_res is not None:
            return self.opt_res.x
        return None

    @params_bra.setter
    def params_bra(self, params_bra):
        self._params_bra = params_bra

    @property
    def param_ids_bra(self) -> List[int]:
        """The mapping from excitations operators to parameters."""
        if self._param_ids_bra is None:
            if self.ex_ops_bra is None:
                if self._param_ids  is not None:
                    return self.param_ids
                raise ValueError("Excitation operators not defined")
            else:
                return tuple(range(len(self.ex_ops_bra)))
        return self._param_ids_bra

    @param_ids_bra.setter
    def param_ids_bra(self, v):
        self._param_ids_bra = v

    @property
    def param_to_ex_ops_bra(self):
        if self.param_ids_bra is None and self.ex_ops_bra is None:
            return self.param_to_ex_ops 
        d = defaultdict(list)
        for i, j in enumerate(self.param_ids_bra):
            d[j].append(self.ex_ops_bra[i])
        return d
    
    @property
    def n_params_bra(self) -> int:
        """The number of parameter in the ansatz/circuit."""
        # this definition ensures that `param[param_id]` is always valid
        if self.param_ids_bra is None:
            return self.n_params 
        if not self.param_ids_bra:
            return 0
        return max(self.param_ids_bra) + 1