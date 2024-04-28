from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes import SabreLayout


def sabre_pass_manager(pass_manager_config: PassManagerConfig):
    # gather
    coupling_map = pass_manager_config.coupling_map
    seed_transpiler = pass_manager_config.seed_transpiler
    assert coupling_map is not None

    # build layout & routing pass using sabre
    sabre = SabreLayout(
        coupling_map,
        max_iterations=4,
        seed=seed_transpiler,
        swap_trials=20,
        layout_trials=20,
        skip_routing=False,
    )

    return PassManager(sabre)
