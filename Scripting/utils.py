import os
from typing import *
import numpy as np


def initialize_script(script_name: str, model: str) -> None:
    """
    func initializes script for simvascular onedsolver
    """
    with open(f"{script_name}.txt", "w") as f:
        f.write("#================================\n")
        f.write(f"#{model} - UNITS CGS\n")
        f.write("#================================\n")
        f.close()


def add_model_name(script_name: str, model: str) -> None:
    """
    func adds model name keyword to script
    """
    with open(f"{script_name}.txt", "a") as f:
        f.write(f"MODEL {model}\n")
        f.close()


def add_nodes(script_name: str, num_nodes: int) -> None:
    """
    Function ads NODE keywords to script

    for 1D simulations all nodes can be at 0.0, 0.0, 0.0
    """
    with open(f"{script_name}.txt", "a") as f:
        for i in range(num_nodes):
            f.write(f"NODE {i} {0.0} {0.0} {0.0}\n")
        f.close()


def add_segments(
    script_name: str,
    num_seg: int,
    names: list,
    #idx: list,
    L: list,
    Nx: list,
    start_nodes: list,
    end_nodes: list,
    area_inlet: list,
    area_outlet: list,
    initial_flow: float,
    material: str,
    ml_type: str,
    angle: float,
    uid: int,
    bid: int,
    bc_type: list,
    dname: list,
) -> None:
    """
    Function ads SEGMENT keywords to script

    SEGMENT name id length nelems inode onode iarea oarea\
          iflow material mltype angle uid bid bctype dname

    name (string) - Segment name.

    id (integer) - Segment ID.

    length (double - Segment length.

    nelems (integer) - Total finite elements in segment.

    inode (integer) - Segment inlet Node.

    onode (integer) - Segment outlet Node.

    iarea (double - Segment inlet area.

    oarea (double - Segment outlet area.

    iflow (double - Segment initial flow.

    material (string) - Segment material.

    mltype (string) - Minor loss type. (NONE or STENOSIS)

    angle (double) - Branch angle. (not used)

    uid (integer) - Upstream segment ID. (in cases of STENOSIS minor loss type)

    bid (integer) - Branch segment ID. (not used)

    bctype (string) - Outlet boundary condition type.

    dname (string) - Data Table Name for boundary condition.
    """
    with open(f"{script_name}.txt", "a") as f:
        for i in range(num_seg):
            f.write(
                f"SEGMENT {names[i]} {i} {L[i]} {Nx[i]} {start_nodes[i]} {end_nodes[i]} {area_inlet[i]} {area_outlet[i]} {initial_flow} {material} {ml_type} {angle} {uid} {bid} {bc_type[i]} {dname[i]}\n"
            )
        f.close()


def add_rcr_vlas(
    script_name: str, dname: list, outlet: list, R1: list, R2: list, CT: list
) -> None:
    """
    Function adds rcr vals datatable to script
    """
    with open(f"{script_name}.txt", "a") as f:
        for i in range(len(outlet)):
            if outlet[i] != 0:
                f.write(f"DATATABLE {dname[i]} LIST\n")
                f.write(f"0.0 {R1[i]}\n")
                f.write(f"0.0 {CT[i]}\n")
                f.write(f"0.0 {R2[i]}\n")
                f.write(f"ENDDATATABLE\n")
        f.close()


def add_inlet_bc(script_name: str, flow_file: str) -> None:
    """
    function adds inlet flow bc to script
    """
    with open(f"{script_name}.txt", "a") as f:
        f.write("DATATABLE PULS_FLOW LIST\n")
        with open(f"{flow_file}", "r") as q:
            lines = q.readlines()
            for line in lines:
                f.write(line)
        f.write("ENDDATATABLE\n")
        f.close()


def add_inlet_bc2(script_name: str, flow_matrix: np.ndarray) -> None:
    """
    Function adds inlet flow bc to script
    """
    with open(f"{script_name}.txt", "a") as f:
        f.write("DATATABLE PULS_FLOW LIST\n")
        for i in range(len(flow_matrix)):
            f.write(f"{flow_matrix[i,0]} {flow_matrix[i,1]}\n")
        f.write("ENDDATATABLE\n")
        f.close()


def add_material(
    script_name: str,
    name: str,
    type: str,
    density: float,
    viscosity: float,
    pressure: float,
    exponent: float,
    k1: float,
    k2: float,
    k3: float,
) -> None:
    """
    Function adds MATERIAL keyword to script
    """
    with open(f"{script_name}.txt", "a") as f:
        # Nie wiem czy to zadziala z linear do zobaczenia chociaz raczej nie bede tego uzywac
        f.write(
            f"MATERIAL {name} {type} {density} {viscosity} {pressure} {exponent} {k1} {k2} {k3}\n"
        )
        f.close()


def add_solver_options(
    script_name: str,
    timestep: float,
    savefreq: int,
    maxsteps: int,
    nquad: int,
    dname: str,
    bctype: str,
    tol: float,
    form: str,
    stab: str,
) -> None:
    """
    Function adds SOLVEROPTIONS keyword to script
    """
    with open(f"{script_name}.txt", "a") as f:
        f.write(
            f"SOLVEROPTIONS {timestep} {savefreq} {maxsteps} {nquad} {dname} {bctype} {tol} {form} {stab}\n"
        )
        f.close()


def add_output(script_name: str, format: str) -> None:
    """
    Function adds OUTPUT keyword to script
    """
    with open(f"{script_name}.txt", "a") as f:
        f.write(f"OUTPUT {format}\n")
        f.close()


def add_joints(
    script_name: str, p: list, d1: list, d2: list, merging: list, tn: list
) -> None:
    """
    Adds JOINT keyword to script
    """
    with open(f"{script_name}.txt", "a") as f:
        for i in range(len(p)):
            if merging[i] == 1:
                f.write(f"JOINT JOINT{i} {tn[d1[i]]} in{i} out{i}\n")
            else:
                f.write(f"JOINT JOINT{i} {tn[p[i]]} in{i} out{i}\n")
        f.close()


def add_joint_inlet(
    script_name: str, p: list, d1: list, d2: list, merging: list
) -> None:
    """
    Adds JOINTINLET keyword to script
    """
    with open(f"{script_name}.txt", "a") as f:
        for i in range(len(p)):
            if merging[i] == 1:
                f.write(f"JOINTINLET in{i} 2 {d1[i]} {d2[i]}\n")
            else:
                f.write(f"JOINTINLET in{i} 1 {p[i]}\n")
        f.close()


def add_joint_outlet(
    script_name: str, p: list, d1: list, d2: list, merging: list
) -> None:
    """
    Adds JOINTOUTLET keyword to script
    """
    with open(f"{script_name}.txt", "a") as f:
        for i in range(len(p)):
            if merging[i] == 1:
                f.write(f"JOINTOUTLET out{i} 1 {p[i]}\n")
            else:
                f.write(f"JOINTOUTLET out{i} 2 {d1[i]} {d2[i]}\n")
        f.close()
