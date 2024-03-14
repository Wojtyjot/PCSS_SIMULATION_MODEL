import pandas as pd
import numpy as np
from typing import *

# from utils import *

# ZAŁOŻENIE dane w formie padas dtataframe i na tej podstawie tworzymy skrypt
# TRZEBA ZAPISYWAC TE CSV BY BYŁO MOZNA UŻYC JAKO FEAT W GNOT
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
    # idx: list,
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
                f"SEGMENT {names[i]} {i} {L[i]} {Nx} {start_nodes[i]} {end_nodes[i]} {area_inlet[i]} {area_outlet[i]} {initial_flow} {material} {ml_type} {angle} {uid} {bid} {bc_type[i]} {dname[i]}\n"
            )
        f.close()


def add_segment(
    script_name: str,
    name: str,
    id: int,
    L: float,
    Nx: int,
    inode: int,
    onode: int,
    iarea: float,
    oarea: float,
    iflow: float,
    material: str,
    mltype: str,
    angle: float,
    uid: int,
    bid: int,
    bctype: str,
    dname: str,
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

    oarea (double) - Segment outlet area.

    iflow (double) - Segment initial flow.

    material (string) - Segment material.

    mltype (string) - Minor loss type. (NONE or STENOSIS)

    angle (double) - Branch angle. (not used)

    uid (integer) - Upstream segment ID. (in cases of STENOSIS minor loss type)

    bid (integer) - Branch segment ID. (not used)

    bctype (string) - Outlet boundary condition type.

    dname (string) - Data Table Name for boundary condition.
    """
    with open(f"{script_name}.txt", "a") as f:
        f.write(
            f"SEGMENT {name} {id} {L} {Nx} {inode} {onode} {iarea} {oarea} {iflow} {material} {mltype} {angle} {uid} {bid} {bctype} {dname}\n"
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
    k2: Optional[float] = None,
    k3: Optional[float] = None,
) -> None:
    """
    Function adds MATERIAL keyword to script
    """
    with open(f"{script_name}.txt", "a") as f:
        # Nie wiem czy to zadziala z linear do zobaczenia chociaz raczej nie bede tego uzywac
        if type == "LINEAR":
            f.write(
                f"MATERIAL {name} {type} {density} {viscosity} {pressure} {exponent} {k1}\n"
            )
        elif type == "OLUFSEN":
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
                f.write(f"JOINT JOINT{i} {tn[d1[i]]} IN{i} OUT{i}\n")
            else:
                f.write(f"JOINT JOINT{i} {tn[p[i]]} IN{i} OUT{i}\n")
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
                f.write(f"JOINTINLET IN{i} 2 {d1[i]} {d2[i]}\n")
            else:
                f.write(f"JOINTINLET IN{i} 1 {p[i]}\n")
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
                f.write(f"JOINTOUTLET OUT{i} 1 {p[i]}\n")
            else:
                f.write(f"JOINTOUTLET OUT{i} 2 {d1[i]} {d2[i]}\n")
        f.close()


def compute_k1(df: pd.DataFrame, id: int) -> float:
    """
    Function computes k1 coeficient for linear material model

    Eh/r0 = k1
    """
    print(df[df.id == id].name)
    print(f"modulus: {df[df.id == id].modulus.iloc[0]}")
    print(f"thickness: {df[df.id == id].thickness.iloc[0]}")
    print(f"r0_in: {df[df.id == id].r0_in.iloc[0]}")
    print(
        f"eh/r = {df[df.id == id].modulus.iloc[0] * df[df.id == id].thickness.iloc[0] / df[df.id == id].r0_in.iloc[0]}"
    )
    return (df[df.id == id].modulus.iloc[0] * df[df.id == id].thickness.iloc[0]) / df[
        df.id == id
    ].r0_in.iloc[0]


def create_script(
    df: pd.DataFrame,
    df_joints: pd.DataFrame,
    script_name: str,
    model: str,
    flow: np.ndarray,
    olufsen: bool = False,
    max_steps: int = 10000,
    dt: float = 0.001,
) -> None:
    """
    Function creates script for simvascular onedsolver from pandas dataframe

    Args:
        df: [DataFrame] pandas dataframe with arteries
        script_name: [str] name of the script
        model: [str] name of the model


    returns nothing
    """
    # nx bedą stałymi = 1000
    initialize_script(script_name, model)
    add_model_name(script_name, model)
    # Quick hack
    if olufsen:
        add_material(
            script_name=script_name,
            name="MAT",
            type="OLUFSEN",
            density=1.06,
            viscosity=0.04,
            pressure=99991.77631578947,  # 75mmHg
            exponent=2.0,  # polynomial order of flow 2.0 = parabolic
            k1=2e7,  # Ba
            k2=-22.53,  # cm^-1
            k3=8.65e5,  # Ba
        )
        add_segments(
            script_name=script_name,
            num_seg=len(df),
            names=df.name,
            # root_node=df.root_node,
            L=df.L,
            Nx=50,
            start_nodes=df.sn,
            end_nodes=df.tn,
            area_inlet=df.area_inlet,
            area_outlet=df.area_outlet,
            initial_flow=0.0,
            material="MAT",
            ml_type="NONE",
            angle=0.0,
            uid=0,
            bid=0,
            bc_type=df.bc_type,
            dname=df.dname,
        )

    else:
        for id in df.id:
            k1 = compute_k1(df, id)
            print(k1)
            add_material(
                script_name=script_name,
                name=f"MAT{id}",
                type="LINEAR",
                density=1.06,
                viscosity=0.04,
                pressure=0.0,  # 99991.77631578947,  # 75mmHg
                exponent=2.0,  # polynomial order of flow 2.0 = parabolic
                k1=k1,
            )
            add_segment(
                script_name=script_name,
                name=df[df.id == id].name.iloc[0],
                id=id,
                L=df[df.id == id].L.iloc[0],
                Nx=100,
                inode=df[df.id == id].sn.iloc[0],
                onode=df[df.id == id].tn.iloc[0],
                iarea=df[df.id == id].area_inlet.iloc[0],
                oarea=df[df.id == id].area_outlet.iloc[0],
                iflow=0.0,
                material=f"MAT{id}",
                mltype="NONE",
                angle=0.0,
                uid=0,
                bid=0,
                bctype=df[df.id == id].bc_type.iloc[0],
                dname=df[df.id == id].dname.iloc[0],
            )

    # Pytanie czy dodać inny material dla naczyn\COW
    add_nodes(script_name, len(df.node_list.dropna()))
    # dodawanie matertialóœ do segmentu ogarnac

    add_rcr_vlas(script_name, df.dname, df.outlet, df.R1, df.R2, df.C)
    add_inlet_bc2(script_name, flow)
    add_joints(
        script_name=script_name,
        p=df_joints.p,
        d1=df_joints.d1,
        d2=df_joints.d2,
        merging=df_joints.merging,
        tn=df.tn,
    )
    add_joint_inlet(
        script_name=script_name,
        p=df_joints.p,
        d1=df_joints.d1,
        d2=df_joints.d2,
        merging=df_joints.merging,
    )
    add_joint_outlet(
        script_name=script_name,
        p=df_joints.p,
        d1=df_joints.d1,
        d2=df_joints.d2,
        merging=df_joints.merging,
    )

    add_output(script_name=script_name, format="TEXT")
    add_solver_options(
        script_name=script_name,
        timestep=dt,
        savefreq=10,
        maxsteps=max_steps, # zmienic na 10 cykli 
        nquad=4,
        dname="PULS_FLOW",
        bctype="FLOW",
        tol=1e-8,
        form=1,
        stab=1,
    )
