import os


def run_simulation():
    """
    OpenFOAM을 사용하여 시뮬레이션을 실행합니다.
    """
    setup_simulation_environment()
    clean_simulation()
    move_block_mesh_dict_and_control_dict()
    generate_mesh()
    remove_processor_directories()
    decompose_mesh()
    set_permissions()
    run_parallel_simulation()
    Cd, Cl = read_force_data()
    return Cd, Cl


def setup_simulation_environment():
    """
    시뮬레이션 디렉토리로 작업 디렉토리를 변경합니다.
    """
    simulation_directory = "~/OpenFOAM/daehwa-11/run/airfoil"
    os.chdir(os.path.expanduser(simulation_directory))


def clean_simulation():
    """
    시뮬레이션 디렉토리를 정리합니다.
    """
    os.system("sh ./Allclean")


def move_block_mesh_dict_and_control_dict():
    """
    blockMeshDict 파일을 적절한 위치로 이동합니다.
    """
    source_path = "~/Documents/3D-propeller-Design/blockMeshDict"
    destination_directory = "~/OpenFOAM/daehwa-11/run/airfoil/system"
    os.system(
        f"mv {os.path.expanduser(source_path)} {os.path.expanduser(destination_directory)}"
    )

    source_path = "~/Documents/3D-propeller-Design/controlDict"
    os.system(
        f"mv {os.path.expanduser(source_path)} {os.path.expanduser(destination_directory)}"
    )

    source_path = "~/Documents/3D-propeller-Design/U"
    destination_directory = "~/OpenFOAM/daehwa-11/run/airfoil/0"
    os.system(
        f"mv {os.path.expanduser(source_path)} {os.path.expanduser(destination_directory)}"
    )


def generate_mesh():
    """
    blockMesh를 사용하여 메시를 생성합니다.
    """
    os.system("blockMesh")


def set_permissions():
    """
    points 파일에 대한 권한을 설정합니다.
    """
    points_path = "~/OpenFOAM/daehwa-11/run/airfoil/constant/polyMesh/points"
    os.system(f"chmod 777 {os.path.expanduser(points_path)}")


def remove_processor_directories():
    """
    기존의 processor 디렉토리를 삭제합니다.
    """
    os.system("rm -rf processor*")


def decompose_mesh():
    """
    메시를 여러 부분으로 나누어 병렬 처리를 준비합니다.
    """
    os.system("decomposePar")


def run_parallel_simulation():
    """
    병렬로 시뮬레이션을 실행합니다.
    """
    os.system(
        "mpirun --oversubscribe -np 20 foamRun -solver incompressibleFluid -parallel"
    )
    os.system("reconstructPar")
    os.system("rm -rf processor*")


def read_force_data():
    """
    forceCoeffs.dat 파일을 읽어 마지막 줄의 시간, 항력 계수, 양력 계수를 반환합니다.
    """
    result_file_path = (
        "~/OpenFOAM/daehwa-11/run/airfoil/postProcessing/forceCoeffs/0/forceCoeffs.dat"
    )
    with open(os.path.expanduser(result_file_path), "r") as file:
        lines = file.readlines()

    last_line = lines[-1].split()
    Cd, Cl = float(last_line[2]), float(last_line[3])
    return Cd, Cl
