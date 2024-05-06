import os

def run_simulation():
    """
    OpenFOAM을 사용하여 시뮬레이션을 실행합니다.
    """
    setup_simulation_environment()
    clean_simulation()
    move_block_mesh_dict()
    generate_mesh()
    decompose_mesh()
    run_parallel_simulation()
    Cd, Cl = read_force_data()
    return Cd, Cl


def setup_simulation_environment():
    """
    시뮬레이션 디렉토리로 작업 디렉토리를 변경합니다.
    """
    simulation_directory = "~/OpenFOAM/daehwa-11/run/airfoil"
    os.chdir(simulation_directory)


def clean_simulation():
    """
    시뮬레이션 디렉토리를 정리합니다.
    """
    os.system("sh ./Allclean")


def move_block_mesh_dict():
    """
    blockMeshDict 파일을 적절한 위치로 이동합니다.
    """
    source_path = "~/Documents/3D-propeller-Design/blockMeshDict"
    destination_directory = "~/OpenFOAM/daehwa-11/run/airfoil/system"
    os.system(f"cp {source_path} {destination_directory}")


def generate_mesh():
    """
    blockMesh를 사용하여 메시를 생성합니다.
    """
    os.system("blockMesh")


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
    result_file_path = "~/OpenFOAM/daehwa-11/run/airfoil/postProcessing/forceCoeffs/0/forceCoeffs.dat"
    with open(result_file_path, "r") as file:
        lines = file.readlines()

    last_line = lines[-1].split()
    Cd, Cl = float(last_line[2]), float(last_line[3])
    return Cd, Cl